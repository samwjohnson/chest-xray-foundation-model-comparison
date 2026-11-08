# Nested repeated patient-grouped CV, verified frozen embeddings only.
assert ALIGNMENT_PASSED
import json,time,uuid,copy,collections
from datetime import datetime
from zoneinfo import ZoneInfo
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
threadpool_limits(1);torch.set_num_threads(2)
start_cv=time.time();runid=datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
base='/content/drive/MyDrive/fm_emb_sweep/nested_cv_'+runid
foldfile=base+'_folds.jsonl';summaryfile=base+'_summary.json'
# Exclusive creation guarantees no prior Drive output is overwritten.
fh=open(foldfile,'x')
CGRID=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,.1]
GRID=[(lr,p) for lr in [1e-4,1e-3,3e-3] for p in [.1,.3,.5]]
design=dict(outer='StratifiedGroupKFold 5 folds, repeats seeds 42..46',inner='StratifiedGroupKFold 3 folds on outer training only',C=CGRID,mlp_lr=[1e-4,1e-3,3e-3],mlp_dropout=[.1,.3,.5],mlp_seeds=[0,1,2],inner_seed=0,max_epochs=60,patience=5,optimizer='AdamW',weight_decay=1e-4,batch=256,outer_early_stop='first fold of 7-fold StratifiedGroupKFold on outer train (~14.3%)',rows=len(y),patients=len(set(P)),frozen=True)
print('DESIGN',json.dumps(design),flush=True);print('OUTPUTS',foldfile,summaryfile,flush=True)
def splitcheck(a,b):assert not set(P[a])&set(P[b])
def fitmlp(X,fit,valid,test,lr,p,seed):
 splitcheck(fit,valid);splitcheck(fit,test)
 torch.manual_seed(seed);np.random.seed(seed)
 sc=StandardScaler().fit(X[fit]);T=lambda idx:torch.tensor(sc.transform(X[idx]),dtype=torch.float32,device='cuda')
 xt,xv,xe=T(fit),T(valid),T(test);yt=torch.tensor(y[fit],dtype=torch.float32,device='cuda')
 net=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Dropout(p),nn.Linear(256,64),nn.ReLU(),nn.Dropout(p),nn.Linear(64,1)).cuda()
 opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
 loss=nn.BCEWithLogitsLoss(pos_weight=(yt==0).sum()/(yt==1).sum());best=-1;bestep=0;weights=None
 g=torch.Generator(device='cuda').manual_seed(seed)
 for ep in range(1,61):
  net.train()
  for ix in torch.randperm(len(xt),device='cuda',generator=g).split(256):
   opt.zero_grad(set_to_none=True);loss(net(xt[ix]).flatten(),yt[ix]).backward();opt.step()
  net.eval()
  with torch.inference_mode():a=roc_auc_score(y[valid],torch.sigmoid(net(xv)).flatten().cpu().numpy())
  if a>best:best=a;bestep=ep;weights=copy.deepcopy(net.state_dict())
  elif ep-bestep>=5:break
 net.load_state_dict(weights);net.eval()
 with torch.inference_mode():score=torch.sigmoid(net(xe)).flatten().cpu().numpy()
 return float(roc_auc_score(y[test],score)),dict(best_epoch=bestep,stopped_epoch=ep,val_auc=float(best))
records=[];ratios=[]
for repeat,seed in enumerate(range(42,47)):
 outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
 for fold,(otr,ote) in enumerate(outer.split(np.zeros(len(y)),y,P)):
  splitcheck(otr,ote);ratios.append(len(ote)/len(otr))
  inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=1000+repeat*5+fold).split(otr,y[otr],P[otr]))
  ef,evs=next(StratifiedGroupKFold(7,shuffle=True,random_state=2000+repeat*5+fold).split(otr,y[otr],P[otr]));ef,evs=otr[ef],otr[evs]
  print('OUTER_START',repeat,fold,'train',len(otr),'test',len(ote),flush=True)
  for model,X in E.items():
   c_scores=[]
   for C in CGRID:
    aucs=[]
    for a,b in inner:
     a,b=otr[a],otr[b];splitcheck(a,b)
     sc=StandardScaler().fit(X[a]);clf=LogisticRegression(C=C,class_weight='balanced',max_iter=5000).fit(sc.transform(X[a]),y[a])
     aucs.append(roc_auc_score(y[b],clf.predict_proba(sc.transform(X[b]))[:,1]))
    c_scores.append(float(np.mean(aucs)))
   C=CGRID[int(np.argmax(c_scores))];sc=StandardScaler().fit(X[otr]);clf=LogisticRegression(C=C,class_weight='balanced',max_iter=5000).fit(sc.transform(X[otr]),y[otr])
   auc=float(roc_auc_score(y[ote],clf.predict_proba(sc.transform(X[ote]))[:,1]))
   rec=dict(repeat=repeat,fold=fold,model=model,head='probe',auc=auc,C=C,inner_aucs=c_scores,ntrain=len(otr),ntest=len(ote))
   records.append(rec);fh.write(json.dumps(rec)+'\n');fh.flush();print('FOLD_RESULT',json.dumps(rec),flush=True)
   cfgscores=[]
   for lr,p in GRID:
    vals=[]
    for a,b in inner:
     a,b=otr[a],otr[b];au,info=fitmlp(X,a,b,b,lr,p,0);vals.append(au)
    cfgscores.append(float(np.mean(vals)))
    print('MLP_TUNE',repeat,fold,model,'lr',lr,'dropout',p,'inner_auc',cfgscores[-1],flush=True)
   lr,p=GRID[int(np.argmax(cfgscores))];aucs=[];infos=[]
   for s in [0,1,2]:
    au,inf=fitmlp(X,ef,evs,ote,lr,p,s);aucs.append(au);infos.append(inf)
   rec=dict(repeat=repeat,fold=fold,model=model,head='MLP',auc=float(np.mean(aucs)),seed_aucs=aucs,seed_sd=float(np.std(aucs,ddof=1)),lr=lr,dropout=p,inner_aucs=cfgscores,epochs=infos,ntrain=len(otr),ntest=len(ote),early_stop_n=len(evs))
   records.append(rec);fh.write(json.dumps(rec)+'\n');fh.flush();print('FOLD_RESULT',json.dumps(rec),flush=True)
  print('PROGRESS',len(records),'of 100 records','minutes',round((time.time()-start_cv)/60,2),flush=True)
fh.close()
def stats(a):
 a=np.asarray(a);sd=float(a.std(ddof=1));se=np.sqrt((1/len(a)+np.mean(ratios))*sd**2);margin=float(student_t.ppf(.975,len(a)-1)*se)
 return dict(mean=float(a.mean()),fold_sd=sd,corrected_t95=[float(a.mean()-margin),float(a.mean()+margin)],fold_percentiles_2_5_97_5=np.percentile(a,[2.5,97.5]).tolist())
summary=dict(design=design,files=[foldfile,summaryfile],runtime_seconds=time.time()-start_cv,ci_note='Nadeau-Bengio approximate corrected repeated-CV t interval: df=24, variance multiplier=1/25+mean(ntest/ntrain); fold percentiles are descriptive, not a mean CI',models={},paired={})
for model in E:
 for head in ['probe','MLP']:
  rs=[r for r in records if r['model']==model and r['head']==head];a=[r['auc'] for r in rs];st=stats(a)
  st['repeat_means']=[float(np.mean([r['auc'] for r in rs if r['repeat']==i])) for i in range(5)];st['repeat_mean_sd']=float(np.std(st['repeat_means'],ddof=1))
  st['selected_C']=dict(collections.Counter(str(r['C']) for r in rs)) if head=='probe' else None
  if head=='MLP':
   st['selected_lr_dropout']=dict(collections.Counter(str((r['lr'],r['dropout'])) for r in rs));st['seed_sd_mean']=float(np.mean([r['seed_sd'] for r in rs]));st['seed_sd_range']=[float(min(r['seed_sd'] for r in rs)),float(max(r['seed_sd'] for r in rs))]
  old={'RAD-DINO':.760,'CheXFound':.755}[model];post={'RAD-DINO':.7578,'CheXFound':.7592}[model]
  st.update(delta_vs_oct27_probe=st['mean']-old,delta_vs_oct31_probe=st['mean']-post)
  summary['models'][model+' '+head]=st
for head in ['probe','MLP']:
 a=[r['auc'] for r in records if r['model']=='CheXFound' and r['head']==head];b=[r['auc'] for r in records if r['model']=='RAD-DINO' and r['head']==head]
 summary['paired'][head]=stats(np.array(a)-np.array(b))
with open(summaryfile,'x') as f:json.dump(summary,f,indent=2)
print('NESTED_CV_SUMMARY',json.dumps(summary),flush=True);print('NESTED_CV_DONE',flush=True)
