"""Linear probe (and MLP head) on frozen embeddings, patient-wise 80/20 split; produces the reported AUROCs.

Adapted from notebook cell 8a. C (probe) / early stopping (MLP) and all thresholds are chosen on
the validation slice only; the test set is scored once.
"""
import os, copy, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from data import load_tables, split_8020

SRC = os.environ.get('EMB_DIR', './fm_emb')
MODELS = {'RAD-DINO': 'RAD_DINO_ViT_B_14_518', 'CheXFound': 'CheXFound_ViT_L_16_512'}

files, tr, ev = load_tables()
pool, y, P, ft, va, te = split_8020(tr, ev)
E = {}
for k, fn in MODELS.items():
    a = np.load(f'{SRC}/{fn}_train.npy'); b = np.load(f'{SRC}/{fn}_test.npy')
    assert a.shape[0] == len(tr) and b.shape[0] == len(ev), 'embedding rows do not match tr/ev'
    E[k] = np.concatenate([a, b])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def probe(X):
    best = None
    for C in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
        sc = StandardScaler().fit(X[ft]); clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(sc.transform(X[ft]), y[ft])
        sv = clf.predict_proba(sc.transform(X[va]))[:, 1]; a = roc_auc_score(y[va], sv)
        if best is None or a > best[0]: best = (a, C, sv, clf.predict_proba(sc.transform(X[te]))[:, 1])
    return best[2], best[3], f'C={best[1]}'


def mlp(X, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); sc = StandardScaler().fit(X[ft])
    T = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32, device=dev)
    Xt, Xv, Xe = T(X[ft]), T(X[va]), T(X[te]); yt = torch.tensor(y[ft], dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=(yt == 0).sum() / (yt == 1).sum()); g = torch.Generator(device=dev).manual_seed(seed); best = (-1, 0, None)
    for ep in range(1, 201):
        net.train()
        for idx in torch.randperm(len(Xt), device=dev, generator=g).split(256):
            opt.zero_grad(); lossf(net(Xt[idx]).squeeze(1), yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): a = roc_auc_score(y[va], torch.sigmoid(net(Xv)).squeeze(1).cpu().numpy())
        if a > best[0]: best = (a, ep, copy.deepcopy(net.state_dict()))
        elif ep - best[1] >= 15: break
    net.load_state_dict(best[2]); net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xv)).squeeze(1).cpu().numpy(), torch.sigmoid(net(Xe)).squeeze(1).cpu().numpy(), f'best_epoch={best[1]}'


def ss(s, yy, t):
    p = s >= t; return (p & (yy == 1)).sum() / (yy == 1).sum(), (~p & (yy == 0)).sum() / (yy == 0).sum(), (p.astype(int) == yy).mean()


TH = np.round(np.arange(0, 1.0001, 0.01), 2); rows = []
for k in MODELS:
    for head, f in [('probe', probe), ('MLP', mlp)]:
        sv, se, info = f(E[k]); yv, yte = y[va], y[te]
        tj = TH[int(np.argmax([sum(ss(sv, yv, t)[:2]) - 1 for t in TH]))]           # Youden J on validation
        th = max(t for t in TH if ss(sv, yv, t)[0] >= 0.90)                          # highest val threshold with sens >= 0.90
        r = dict(model=k, head=head, info=info, val_auroc=roc_auc_score(yv, sv), test_auroc=roc_auc_score(yte, se), youden_t=tj, hs90_t=th)
        r['youden_test_sens'], r['youden_test_spec'], r['youden_test_acc'] = ss(se, yte, tj)
        r['hs90_test_sens'], r['hs90_test_spec'], r['hs90_test_acc'] = ss(se, yte, th)
        rows.append(r); print(k, head, info, 'test AUROC', round(r['test_auroc'], 4))
R = pd.DataFrame(rows); R.to_csv('resplit_8020_results.csv', index=False); print(R.round(3).to_string())
