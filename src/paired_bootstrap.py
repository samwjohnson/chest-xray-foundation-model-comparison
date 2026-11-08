"""Paired patient-level bootstrap of the test-AUROC difference, RAD-DINO minus CheXFound (linear probes).

Adapted from the original notebook (cell 9b). Patients are resampled with replacement (2,000 reps);
both models are scored on the same resampled images each rep.
"""
import os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from data import load_tables, split_8020

SRC = os.environ.get('EMB_DIR', './fm_emb')
MODELS = {'RAD-DINO': 'RAD_DINO_ViT_B_14_518', 'CheXFound': 'CheXFound_ViT_L_16_512'}
files, tr, ev = load_tables()
pool, y, P, ft, va, te = split_8020(tr, ev)
S = {}
for k, fn in MODELS.items():
    X = np.concatenate([np.load(f'{SRC}/{fn}_train.npy'), np.load(f'{SRC}/{fn}_test.npy')]); assert X.shape[0] == len(pool)
    best = None
    for C in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
        sc = StandardScaler().fit(X[ft]); clf = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit(sc.transform(X[ft]), y[ft])
        a = roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
        if best is None or a > best[0]: best = (a, C, clf.predict_proba(sc.transform(X[te]))[:, 1])
    S[k] = best[2]; print(k, 'C', best[1], 'test AUROC', round(roc_auc_score(y[te], S[k]), 4))
yt, Pt = y[te], P[te]; a, b = S['RAD-DINO'], S['CheXFound']
obs = roc_auc_score(yt, a) - roc_auc_score(yt, b)
up = np.unique(Pt); grp = {p: np.where(Pt == p)[0] for p in up}; br = np.random.default_rng(123); D = []
for _ in range(2000):
    idx = np.concatenate([grp[p] for p in br.choice(up, len(up), replace=True)])
    if yt[idx].min() == yt[idx].max(): continue
    D.append(roc_auc_score(yt[idx], a[idx]) - roc_auc_score(yt[idx], b[idx]))
D = np.array(D); lo, hi = np.percentile(D, [2.5, 97.5])
print(f'observed diff {obs:.4f} | 95% CI [{lo:.4f}, {hi:.4f}] | crosses zero: {lo < 0 < hi} | reps {len(D)}')
