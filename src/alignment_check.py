"""Row-alignment spot check: re-extract embeddings for a few rows from the original images and
compare them (cosine similarity) with the saved embedding file rows.

Written for this repository to document the check described in the README; the Oct 31, 2026
check used rows 0, 4, 6000, 12662, 13000 of the pooled (train + test) order.
"""
import os, sys, numpy as np, cv2, pandas as pd
from data import load_tables
from extract_embeddings import embed, rad_dino, chexfound

SRC = os.environ.get('EMB_DIR', './fm_emb'); ROWS = [0, 4, 6000, 12662, 13000]
files, tr, ev = load_tables(); pool = pd.concat([tr, ev]).reset_index(drop=True)
for build in [rad_dino, chexfound]:
    name, f, res, ms, bs = build()
    saved = np.concatenate([np.load(f'{SRC}/{name}_train.npy'), np.load(f'{SRC}/{name}_test.npy')])
    X = np.stack([cv2.resize(cv2.imread(files[pool['Image Index'][r]], cv2.IMREAD_GRAYSCALE), (518, 518), interpolation=cv2.INTER_AREA) for r in ROWS])
    fresh = embed(f, X, res, ms, bs=len(ROWS))
    cos = (fresh * saved[ROWS]).sum(1) / (np.linalg.norm(fresh, axis=1) * np.linalg.norm(saved[ROWS], axis=1))
    print(name, dict(zip(ROWS, np.round(cos, 4))))
