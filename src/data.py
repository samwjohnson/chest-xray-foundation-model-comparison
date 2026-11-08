"""Download NIH ChestX-ray14 (image parts 01-02 + labels CSV) and build the patient-level splits.

Adapted from the original Colab notebook (setup + label cells). Labels:
    y = 0  if 'Finding Labels' == 'No Finding'   (normal)
    y = 1  otherwise, i.e. any of the 14 findings (abnormal)
"""
import os, glob, tarfile
import numpy as np, pandas as pd

ROOT = os.environ.get('CXR_ROOT', './cxr_data')
# Official NIH Box links (from NIH's batch_download_zips.py)
PARTS = {'images_01.tar.gz': 'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
         'images_02.tar.gz': 'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz'}
CSV_URL = 'https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017_v2020.csv'


def download():
    os.makedirs(ROOT, exist_ok=True)
    for f, u in PARTS.items():
        p = os.path.join(ROOT, f)
        if not os.path.exists(p): os.system(f'wget -q -O "{p}" "{u}"')
    csv = os.path.join(ROOT, 'Data_Entry.csv')
    if not os.path.exists(csv): os.system(f'wget -q -O "{csv}" "{CSV_URL}"')
    if not os.path.isdir(os.path.join(ROOT, 'images')):
        for f in PARTS:
            with tarfile.open(os.path.join(ROOT, f)) as t: t.extractall(ROOT)


def load_tables():
    """Returns files (name->path), tr (train table), ev (500-image held-out table).

    Row order of tr/ev defines the row order of every saved embedding file.
    """
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(ROOT, 'images', '*.png'))}
    df = pd.read_csv(os.path.join(ROOT, 'Data_Entry.csv'))
    df = df[df['Image Index'].isin(files)].reset_index(drop=True)
    df['y'] = (df['Finding Labels'] != 'No Finding').astype(int)
    rng = np.random.default_rng(0); pats = df['Patient ID'].unique(); rng.shuffle(pats)
    held = set(pats[:int(0.15 * len(pats))]); hd = df[df['Patient ID'].isin(held)]
    ev = pd.concat([hd[hd.y == 0].sample(250, random_state=0), hd[hd.y == 1].sample(250, random_state=0)]) \
           .sample(frac=1, random_state=0).reset_index(drop=True)
    tr = df[~df['Patient ID'].isin(held)].reset_index(drop=True)
    assert not set(tr['Patient ID']) & set(ev['Patient ID'])
    return files, tr, ev


def split_8020(tr, ev):
    """Pool tr+ev and make the patient-wise 80/20 split used for the reported results.

    Returns pool, y, P (patient ids), and boolean masks ft (fit), va (validation), te (test).
    Validation = 15% of the training patients; used only for C / early-stopping / thresholds.
    """
    pool = pd.concat([tr, ev]).reset_index(drop=True); y = pool.y.values.astype(int); P = pool['Patient ID'].values
    assert pool['Image Index'].is_unique
    rng = np.random.default_rng(0); pats = np.unique(P); rng.shuffle(pats); cnt = pool['Patient ID'].value_counts()
    tp, n = [], 0
    for p in pats:
        if n >= 0.2 * len(pool): break
        tp.append(p); n += cnt[p]
    TEP = set(tp); te = np.isin(P, list(TEP)); trp = np.array([p for p in pats if p not in TEP])
    VP = set(trp[:int(round(0.15 * len(trp)))]); va = (~te) & np.isin(P, list(VP)); ft = (~te) & (~va)
    assert not set(P[te]) & set(P[~te]) and not set(P[va]) & set(P[ft])
    return pool, y, P, ft, va, te


if __name__ == '__main__':
    download(); files, tr, ev = load_tables()
    pool, y, P, ft, va, te = split_8020(tr, ev)
    for nm, m in [('fit', ft), ('val', va), ('test', te)]:
        print(f'{nm}: n={m.sum()} abnormal_frac={y[m].mean():.4f} patients={len(set(P[m]))}')
