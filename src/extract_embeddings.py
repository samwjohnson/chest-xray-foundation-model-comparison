"""Frozen-embedding extraction for RAD-DINO and CheXFound (GPU required).

Adapted from the original Colab notebook. Images are cached at 518 px from the original
1024 px PNGs, replicated to 3 channels, resized (bicubic) to each model's input size, normalized.
Embedding = concat[CLS token, mean of patch tokens]. Saved as <name>_train.npy / <name>_test.npy,
with rows in the exact order of tr / ev from data.load_tables().
"""
import os, sys, subprocess, numpy as np, cv2, torch, torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from data import load_tables

OUT = os.environ.get('EMB_DIR', './fm_emb')
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
RDN = ((0.5307,) * 3, (0.2583,) * 3)  # RAD-DINO preprocessor_config mean/std


@torch.no_grad()
def embed(fwd, X, res, mean_std, bs=64):
    mean = torch.tensor(mean_std[0], device='cuda').view(1, 3, 1, 1); std = torch.tensor(mean_std[1], device='cuda').view(1, 3, 1, 1); out = []
    for k in range(0, len(X), bs):
        x = torch.from_numpy(X[k:k + bs]).cuda().float().div(255).unsqueeze(1)
        if res != x.shape[-1]: x = F.interpolate(x, size=(res, res), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
        x = (x.repeat(1, 3, 1, 1) - mean) / std
        with torch.autocast('cuda', dtype=torch.float16): out.append(fwd(x).float().cpu())
    return torch.cat(out).numpy()


def rad_dino():
    from transformers import AutoModel
    m = AutoModel.from_pretrained('microsoft/rad-dino').cuda().eval().half()
    def f(x):
        h = m(pixel_values=x.half()).last_hidden_state; return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
    return 'RAD_DINO_ViT_B_14_518', f, 518, RDN, 32


def chexfound():
    os.environ['XFORMERS_DISABLED'] = '1'
    if not os.path.exists('CheXFound'): subprocess.run('git clone -q --depth 1 https://github.com/RPIDIAL/CheXFound', shell=True)
    sys.path.insert(0, 'CheXFound')
    from chexfound.models import vision_transformer as vits
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    m = vits.vit_large(patch_size=16, img_size=512, init_values=1.0, ffn_layer='swiglufused', block_chunks=0, num_register_tokens=4)
    sd = {k.removeprefix('model.').replace('.ls1.weight', '.ls1.gamma').replace('.ls2.weight', '.ls2.gamma'): v
          for k, v in load_file(hf_hub_download('DIAL-RPI/CheXFound', 'model.safetensors')).items()}
    m.load_state_dict(sd, strict=True); m = m.cuda().eval().half()
    def f(x):
        o = m.forward_features(x.half()); return torch.cat([o['x_norm_clstoken'], o['x_norm_patchtokens'].mean(1)], 1)
    return 'CheXFound_ViT_L_16_512', f, 512, IMNET, 16


if __name__ == '__main__':
    files, tr, ev = load_tables()
    load518 = lambda n: cv2.resize(cv2.imread(files[n], cv2.IMREAD_GRAYSCALE), (518, 518), interpolation=cv2.INTER_AREA)
    with ThreadPoolExecutor(8) as ex:
        Ctr = np.stack(list(ex.map(load518, tr['Image Index']))); Cev = np.stack(list(ex.map(load518, ev['Image Index'])))
    os.makedirs(OUT, exist_ok=True)
    for build in [rad_dino, chexfound]:
        name, f, res, ms, bs = build()
        Etr = embed(f, Ctr, res, ms, bs); Eev = embed(f, Cev, res, ms, bs)
        assert Etr.shape[0] == len(tr) and Eev.shape[0] == len(ev)
        np.save(f'{OUT}/{name}_train.npy', Etr); np.save(f'{OUT}/{name}_test.npy', Eev); print(name, Etr.shape, Eev.shape)
        torch.cuda.empty_cache()
