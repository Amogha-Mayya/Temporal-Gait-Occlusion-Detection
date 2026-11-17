import sys, os, glob, argparse, random, pickle, yaml
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/sustech1k.yaml')
ap.add_argument('--ckpt', default='checkpoints/best_model_4heads.pth')
ap.add_argument('--occ_root', required=True)
ap.add_argument('--n', type=int, default=300)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = yaml.safe_load(open(a.config))['model']
from models.model import OcclusionDetectionModel
model = OcclusionDetectionModel(backbone_name=cfg.get('backbone', 'resnet18'), pretrained=False,
    transformer_cfg=cfg.get('transformer', {}), head_hidden=cfg.get('heads', {}).get('hidden_dim', 128),
    enable_extra_heads=True)
ck = torch.load(a.ckpt, map_location='cpu')
state = ck.get('model', ck.get('model_state_dict', ck))
miss, unexp = model.load_state_dict(state, strict=False)
print('checkpoint loaded: missing', len(miss), 'unexpected', len(unexp))
model.to(dev).eval()
W = 30

@torch.no_grad()
def detect(seq):
    T = seq.shape[0]
    x = torch.from_numpy(seq[:, :, 10:54].astype(np.float32) / 255.0).unsqueeze(1)
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    if T < W:
        x = torch.cat([x, x[-1:].repeat(W - T, 1, 1, 1)])
    n = x.shape[0]
    starts = list(range(0, n - W + 1, W))
    if starts[-1] + W < n:
        starts.append(n - W)
    prob, sev, cnt = torch.zeros(n), torch.zeros(n), torch.zeros(n)
    for s in starts:
        out = model(x[s:s + W].unsqueeze(0).to(dev))
        prob[s:s + W] += torch.sigmoid(out[0][0]).cpu()
        sev[s:s + W] += out[1][0].cpu()
        cnt[s:s + W] += 1
    return (prob / cnt)[:T].numpy(), (sev / cnt)[:T].numpy()

files = sorted(glob.glob(os.path.join(a.occ_root, '*', '*', '*', '*.pkl')))
files = [f for f in files if f.split('/')[-3] in ('nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02')]
random.seed(0)
files = random.sample(files, min(a.n, len(files)))
P, L, SP, ST = [], [], [], []
for f in files:
    parts = f.split('/')
    m = np.load(os.path.join(a.occ_root + '_meta', parts[-4], parts[-3], parts[-2] + '.npz'))
    p, s = detect(np.asarray(pickle.load(open(f, 'rb'))))
    P.append(p); L.append(m['label']); SP.append(s); ST.append(m['severity'])
P, L, SP, ST = [np.concatenate(v) for v in (P, L, SP, ST)]
pred = P > 0.5
tp = (pred & (L == 1)).sum(); fp = (pred & (L == 0)).sum()
fn = (~pred & (L == 1)).sum(); tn = (~pred & (L == 0)).sum()
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
order = np.argsort(P); ranks = np.empty(len(P)); ranks[order] = np.arange(1, len(P) + 1)
n1, n0 = (L == 1).sum(), (L == 0).sum()
auc = (ranks[L == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
print('sequences %d | frames %d | truly occluded %.3f' % (len(files), len(L), L.mean()))
print('accuracy %.4f | precision %.4f | recall %.4f | F1 %.4f | AUC %.4f' % ((tp + tn) / len(L), prec, rec, f1, auc))
print('mean detector score: occluded frames %.3f | clean frames %.3f' % (P[L == 1].mean(), P[L == 0].mean()))
print('severity MAE on occluded frames: %.4f' % np.abs(SP[L == 1] - ST[L == 1]).mean())
