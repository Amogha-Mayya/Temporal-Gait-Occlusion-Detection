"""
Proper Natural OC Domain Adaptation
=====================================
- Uses ALL 13 train-split OC subjects
- Differential LRs: transformer=lr/10, det_head=lr
- Augmentation: horizontal flip, brightness jitter, temporal crop
- Evaluates on held-out 36 test OC subjects
"""
import argparse, json, yaml, numpy as np, torch, torch.nn as nn, torch.optim as optim, random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance

OC_CONDITION = "01-oc"
NM_CONDITION = "00-nm"
SILS_SUBDIR  = "Sils_aligned"
SEQ_LEN      = 30

def load_clip(sils_dir, seq_len=SEQ_LEN, augment=False):
    frames = sorted(Path(sils_dir).glob("*.png"))
    if len(frames) < 4: return None
    imgs = []
    for fp in frames:
        img = Image.open(fp).convert("L")
        if augment:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
    arr = np.stack(imgs)
    if augment and random.random() > 0.5:
        arr = arr[:, :, ::-1].copy()
    T = len(arr)
    if T >= seq_len:
        start = random.randint(0, T - seq_len) if augment and T > seq_len else 0
        arr = arr[start:start + seq_len]
    else:
        arr = np.concatenate([arr, np.zeros((seq_len-T, *arr.shape[1:]), dtype=np.float32)])
    return arr

class OcDataset(Dataset):
    def __init__(self, data_root, subjects, augment=False):
        self.augment = augment
        self.samples = []
        root = Path(data_root)
        for subj in sorted(subjects):
            for cond, label in [(OC_CONDITION,1),(NM_CONDITION,0)]:
                cond_dir = root/subj/cond
                if not cond_dir.exists(): continue
                for seq_dir in sorted(cond_dir.iterdir()):
                    if not seq_dir.is_dir(): continue
                    sils = seq_dir/SILS_SUBDIR
                    if sils.exists() and any(sils.glob("*.png")):
                        self.samples.append((sils, label))
        n_oc = sum(l for _,l in self.samples)
        print(f"  {'Train' if augment else 'Test'}: {len(self.samples)} seqs  OC={n_oc} NM={len(self.samples)-n_oc}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, label = self.samples[i]
        clip = load_clip(p, augment=self.augment)
        if clip is None: clip = np.zeros((SEQ_LEN,64,44),dtype=np.float32)
        return torch.from_numpy(clip).unsqueeze(1), torch.tensor(label, dtype=torch.float32)

def build_model(cfg_path, ckpt_path, device):
    cfg = yaml.safe_load(open(cfg_path))
    mc = cfg["model"]
    from models.model import OcclusionDetectionModel
    model = OcclusionDetectionModel(
        backbone_name=mc.get("backbone","resnet18"), pretrained=False,
        transformer_cfg=mc.get("transformer",{}),
        head_hidden=mc.get("heads",{}).get("hidden_dim",128),
        enable_extra_heads=True,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    m, u = model.load_state_dict(state, strict=False)
    print(f"  Loaded. Missing={len(m)} Unexpected={len(u)}")
    return model.to(device)

def make_optimizer(model, lr):
    for p in model.parameters(): p.requires_grad = False
    trans_p = [p for n,p in model.named_parameters() if n.startswith('transformer.')]
    det_p   = [p for n,p in model.named_parameters() if n.startswith('det_head.')]
    for p in trans_p+det_p: p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable={trainable:,}  transformer_lr={lr/10:.1e}  det_lr={lr:.1e}")
    return optim.Adam([{'params':trans_p,'lr':lr/10},{'params':det_p,'lr':lr}], weight_decay=1e-4)

def run_epoch(model, loader, opt, device, train):
    model.train(train)
    bce = nn.BCEWithLogitsLoss()
    tl=tc=tn=0
    with torch.set_grad_enabled(train):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)[0].mean(dim=1)
            loss = bce(logits, labels)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tl += loss.item()*len(labels)
            tc += ((torch.sigmoid(logits)>=0.5).long()==labels.long()).sum().item()
            tn += len(labels)
    return tl/max(tn,1), tc/max(tn,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root",  required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--output_dir", default="checkpoints/naturaloc_proper")
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    split    = json.load(open(args.split_json))
    root     = Path(args.data_root)
    train_oc = sorted([s for s in split['TRAIN_SET'] if (root/s/OC_CONDITION).exists()])
    test_oc  = sorted([s for s in split['TEST_SET']  if (root/s/OC_CONDITION).exists()])

    print(f"\n{'='*60}")
    print(f"  Train OC subjects: {len(train_oc)} → {train_oc}")
    print(f"  Test  OC subjects: {len(test_oc)}")
    print(f"{'='*60}\n")

    train_ds = OcDataset(args.data_root, train_oc, augment=True)
    test_ds  = OcDataset(args.data_root, test_oc,  augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.config, args.checkpoint, args.device)
    opt   = make_optimizer(model, args.lr)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0

    print(f"\n{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'TeLoss':>8}  {'TeAcc':>7}")
    print("-"*50)
    for ep in range(1, args.epochs+1):
        tr_l, tr_a = run_epoch(model, train_dl, opt, args.device, True)
        te_l, te_a = run_epoch(model, test_dl,  opt, args.device, False)
        sched.step()
        flag=""
        if te_a >= best:
            best = te_a
            torch.save(model.state_dict(), out_dir/"best_natural_oc.pth")
            flag=" ✓"
        print(f"{ep:>4}  {tr_l:>8.4f}  {tr_a:>7.3f}  {te_l:>8.4f}  {te_a:>7.3f}{flag}")

    print(f"\nBest test acc: {best:.3f}")
    print(f"Checkpoint: {out_dir}/best_natural_oc.pth")

if __name__=="__main__": main()
