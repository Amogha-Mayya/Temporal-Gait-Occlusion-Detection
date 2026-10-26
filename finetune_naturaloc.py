"""
Few-shot Domain Adaptation for Natural Occlusion Detection  (v2)
================================================================
Fixed dataset loader: SUSTech1K structure is
  data_root / subject / condition / seq_id / Sils_aligned / *.png

Fine-tunes ONLY the detection head on 5 real OC subjects.
Everything else stays frozen.

Run:
  cd ~/AMOGHA-DIRECTORY/severity-folder
  python3 finetune_naturaloc_v2.py \
      --config configs/sustech1k.yaml \
      --checkpoint checkpoints/sustech1k/best_model_4heads.pth \
      --data_root /home/himanshu/sustek-data/SUSTech1K-Released-2023 \
      --output_dir checkpoints/naturaloc_adapted \
      --epochs 20 --lr 3e-4
"""

import argparse, os, sys, yaml, numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ADAPT_SUBJECTS = {"0812", "0813", "0814", "0816", "0817"}
OC_CONDITION   = "01-oc"
NM_CONDITION   = "00-nm"
SILS_SUBDIR    = "Sils_aligned"
SEQ_LEN        = 30


def load_silhouettes(sils_dir: Path, seq_len: int = SEQ_LEN):
    from PIL import Image
    frames = sorted(sils_dir.glob("*.png")) + sorted(sils_dir.glob("*.jpg"))
    if len(frames) < 4:
        return None
    imgs = []
    for fp in frames:
        img = Image.open(fp).convert("L")
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
    arr = np.stack(imgs, axis=0)   # (T, H, W)
    T = arr.shape[0]
    if T >= seq_len:
        arr = arr[:seq_len]
    else:
        pad = np.zeros((seq_len - T, *arr.shape[1:]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr   # (seq_len, H, W)


class NaturalOCDataset(Dataset):
    """
    Walks: data_root / subject / condition / seq_id / Sils_aligned / *.png
    OC condition → label 1,  NM condition → label 0
    """
    def __init__(self, data_root: str, subjects: set, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.samples = []   # list of (sils_dir_path, label)
        root = Path(data_root)
        for subj in sorted(subjects):
            subj_dir = root / subj
            if not subj_dir.exists():
                print(f"  WARNING: {subj_dir} not found"); continue
            for cond_name, label in [(OC_CONDITION, 1), (NM_CONDITION, 0)]:
                cond_dir = subj_dir / cond_name
                if not cond_dir.exists():
                    continue
                for seq_dir in sorted(cond_dir.iterdir()):
                    if not seq_dir.is_dir(): continue
                    sils_dir = seq_dir / SILS_SUBDIR
                    if sils_dir.exists() and any(sils_dir.glob("*.png")):
                        self.samples.append((sils_dir, label))

        n_oc = sum(1 for _, l in self.samples if l == 1)
        n_nm = sum(1 for _, l in self.samples if l == 0)
        print(f"  Dataset: {len(self.samples)} sequences  (OC={n_oc}, NM={n_nm})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sils_dir, label = self.samples[idx]
        clip = load_silhouettes(sils_dir, self.seq_len)
        if clip is None:
            clip = np.zeros((self.seq_len, 64, 44), dtype=np.float32)
        clip_t = torch.from_numpy(clip).unsqueeze(1)   # (T, 1, H, W)
        return clip_t, torch.tensor(label, dtype=torch.float32)


def load_model(cfg_path, ckpt_path, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    try:
        from models.model import build_model_with_pretrained_base
        model = build_model_with_pretrained_base(cfg, ckpt_path, device=device)
    except Exception:
        from models.four_head_model import FourHeadOcclusionModel
        model = FourHeadOcclusionModel(cfg)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
    return model.to(device), cfg


def freeze_except_det(model):
    frozen = trainable = 0
    for name, p in model.named_parameters():
        if "det_head" in name or "detection" in name.lower():
            p.requires_grad = True; trainable += p.numel()
        else:
            p.requires_grad = False; frozen += p.numel()
    print(f"  Frozen: {frozen:,}  |  Trainable (det head): {trainable:,}")
    if trainable == 0:
        print("  WARNING: no det_head params found — training all params")
        for p in model.parameters(): p.requires_grad = True


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    bce = nn.BCEWithLogitsLoss()
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            try:
                out = model.forward_all(clips)
                if "det_logit" in out:
                    logits = out["det_logit"]
                else:
                    probs = out.get("det_prob", list(out.values())[0])
                    logits = torch.logit(probs.clamp(1e-6, 1-1e-6))
            except Exception:
                out = model.predict_all(clips)
                probs = out["det_prob"]
                logits = torch.logit(probs.clamp(1e-6, 1-1e-6))
            if logits.dim() == 2:
                logits = logits.mean(dim=1)
            loss = bce(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root",  required=True)
    p.add_argument("--output_dir", default="checkpoints/naturaloc_adapted")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  Few-shot Natural OC Adaptation  (v2 — fixed data loader)")
    print(f"  Adapt subjects : {sorted(ADAPT_SUBJECTS)}")
    print(f"  Data structure : subject/condition/seq/Sils_aligned/*.png")
    print(f"{'='*60}\n")

    print("Building dataset...")
    ds = NaturalOCDataset(args.data_root, ADAPT_SUBJECTS)
    if len(ds) == 0:
        print("ERROR: no sequences found. Check --data_root"); sys.exit(1)

    n_val   = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"  Train={n_train}  Val={n_val}\n")

    print("Loading model...")
    model, cfg = load_model(args.config, args.checkpoint, args.device)
    print("Freezing...")
    freeze_except_det(model)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_val = 0.0

    print(f"\nTraining {args.epochs} epochs  lr={args.lr}")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}")
    print("-" * 45)
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_loader, opt, args.device, train=True)
        vl, va = run_epoch(model, val_loader,   opt, args.device, train=False)
        sched.step()
        flag = ""
        if va >= best_val:
            best_val = va
            # Save FULL state dict so eval script can load it directly
            full_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(full_state, out_dir / "best_adapted_full.pth")
            flag = " ✓ saved"
        print(f"{ep:>4}  {tl:>8.4f}  {ta:>7.3f}  {vl:>8.4f}  {va:>7.3f}{flag}")

    print(f"\nBest val acc: {best_val:.3f}")
    print(f"Full checkpoint: {out_dir}/best_adapted_full.pth")
    print(f"\nRun eval:")
    print(f"  python3 evaluate_natural_occlusion.py \\")
    print(f"      --config {args.config} \\")
    print(f"      --checkpoint {out_dir}/best_adapted_full.pth \\")
    print(f"      --report_name natural_oc_fewshot_v2")

if __name__ == "__main__":
    main()