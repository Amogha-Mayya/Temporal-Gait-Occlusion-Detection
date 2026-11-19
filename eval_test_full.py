#!/usr/bin/env python
"""
eval_test_full.py
-----------------
Held-out TEST-set evaluation of the 4-head occlusion model, for the paper.

For one dataset it
  1. runs best_model_4heads.pth over the test subjects (synthetic occlusion with a
     FIXED seed -> reproducible), collecting per-frame outputs of all four heads,
  2. computes every metric at a FIXED decision threshold (0.5 - nothing is tuned on
     the test set) with 95% subject-level cluster-bootstrap CIs,
  3. writes results JSON + raw predictions (.npz) + publication figures (PNG+PDF).

Usage
  python eval_test_full.py --dataset sustech1k
  python eval_test_full.py --dataset casiab --data_root /home/himanshu/CASIA-B-pkl
  python eval_test_full.py --dataset oumvlp           # needs OU-MVLP raw data on disk
  python eval_test_full.py --summarize                # cross-dataset tables (md + LaTeX)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, Dataset


# --- data/occlusion_generator.py on disk lacks 3 sampler functions (silhouette_merge,
# --- edge_noise, vertical_stripe) and fails to import. Without modifying that file,
# --- inject stand-in samplers in memory. The 5 original types are untouched.
import sys, types
try:
    import data.occlusion_generator  # noqa: F401
    FALLBACK_SAMPLERS = False
except NameError:
    FALLBACK_SAMPLERS = True
    _src = open("data/occlusion_generator.py").read()
    _inj = '''
def _sample_silhouette_merge_params(H, W, cfg):
    return {"cx": random.randint(int(0.2*W), int(0.8*W)), "cy": random.randint(int(0.2*H), int(0.8*H)),
            "rx": int(random.uniform(0.15, 0.35)*W), "ry": int(random.uniform(0.15, 0.35)*H)}

def _sample_edge_noise_params(H, W, cfg):
    return {"kernel_size": random.choice([3, 5]), "noise_std": random.uniform(10, 40)}

def _sample_vertical_stripe_params(H, W, cfg):
    n = cfg.get("n_stripes", 2); lo = cfg.get("min_width", 3); hi = cfg.get("max_width", 10)
    st = []
    for _ in range(n):
        w = random.randint(lo, hi); x1 = random.randint(0, max(0, W - w)); st.append((x1, x1 + w))
    return {"stripes": st}

'''
    _src = _src.replace("_SAMPLE_FN = {", _inj + "_SAMPLE_FN = {", 1)
    _mod = types.ModuleType("data.occlusion_generator")
    _mod.__file__ = "data/occlusion_generator.py"
    sys.modules["data.occlusion_generator"] = _mod
    exec(compile(_src, "data/occlusion_generator.py(patched-in-memory)", "exec"), _mod.__dict__)
    import data
    data.occlusion_generator = _mod

from data.occlusion_generator import OcclusionGenerator
from data.transforms import build_transforms
from models.model import build_model
from utils.confidence import compute_confidence_targets

OUT_ROOT = Path("outputs/test_eval")
REGIONS = ["upper", "lower", "left", "right"]
# Okabe-Ito (colour-blind safe)
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#7f7f7f"

DATASETS = {
    "casiab":   dict(cfg="configs/config.yaml",    ckpt="checkpoints/best_model_4heads.pth",         title="CASIA-B"),
    "oumvlp":   dict(cfg="configs/oumvlp.yaml",    ckpt="checkpoints/oumvlp/best_model_4heads.pth",  title="OU-MVLP"),
    "sustech1k": dict(cfg="configs/sustech1k.yaml", ckpt="checkpoints/sustech1k/best_model_4heads.pth", title="SUSTech1K"),
}


# --------------------------------------------------------------------------- data
class CasiaPklTestDataset(Dataset):
    """CASIA-B test windows read from OpenGait-style pkl (T,64,64 uint8) files.
    Same windowing (len 30, stride 15), subject split and occlusion generator as
    CasiaGaitDataset - only the on-disk format differs (the PNG copy is gone)."""

    def __init__(self, root, subjects, cfg, transform):
        self.gen = OcclusionGenerator(cfg.get("occlusion", {}))
        self.transform = transform
        L, S = cfg["dataset"]["sequence_length"], cfg["dataset"]["stride"]
        self.index, self.meta, self._cache = [], [], {}
        for s in subjects:
            sd = Path(root) / f"{s:03d}"
            for cond in sorted(p for p in sd.iterdir() if p.is_dir()):
                for view in sorted(p for p in cond.iterdir() if p.is_dir()):
                    pk = sorted(view.glob("*.pkl"))
                    if not pk:
                        continue
                    n = pickle.load(open(pk[0], "rb")).shape[0]
                    if n < cfg["dataset"]["min_frames"]:
                        continue
                    for st in range(0, n - L + 1, S):
                        self.index.append((str(pk[0]), st, L))
                        self.meta.append((s, cond.name, view.name))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        path, st, L = self.index[i]
        arr = pickle.load(open(path, "rb"))[st:st + L]
        if arr.max() <= 1:
            arr = (arr * 255).astype(np.uint8)
        res = self.gen.apply_to_sequence([f for f in arr])
        return dict(
            frames=self.transform([r.frame for r in res]),
            det_labels=torch.tensor([r.is_occluded for r in res], dtype=torch.long),
            sev_labels=torch.tensor([r.severity for r in res], dtype=torch.float32),
            region_labels=torch.tensor([[r.region_upper, r.region_lower, r.region_left, r.region_right]
                                        for r in res], dtype=torch.float32),
        )


class Seeded(Dataset):
    """Makes any dataset deterministic: reseeds python/numpy RNG from the sample index."""

    def __init__(self, ds, seed):
        self.ds, self.seed = ds, seed

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        random.seed(self.seed * 1_000_003 + i)
        np.random.seed((self.seed * 1_000_003 + i) % (2**32 - 1))
        d = self.ds[i]
        d["idx"] = i
        return d


def sample_meta(ds, name, i):
    """(subject, condition, view) for sample i."""
    if hasattr(ds, "meta"):
        return ds.meta[i]
    p = Path(ds.index[i][0][0]).parts            # .../subj/cond/view/<sil_folder>/file
    if name == "sustech1k":
        return int(p[-5]), p[-4], p[-3]
    # OU-MVLP / other: fall back to path pieces
    return p[-4], p[-3], p[-2]


def build_test_dataset(name, cfg, data_root, subject_limit):
    tf = build_transforms(cfg, split="test")
    if data_root:
        cfg["dataset"]["path"] = data_root
    if name == "casiab":
        sp = cfg["split"]
        subs = [s for s in range(sp["val_end"] + 1, sp["test_end"] + 1)]
        if subject_limit:
            subs = subs[:subject_limit]
        return CasiaPklTestDataset(cfg["dataset"]["path"], subs, cfg, tf)
    if name == "sustech1k":
        from data.sustech1k_dataset import build_sustech1k_dataset
        return build_sustech1k_dataset(cfg, "test", transform=tf)
    if name == "oumvlp":
        from data.oumvlp_dataset import build_oumvlp_dataset
        return build_oumvlp_dataset(cfg, split="test", transform=tf)
    raise ValueError(name)


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def run_inference(model, loader, device, amp):
    out = {k: [] for k in ["det_prob", "det_lab", "sev_pred", "sev_lab", "reg_prob", "reg_lab",
                           "conf_pred", "conf_tgt", "idx"]}
    for b, batch in enumerate(loader):
        fr = batch["frames"].to(device)
        dl = batch["det_labels"].to(device)
        sl = batch["sev_labels"].to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            o = model.forward_all(fr)
        dp = torch.sigmoid(o["det_logits"].float())
        sp = o["sev_preds"].float()
        ct = compute_confidence_targets(det_prob=dp, det_labels=dl, sev_preds=sp, sev_labels=sl)
        out["det_prob"].append(dp.cpu()); out["det_lab"].append(batch["det_labels"])
        out["sev_pred"].append(sp.cpu()); out["sev_lab"].append(batch["sev_labels"])
        out["reg_prob"].append(torch.sigmoid(o["region_logits"].float()).cpu())
        out["reg_lab"].append(batch["region_labels"])
        out["conf_pred"].append(o["conf_preds"].float().cpu()); out["conf_tgt"].append(ct.cpu())
        out["idx"].append(batch["idx"])
        if b % 50 == 0:
            print(f"  batch {b}/{len(loader)}", flush=True)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


# --------------------------------------------------------------------------- statistics
def per_group_stats(P, group):
    """Additive sufficient statistics per group (subject). Returns (G, D) array."""
    pred = (P["det_prob"] >= 0.5)
    lab = P["det_lab"].astype(bool)
    occ = lab
    cols = [
        (pred & lab).sum(1), (pred & ~lab).sum(1), (~pred & ~lab).sum(1), (~pred & lab).sum(1),   # tp fp tn fn
        (np.abs(P["sev_pred"] - P["sev_lab"]) * occ).sum(1),
        (((P["sev_pred"] - P["sev_lab"]) ** 2) * occ).sum(1), occ.sum(1),
    ]
    rp = P["reg_prob"] >= 0.5
    rl = P["reg_lab"].astype(bool)
    for r in range(4):
        cols += [(rp[..., r] & rl[..., r]).sum(1), (rp[..., r] & ~rl[..., r]).sum(1),
                 (~rp[..., r] & ~rl[..., r]).sum(1), (~rp[..., r] & rl[..., r]).sum(1)]
    cols += [np.abs(P["conf_pred"] - P["conf_tgt"]).sum(1), ((P["conf_pred"] - P["conf_tgt"]) ** 2).sum(1),
             np.full(len(pred), pred.shape[1])]
    X = np.stack(cols, 1).astype(np.float64)
    G = group.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, group, X)
    return S


def _prf(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(accuracy=(tp + tn) / n, precision=pr, recall=rc, specificity=sp, f1=f1,
                balanced_accuracy=(rc + sp) / 2, mcc=mcc)


def metrics_from_sum(s):
    m = {}
    tp, fp, tn, fn = s[:4]
    m.update({f"det_{k}": v for k, v in _prf(tp, fp, tn, fn).items()})
    nocc = s[6]
    m["sev_mae"] = s[4] / nocc if nocc else 0.0
    m["sev_rmse"] = float(np.sqrt(s[5] / nocc)) if nocc else 0.0
    f1s, accs = [], []
    for r, name in enumerate(REGIONS):
        d = _prf(*s[7 + 4 * r: 11 + 4 * r])
        for k in ("accuracy", "precision", "recall", "f1"):
            m[f"region_{name}_{k}"] = d[k]
        f1s.append(d["f1"]); accs.append(d["accuracy"])
    m["region_macro_f1"], m["region_macro_accuracy"] = float(np.mean(f1s)), float(np.mean(accs))
    n = s[-1]
    m["conf_mae"], m["conf_rmse"] = s[-3] / n, float(np.sqrt(s[-2] / n))
    return m


def ece_brier(prob, lab, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mk = ids == b
        if mk.any():
            ece += mk.mean() * abs(prob[mk].mean() - lab[mk].mean())
    return float(ece), float(np.mean((prob - lab) ** 2))


def compute_all(P, subj_codes, n_boot, seed):
    S = per_group_stats(P, subj_codes)
    point = metrics_from_sum(S.sum(0))
    rng = np.random.default_rng(seed)
    G = len(S)
    boots = []
    for _ in range(n_boot):
        boots.append(metrics_from_sum(S[rng.integers(0, G, G)].sum(0)))
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)),
              float(np.percentile([b[k] for b in boots], 97.5))] for k in point}

    # threshold-free / frame-level extras
    y = P["det_lab"].reshape(-1)
    p = P["det_prob"].reshape(-1)
    extra = {"det_roc_auc": float(roc_auc_score(y, p)), "det_pr_auc": float(average_precision_score(y, p)),
             "det_prevalence": float(y.mean())}
    extra["det_ece"], extra["det_brier"] = ece_brier(p, y)
    occ = y.astype(bool)
    sp, sl = P["sev_pred"].reshape(-1)[occ], P["sev_lab"].reshape(-1)[occ]
    extra["sev_pearson_r"] = float(np.corrcoef(sp, sl)[0, 1])
    extra["sev_r2"] = float(1 - ((sp - sl) ** 2).sum() / ((sl - sl.mean()) ** 2).sum())
    extra["sev_mae_all_frames"] = float(np.abs(P["sev_pred"] - P["sev_lab"]).mean())
    # severity MAE by bin of true severity
    bins = [0.0, 0.05, 0.10, 0.20, 0.35, 1.01]
    extra["sev_mae_by_bin"] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (sl >= lo) & (sl < hi)
        if mk.any():
            extra["sev_mae_by_bin"][f"{lo:.2f}-{min(hi,1):.2f}"] = dict(n=int(mk.sum()), mae=float(np.abs(sp[mk] - sl[mk]).mean()))
    # region extras
    rp = (P["reg_prob"] >= 0.5).reshape(-1, 4)
    rl = P["reg_lab"].reshape(-1, 4).astype(bool)
    extra["region_exact_match"] = float((rp == rl).all(1).mean())
    extra["region_hamming_acc"] = float((rp == rl).mean())
    extra["region_auc"] = {n: float(roc_auc_score(rl[:, i], P["reg_prob"].reshape(-1, 4)[:, i]))
                           for i, n in enumerate(REGIONS) if 0 < rl[:, i].sum() < len(rl)}
    extra["region_prevalence"] = {n: float(rl[:, i].mean()) for i, n in enumerate(REGIONS)}
    # confidence head as an error predictor
    wrong = ((P["det_prob"] >= 0.5) != P["det_lab"].astype(bool)).reshape(-1)
    if 0 < wrong.sum() < len(wrong):
        extra["conf_auc_for_detecting_errors"] = float(roc_auc_score(wrong, -P["conf_pred"].reshape(-1)))
    extra["conf_pearson_r"] = float(np.corrcoef(P["conf_pred"].reshape(-1), P["conf_tgt"].reshape(-1))[0, 1])
    extra["conf_mean_correct"] = float(P["conf_pred"].reshape(-1)[~wrong].mean())
    extra["conf_mean_wrong"] = float(P["conf_pred"].reshape(-1)[wrong].mean()) if wrong.any() else None
    extra["det_counts"] = {k: int(v) for k, v in zip(["tp", "fp", "tn", "fn"], S.sum(0)[:4])}
    return point, ci, extra


def bootstrap_auc(P, subj_codes, n_boot, seed):
    """Subject-level bootstrap CI for ROC-AUC / PR-AUC (subsamples frames for speed)."""
    rng = np.random.default_rng(seed)
    G = subj_codes.max() + 1
    members = [np.where(subj_codes == g)[0] for g in range(G)]
    r, a = [], []
    for _ in range(n_boot):
        ids = np.concatenate([members[g] for g in rng.integers(0, G, G)])
        y, p = P["det_lab"][ids].reshape(-1), P["det_prob"][ids].reshape(-1)
        if len(y) > 400_000:                      # random frame subsample keeps this tractable
            sel = rng.choice(len(y), 400_000, replace=False)
            y, p = y[sel], p[sel]
        r.append(roc_auc_score(y, p)); a.append(average_precision_score(y, p))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], \
           [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# --------------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
    })


def save(fig, d, name):
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)


def make_figures(P, point, extra, meta_view, title, fdir, thr=0.5):
    style()
    fdir.mkdir(parents=True, exist_ok=True)
    y, p = P["det_lab"].reshape(-1), P["det_prob"].reshape(-1)
    c = extra["det_counts"]

    # 1. confusion matrix (counts + row-normalised)
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], float)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            v = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{v*100:.1f}%\n(n={int(cm[i,j]):,})", ha="center", va="center",
                    color="white" if v > 0.5 else "black")
    ax.set_xticks([0, 1], ["Clean", "Occluded"]); ax.set_yticks([0, 1], ["Clean", "Occluded"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.grid(False)
    ax.set_title(f"{title} - detection confusion (test)")
    save(fig, fdir, "fig1_confusion_matrix")

    # 2. ROC and PR
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, color=C_BLUE, lw=1.8, label=f"AUC = {extra['det_roc_auc']:.4f}")
    ax[0].plot([0, 1], [0, 1], "--", color=C_GREY, lw=1)
    ax[0].plot(1 - point["det_specificity"], point["det_recall"], "o", color=C_VERM, ms=7, label=f"thr = {thr}")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC"); ax[0].legend(frameon=False)
    pr, rc, _ = precision_recall_curve(y, p)
    ax[1].plot(rc, pr, color=C_BLUE, lw=1.8, label=f"AP = {extra['det_pr_auc']:.4f}")
    ax[1].axhline(extra["det_prevalence"], ls="--", color=C_GREY, lw=1, label=f"prevalence = {extra['det_prevalence']:.3f}")
    ax[1].plot(point["det_recall"], point["det_precision"], "o", color=C_VERM, ms=7, label=f"thr = {thr}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall"); ax[1].legend(frameon=False)
    fig.suptitle(f"{title} - detection (test)", y=1.02)
    save(fig, fdir, "fig2_roc_pr")

    # 3. reliability diagram
    edges = np.linspace(0, 1, 11)
    ids = np.clip(np.digitize(p, edges) - 1, 0, 9)
    cx, cy, cn = [], [], []
    for b in range(10):
        mk = ids == b
        if mk.sum() > 0:
            cx.append(p[mk].mean()); cy.append(y[mk].mean()); cn.append(mk.sum())
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    ax[0].plot([0, 1], [0, 1], "--", color=C_GREY, lw=1, label="perfect")
    ax[0].plot(cx, cy, "o-", color=C_BLUE, label=f"model (ECE={extra['det_ece']:.3f})")
    ax[0].set(xlabel="Mean predicted probability", ylabel="Observed occlusion rate", title="Reliability"); ax[0].legend(frameon=False)
    ax[1].hist(p[y == 0], bins=40, alpha=.75, color=C_BLUE, label="clean frames", log=True)
    ax[1].hist(p[y == 1], bins=40, alpha=.75, color=C_ORANGE, label="occluded frames", log=True)
    ax[1].axvline(thr, color=C_VERM, ls="--", lw=1)
    ax[1].set(xlabel="Predicted occlusion probability", ylabel="Frames (log)", title="Score distribution"); ax[1].legend(frameon=False)
    save(fig, fdir, "fig3_calibration")

    # 4. severity
    occ = y.astype(bool)
    sp, sl = P["sev_pred"].reshape(-1)[occ], P["sev_lab"].reshape(-1)[occ]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    hb = ax[0].hexbin(sl, sp, gridsize=45, bins="log", cmap="Blues", mincnt=1)
    lim = max(sl.max(), sp.max()) * 1.02
    ax[0].plot([0, lim], [0, lim], "--", color=C_VERM, lw=1)
    ax[0].set(xlabel="True severity", ylabel="Predicted severity", title=f"Pred vs true (r={extra['sev_pearson_r']:.3f}, R²={extra['sev_r2']:.3f})")
    fig.colorbar(hb, ax=ax[0], label="log10(frames)")
    ax[1].hist(sp - sl, bins=80, color=C_BLUE)
    ax[1].axvline(0, color=C_VERM, lw=1)
    ax[1].set(xlabel="Predicted - true severity", ylabel="Occluded frames", title=f"Error (MAE={point['sev_mae']:.4f}, RMSE={point['sev_rmse']:.4f})")
    bins = extra["sev_mae_by_bin"]
    ax[2].bar(range(len(bins)), [v["mae"] for v in bins.values()], color=C_BLUE, width=.6)
    ax[2].set_xticks(range(len(bins)), [f"{k}\n(n={v['n']/1000:.1f}k)" for k, v in bins.items()], fontsize=7)
    ax[2].set(xlabel="True severity bin", ylabel="MAE", title="MAE by severity")
    fig.suptitle(f"{title} - severity head (occluded test frames)")
    save(fig, fdir, "fig4_severity")

    # 5. region head
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(4); w = .26
    for k, (m, col) in enumerate([("precision", C_BLUE), ("recall", C_ORANGE), ("f1", C_GREEN)]):
        vals = [point[f"region_{r}_{m}"] for r in REGIONS]
        ax[0].bar(x + (k - 1) * w, vals, w, label=m.capitalize() if m != "f1" else "F1", color=col)
    ax[0].set_xticks(x, [r.capitalize() for r in REGIONS]); ax[0].set_ylim(0, 1.05)
    ax[0].set(ylabel="Score", title=f"Per-quadrant (macro-F1={point['region_macro_f1']:.3f})"); ax[0].legend(frameon=False, ncol=3, loc="lower center")
    rp = (P["reg_prob"] >= .5).reshape(-1, 4); rl = P["reg_lab"].reshape(-1, 4).astype(bool)
    prev = [rl[:, i].mean() for i in range(4)]
    ax[1].bar(x, prev, .5, color=C_GREY)
    ax[1].set_xticks(x, [r.capitalize() for r in REGIONS])
    ax[1].set(ylabel="Fraction of frames occluded", title=f"Region prevalence (exact-match acc={extra['region_exact_match']:.3f})")
    fig.suptitle(f"{title} - region head (test)", y=1.03)
    save(fig, fdir, "fig5_region")

    # 6. confidence head
    cp, ct = P["conf_pred"].reshape(-1), P["conf_tgt"].reshape(-1)
    wrong = ((p >= thr) != y.astype(bool))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    hb = ax[0].hexbin(ct, cp, gridsize=45, bins="log", cmap="Blues", mincnt=1)
    ax[0].plot([0, 1], [0, 1], "--", color=C_VERM, lw=1)
    ax[0].set(xlabel="Confidence target", ylabel="Predicted confidence", title=f"MAE={point['conf_mae']:.4f}, r={extra['conf_pearson_r']:.3f}")
    fig.colorbar(hb, ax=ax[0], label="log10(frames)")
    ax[1].hist(cp[~wrong], bins=40, density=True, alpha=.75, color=C_BLUE, label="correct detections")
    if wrong.any():
        ax[1].hist(cp[wrong], bins=40, density=True, alpha=.75, color=C_VERM, label="wrong detections")
    ax[1].set(xlabel="Predicted confidence", ylabel="Density", title="Confidence vs correctness"); ax[1].legend(frameon=False)
    fig.suptitle(f"{title} - confidence head (test)")
    save(fig, fdir, "fig6_confidence")

    # 7. detection F1 / recall by view
    views = np.array(meta_view)
    uv = sorted(set(views))
    if 1 < len(uv) <= 30:
        f1s, rcs, ns = [], [], []
        for v in uv:
            mk = views == v
            yy, pp = P["det_lab"][mk].reshape(-1), (P["det_prob"][mk].reshape(-1) >= thr)
            tp, fp, fn = (pp & (yy == 1)).sum(), (pp & (yy == 0)).sum(), (~pp & (yy == 1)).sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1)); rcs.append(tp / max(tp + fn, 1)); ns.append(mk.sum())
        fig, ax = plt.subplots(figsize=(max(5, .55 * len(uv) + 2), 3.4))
        ax.bar(np.arange(len(uv)) - .2, f1s, .4, color=C_BLUE, label="F1")
        ax.bar(np.arange(len(uv)) + .2, rcs, .4, color=C_ORANGE, label="Recall")
        ax.set_xticks(range(len(uv)), uv, rotation=45, ha="right"); ax.set_ylim(0, 1.05)
        ax.set(xlabel="View angle", ylabel="Score", title=f"{title} - detection by view (test)"); ax.legend(frameon=False, ncol=2)
        save(fig, fdir, "fig7_per_view")

    # 8. example timelines: sequences with the most occluded frames, evenly spaced picks
    nocc = P["det_lab"].sum(1)
    T = P["det_lab"].shape[1]
    ok = np.where((nocc >= 8) & (nocc <= T - 8))[0]          # has both clean and occluded frames
    rs = np.random.default_rng(0)
    pick = rs.choice(ok, 3, replace=False) if len(ok) >= 3 else np.argsort(-nocc)[:3]
    fig, axes = plt.subplots(len(pick), 1, figsize=(8, 2.2 * len(pick)), sharex=True)
    axes = np.atleast_1d(axes)
    for a, i in zip(axes, pick):
        t = np.arange(P["det_prob"].shape[1])
        a.fill_between(t, 0, P["det_lab"][i], step="mid", color=C_ORANGE, alpha=.3, label="GT occluded")
        a.plot(t, P["det_prob"][i], color=C_BLUE, lw=1.8, label="P(occluded)")
        a.plot(t, P["sev_lab"][i], color=C_GREY, ls="--", lw=1.2, label="true severity")
        a.plot(t, P["sev_pred"][i], color=C_VERM, lw=1.2, label="pred severity")
        a.axhline(thr, color="k", lw=.6, ls=":"); a.set_ylim(-0.02, 1.05); a.set_ylabel(f"seq {i}")
    axes[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.45))
    axes[-1].set_xlabel("Frame")
    save(fig, fdir, "fig8_example_sequences")


# --------------------------------------------------------------------------- summary
def fmt(v, ci=None, d=4):
    return f"{v:.{d}f}" + (f" [{ci[0]:.{d}f}, {ci[1]:.{d}f}]" if ci else "")


def summarize():
    res = {}
    for k in DATASETS:
        f = OUT_ROOT / k / "test_results.json"
        if f.exists():
            res[k] = json.load(open(f))
    rows = [("Detection accuracy", "det_accuracy"), ("Detection precision", "det_precision"),
            ("Detection recall", "det_recall"), ("Detection F1", "det_f1"),
            ("Balanced accuracy", "det_balanced_accuracy"), ("MCC", "det_mcc"),
            ("Severity MAE", "sev_mae"), ("Severity RMSE", "sev_rmse"),
            ("Region macro-F1", "region_macro_f1"), ("Region upper F1", "region_upper_f1"),
            ("Region lower F1", "region_lower_f1"), ("Region left F1", "region_left_f1"),
            ("Region right F1", "region_right_f1"), ("Confidence MAE", "conf_mae")]
    ks = list(res)
    md = ["| Metric | " + " | ".join(DATASETS[k]["title"] for k in ks) + " |", "|---|" + "---|" * len(ks)]
    tex = ["\\begin{table}[t]\\centering\\caption{Test-set results (fixed threshold 0.5, 95\\% subject-level bootstrap CI).}",
           "\\begin{tabular}{l" + "c" * len(ks) + "}\\toprule",
           "Metric & " + " & ".join(DATASETS[k]["title"] for k in ks) + " \\\\\\midrule"]
    for lab, key in rows:
        md.append(f"| {lab} | " + " | ".join(fmt(res[k]["metrics"][key], res[k]["ci95"][key]) for k in ks) + " |")
        tex.append(f"{lab} & " + " & ".join(f"{res[k]['metrics'][key]:.3f} [{res[k]['ci95'][key][0]:.3f}, {res[k]['ci95'][key][1]:.3f}]" for k in ks) + " \\\\")
    for lab, key in [("ROC-AUC", "det_roc_auc"), ("PR-AUC (AP)", "det_pr_auc"), ("ECE", "det_ece"), ("Severity R²", "sev_r2")]:
        md.append(f"| {lab} | " + " | ".join(fmt(res[k]["extra"][key]) for k in ks) + " |")
        tex.append(f"{lab} & " + " & ".join(f"{res[k]['extra'][key]:.3f}" for k in ks) + " \\\\")
    tex += ["\\bottomrule\\end{tabular}\\end{table}"]
    md.append("")
    md.append("| Dataset | Test subjects | Test windows | Test frames | Occluded-frame prevalence |")
    md.append("|---|---|---|---|---|")
    for k in ks:
        r = res[k]
        md.append(f"| {DATASETS[k]['title']} | {r['n_subjects']} | {r['n_samples']:,} | {r['n_frames']:,} | {r['extra']['det_prevalence']:.3f} |")
    (OUT_ROOT / "summary_table.md").write_text("\n".join(md) + "\n")
    (OUT_ROOT / "summary_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))

    if len(ks) > 1:                                   # cross-dataset comparison figure
        style()
        keys = [("Det. F1", "det_f1"), ("Det. recall", "det_recall"), ("Det. precision", "det_precision"),
                ("Region macro-F1", "region_macro_f1")]
        cols = [C_BLUE, C_ORANGE, C_GREEN]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        w = .8 / len(ks)
        for j, k in enumerate(ks):
            v = [res[k]["metrics"][m] for _, m in keys]
            lo = [v[i] - res[k]["ci95"][m][0] for i, (_, m) in enumerate(keys)]
            hi = [res[k]["ci95"][m][1] - v[i] for i, (_, m) in enumerate(keys)]
            ax.bar(np.arange(len(keys)) + (j - (len(ks) - 1) / 2) * w, v, w, yerr=[lo, hi], capsize=2,
                   color=cols[j], label=DATASETS[k]["title"])
        ax.set_xticks(range(len(keys)), [a for a, _ in keys]); ax.set_ylim(0, 1.05)
        ax.set(ylabel="Score (test set, 95% CI)", title="Test-set performance across datasets"); ax.legend(frameon=False, ncol=3)
        save(fig, OUT_ROOT, "fig_cross_dataset")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--max_samples", type=int, default=0, help="smoke test: use first N windows")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved npz/json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--types", default=None, help="comma list; override occlusion.types (e.g. the 5 original types)")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    assert a.dataset, "--dataset required"

    d = DATASETS[a.dataset]
    cfg = yaml.safe_load(open(d["cfg"]))
    out = Path(a.out) if a.out else OUT_ROOT / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.replot:
        z = np.load(out / "test_predictions.npz")
        R = json.load(open(out / "test_results.json"))
        PP = {k: z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k] for k in
              ["det_prob","det_lab","sev_pred","sev_lab","reg_prob","reg_lab","conf_pred","conf_tgt","idx"]}
        make_figures(PP, R["metrics"], R["extra"], z["view"].tolist(), d["title"], out / "figures")
        return print("replotted", out)
    if a.types:
        cfg["occlusion"]["types"] = a.types.split(",")
    print("occlusion types:", cfg["occlusion"]["types"], "| fallback samplers:", FALLBACK_SAMPLERS, flush=True)
    ds = build_test_dataset(a.dataset, cfg, a.data_root, 0)
    if a.max_samples:
        ds.index = ds.index[: a.max_samples]
        if hasattr(ds, "meta"):
            ds.meta = ds.meta[: a.max_samples]
    print(f"[{a.dataset}] test windows: {len(ds):,}", flush=True)
    meta = [sample_meta(ds, a.dataset, i) for i in range(len(ds))]
    subj = np.array([m[0] for m in meta])
    uniq = {s: i for i, s in enumerate(sorted(set(subj.tolist())))}
    codes = np.array([uniq[s] for s in subj.tolist()])

    model = build_model(cfg, enable_extra_heads=True).to(device)
    ck = torch.load(a.checkpoint or d["ckpt"], map_location=device)
    missing, unexpected = model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True), None
    model.eval()
    print(f"loaded {a.checkpoint or d['ckpt']}", flush=True)

    loader = DataLoader(Seeded(ds, a.seed), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=device.type == "cuda")
    t0 = time.time()
    P = run_inference(model, loader, device, cfg["training"]["amp"])
    order = np.argsort(P["idx"])
    P = {k: v[order] for k, v in P.items()}
    print(f"inference done in {(time.time()-t0)/60:.1f} min", flush=True)
    np.savez_compressed(out / "test_predictions.npz", **{k: (v.astype(np.float16) if v.dtype == np.float32 else v) for k, v in P.items()},
                        subject=subj, view=np.array([m[2] for m in meta]), condition=np.array([m[1] for m in meta]))

    point, ci, extra = compute_all(P, codes, a.n_boot, a.seed)
    ci["det_roc_auc"], ci["det_pr_auc"] = bootstrap_auc(P, codes, min(a.n_boot, 100), a.seed)
    res = dict(dataset=d["title"], occlusion_types=cfg["occlusion"]["types"], fallback_samplers=FALLBACK_SAMPLERS, checkpoint=a.checkpoint or d["ckpt"], seed=a.seed, threshold=0.5,
               n_subjects=int(len(uniq)), n_samples=int(len(ds)), n_frames=int(P["det_lab"].size),
               metrics=point, ci95=ci, extra=extra)
    json.dump(res, open(out / "test_results.json", "w"), indent=2)

    make_figures(P, point, extra, [m[2] for m in meta], d["title"], out / "figures")

    print(f"\n===== {d['title']} TEST ({res['n_subjects']} subjects, {res['n_samples']:,} windows, {res['n_frames']:,} frames) =====")
    for k in ["det_accuracy", "det_precision", "det_recall", "det_f1", "det_specificity", "det_balanced_accuracy",
              "det_mcc", "sev_mae", "sev_rmse", "region_macro_f1", "region_upper_f1", "region_lower_f1",
              "region_left_f1", "region_right_f1", "conf_mae"]:
        print(f"  {k:24s} {fmt(point[k], ci[k])}")
    for k in ["det_roc_auc", "det_pr_auc", "det_ece", "sev_r2", "sev_pearson_r"]:
        print(f"  {k:24s} {extra[k]:.4f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
