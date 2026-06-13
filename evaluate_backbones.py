import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from anomaly_system import Config, MemoryBank, image_anomaly_score, get_feature_extractor, compute_threshold

BACKBONE = sys.argv[1] if len(sys.argv) > 1 else 'resnet50'

config = Config()
# optional threshold-policy override:  python evaluate_backbones.py <backbone> <method> <param>
if len(sys.argv) > 2:
    config.threshold_method = sys.argv[2]
if len(sys.argv) > 3:
    config.threshold_param = float(sys.argv[3])
backbone = get_feature_extractor(BACKBONE, device=config.device)
tf = backbone.transform
RESULTS_FILE = f"results_{BACKBONE}.md"


def raw_feats(path):
    return backbone(tf(path).to(config.device).unsqueeze(0)).cpu()


def make_scorer(mb):
    def score(path):
        t = tf(path).to(config.device).unsqueeze(0)
        with torch.no_grad():
            f = backbone(t)
            f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-8)
        d = torch.cdist(f, mb.get(), p=2.0)
        patch_min = torch.min(d, dim=1)[0]
        return image_anomaly_score(patch_min, method=config.score_method,
                                   top_k=config.score_top_k,
                                   smooth_sigma=config.smooth_sigma).item()
    return score


def write_md(rows, total, finished):
    status = "FINISHED" if finished else f"RUNNING... {len(rows)}/{total} categories done"
    lines = [
        f"# Cross-Category Evaluation — backbone: {BACKBONE}",
        "",
        f"Status: {status}",
        f"Coreset ratio: {config.coreset_ratio} | Scoring: method={config.score_method}, "
        f"top_k={config.score_top_k}, smooth_sigma={config.smooth_sigma}",
        f"Threshold: method={config.threshold_method}, param={config.threshold_param} "
        f"(leakage-free: set on held-out good only)",
        "",
        "| Category | Build | Test good | Test def | Threshold | AUC-ROC | F1 | FP |",
        "|----------|------:|----------:|---------:|----------:|--------:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]}/{r[2]} |")
    if rows:
        aucs = [r[5] for r in rows]
        f1s = [r[6] for r in rows]
        lines.append(f"| **mean ({len(rows)})** | | | | | **{np.mean(aucs):.4f}** | **{np.mean(f1s):.4f}** | |")
    open(RESULTS_FILE, "w", encoding="utf-8").write("\n".join(lines) + "\n")


base = config.base_path
categories = sorted([d.name for d in base.iterdir()
                     if d.is_dir() and (d / 'train' / 'good').exists()])

write_md([], len(categories), False)
print(f"[{BACKBONE}] Created {RESULTS_FILE}. Categories: {len(categories)}", flush=True)

rows = []
for cat in categories:
    train_dir = base / cat / 'train' / 'good'
    all_train = sorted(train_dir.glob('*.png'))
    n80 = int(len(all_train) * 0.8)
    build_files = all_train[:n80]
    calib_files = all_train[n80:]
    if not calib_files:
        calib_files = all_train[-3:]

    mb = MemoryBank(config.device)
    mb.build([raw_feats(p) for p in build_files])
    mb.k_center_greedy_select(ratio=config.coreset_ratio)
    score = make_scorer(mb)

    calib_scores = np.array([score(p) for p in calib_files])
    thr = compute_threshold(calib_scores, method=config.threshold_method,
                            param=config.threshold_param)

    test_root = base / cat / 'test'
    y_true, y_score = [], []
    for sub in sorted(test_root.iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == 'good' else 1
        for p in sorted(sub.glob('*.png')):
            y_score.append(score(p))
            y_true.append(label)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= thr).astype(int)
    n_good = int((y_true == 0).sum())
    n_def = int((y_true == 1).sum())
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    rows.append((cat, len(build_files), n_good, n_def, thr, auc, f1, fp))
    print(f"[{BACKBONE}] {cat:12s} AUC={auc:.4f} F1={f1:.4f} FP={fp}/{n_good}", flush=True)
    write_md(rows, len(categories), len(rows) == len(categories))

    del mb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"[{BACKBONE}] DONE -> {RESULTS_FILE}", flush=True)
