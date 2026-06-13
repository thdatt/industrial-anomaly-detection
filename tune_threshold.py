"""Threshold analysis tool (leakage-free).

Usage:  python tune_threshold.py <backbone> <category>
        python tune_threshold.py dinov2 screw

Shows, for ONE category, how precision/recall/F1 change with the decision
threshold. The OPERATING POINT must be chosen from a policy on held-out GOOD
(sigma-k or quantile-q), NEVER by reading test F1 -- that would be data leakage.

The last row ("ORACLE best-F1 on test") is the F1 you'd get if you cheated and
tuned on test. It is NOT a usable result -- it is printed only as the CEILING
that the feature representation allows. A big gap between your chosen policy and
the oracle means "better threshold policy could help"; a small gap means the
feature itself is the limit (e.g. defect scores overlap good scores).
"""
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from anomaly_system import Config, MemoryBank, image_anomaly_score, get_feature_extractor, compute_threshold

BACKBONE = sys.argv[1] if len(sys.argv) > 1 else 'dinov2'
CATEGORY = sys.argv[2] if len(sys.argv) > 2 else 'screw'

config = Config()
backbone = get_feature_extractor(BACKBONE, device=config.device)
tf = backbone.transform
base = config.base_path


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


# --- build memory bank + held-out calibration split (same as main eval) ---
train_dir = base / CATEGORY / 'train' / 'good'
all_train = sorted(train_dir.glob('*.png'))
n80 = int(len(all_train) * 0.8)
build_files, calib_files = all_train[:n80], all_train[n80:]
if not calib_files:
    calib_files = all_train[-3:]

mb = MemoryBank(config.device)
mb.build([raw_feats(p) for p in build_files])
mb.k_center_greedy_select(ratio=config.coreset_ratio)
score = make_scorer(mb)

calib_scores = np.array([score(p) for p in calib_files])   # held-out GOOD only

# --- score the test set once ---
test_root = base / CATEGORY / 'test'
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
auc = roc_auc_score(y_true, y_score)
n_good = int((y_true == 0).sum())
n_def = int((y_true == 1).sum())


def row(name, thr):
    y_pred = (y_score >= thr).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return f"{name:28s} thr={thr:.4f}  P={p:.3f}  R={r:.3f}  F1={f:.4f}  FP={fp}/{n_good}"


print(f"\n=== {BACKBONE} / {CATEGORY} ===  AUC={auc:.4f}  (good={n_good}, defect={n_def})")
print(f"calib good (held-out): mean={calib_scores.mean():.4f} std={calib_scores.std():.4f} "
      f"n={len(calib_scores)}\n")

print("-- POLICY: sigma  (threshold = mean + k*std of held-out good) --")
for k in (1.0, 2.0, 3.0, 4.0):
    print("  " + row(f"sigma k={k}", compute_threshold(calib_scores, 'sigma', k)))

print("\n-- POLICY: quantile  (threshold = q-quantile of held-out good; FPR target ~ 1-q) --")
for q in (0.90, 0.95, 0.98, 0.99, 1.00):
    print("  " + row(f"quantile q={q}", compute_threshold(calib_scores, 'quantile', q)))

# --- ORACLE upper bound: best F1 over all thresholds ON TEST (leakage; do NOT use) ---
cand = np.unique(y_score)
best_f1, best_thr = -1.0, None
for thr in cand:
    f = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
    if f > best_f1:
        best_f1, best_thr = f, thr
print("\n-- CEILING (NOT usable: tuned on test labels = leakage) --")
print("  " + row("ORACLE best-F1 on test", best_thr))
print(f"\n  current Config default: method={config.threshold_method}, param={config.threshold_param}")
print("  -> pick the highest-F1 row among the sigma/quantile policies above and set")
print("     config.threshold_method / config.threshold_param to it. The gap to ORACLE")
print("     is the headroom the feature representation still leaves on the table.\n")
