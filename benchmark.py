"""
benchmark.py for inspecting the saved model and measure performance + speed.
Saves results to benchmark_results.json so future lookups need zero computation.
"""

import os
import json
import time
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from preprocessing import seed_everything, prepare_image, hook_fn, PatchCoreLite

MODEL_PATH  = "transistor_lite_model.pt"
TEST_ROOT   = "transistor/test"
TEMPERATURE = 0.1
RESULTS_OUT = "benchmark_results.json"
SPEED_RUNS  = 20  # number of images used to measure average inference time


def build_backbone(device):
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    return resnet

def register_hooks(resnet, buf):
    resnet.layer2[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, buf))
    resnet.layer3[-1].register_forward_hook(lambda m, i, o: hook_fn(m, i, o, buf))

def score_image(img_path, resnet, device, memory_bank, buf):
    t = prepare_image(img_path, device)
    buf.clear()
    with torch.no_grad():
        resnet(t)
    patches = PatchCoreLite.aggregate_features(buf).to(device)
    return PatchCoreLite.compute_score(patches, memory_bank, temperature=TEMPERATURE)

def collect_test_images(test_root):
    """Returns list of (path, is_defect) tuples."""
    items = []
    for cat in sorted(os.listdir(test_root)):
        cat_path = os.path.join(test_root, cat)
        if not os.path.isdir(cat_path):
            continue
        is_defect = cat != "good"
        for fname in sorted(os.listdir(cat_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                items.append((os.path.join(cat_path, fname), is_defect, cat))
    return items


def print_checkpoint_info(ckpt, device):
    mb = ckpt['memory_bank']
    print(f"\n{'='*60}")
    print("  CHECKPOINT  (read directly — no computation)")
    print(f"{'='*60}")
    print(f"  Device used now : {device}")
    print(f"  Memory bank     : {mb.shape[0]} patches  ×  {mb.shape[1]} dims")
    print(f"  Threshold       : {ckpt['threshold']:.4f}")
    print(f"  Train mean      : {ckpt['train_mu']:.4f}")
    print(f"  Train std       : {ckpt['train_sigma']:.4f}")


def measure_speed(items, resnet, device, memory_bank, buf, n=SPEED_RUNS):
    print(f"\n{'='*60}")
    print(f"  INFERENCE SPEED  (avg over {n} images)")
    print(f"{'='*60}")
    sample = items[:n]
    times = []
    for path, _, _ in sample:
        start = time.perf_counter()
        score_image(path, resnet, device, memory_bank, buf)
        times.append(time.perf_counter() - start)
    avg_ms  = 1000 * sum(times) / len(times)
    min_ms  = 1000 * min(times)
    max_ms  = 1000 * max(times)
    fps     = 1000 / avg_ms
    print(f"  avg : {avg_ms:.1f} ms/image   ({fps:.1f} FPS)")
    print(f"  min : {min_ms:.1f} ms   max : {max_ms:.1f} ms")
    return {"avg_ms": round(avg_ms, 2), "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2), "fps": round(fps, 2)}


def evaluate(items, resnet, device, memory_bank, buf, threshold):
    print(f"\n{'='*60}")
    print("  PERFORMANCE METRICS")
    print(f"{'='*60}")

    by_cat   = {}
    all_scores, all_labels = [], []

    for path, is_defect, cat in items:
        s = score_image(path, resnet, device, memory_bank, buf)
        all_scores.append(s)
        all_labels.append(int(is_defect))
        by_cat.setdefault(cat, []).append((os.path.basename(path), s, is_defect))

    # per-category breakdown
    cat_stats = {}
    for cat, entries in sorted(by_cat.items()):
        is_defect_cat = cat != "good"
        tp = sum(1 for _, s, _ in entries if s > threshold and is_defect_cat)
        tn = sum(1 for _, s, _ in entries if s <= threshold and not is_defect_cat)
        fp = sum(1 for _, s, _ in entries if s > threshold and not is_defect_cat)
        fn = sum(1 for _, s, _ in entries if s <= threshold and is_defect_cat)
        n  = len(entries)
        correct = tp + tn
        avg_score = sum(s for _, s, _ in entries) / n
        cat_stats[cat] = {
            "count": n, "correct": correct,
            "avg_score": round(avg_score, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        }
        print(f"\n  [{cat.upper()}]  {correct}/{n} correct  (avg score {avg_score:.4f})")

    # global metrics
    total   = len(all_labels)
    preds   = [1 if s > threshold else 0 for s in all_scores]
    correct = sum(p == l for p, l in zip(preds, all_labels))
    tp_g = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 1)
    fp_g = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 0)
    fn_g = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 1)
    tn_g = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 0)

    accuracy  = correct / total
    precision = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0.0
    recall    = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    auroc = None
    try:
        from sklearn.metrics import roc_auc_score
        auroc = round(roc_auc_score(all_labels, all_scores), 4)
    except ImportError:
        # simple trapezoidal AUROC
        pairs = sorted(zip(all_scores, all_labels), key=lambda x: -x[0])
        pos = sum(all_labels)
        neg = total - pos
        if pos > 0 and neg > 0:
            tp_c = fp_c = 0
            prev_tpr = prev_fpr = 0.0
            area = 0.0
            for _, label in pairs:
                if label:
                    tp_c += 1
                else:
                    fp_c += 1
                tpr = tp_c / pos
                fpr = fp_c / neg
                area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
                prev_tpr, prev_fpr = tpr, fpr
            auroc = round(area, 4)

    print(f"\n  {'─'*40}")
    print(f"  Accuracy  : {correct}/{total}  ({100*accuracy:.1f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    if auroc is not None:
        print(f"  AUROC     : {auroc:.4f}")
    print(f"  {'─'*40}")

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": auroc,
        "total": total,
        "correct": correct,
        "confusion": {"tp": tp_g, "tn": tn_g, "fp": fp_g, "fn": fn_g},
        "per_category": cat_stats,
    }


if __name__ == "__main__":
    seed_everything(99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"|| Device: {device} ||")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at '{MODEL_PATH}'. Run train.py first.")
    ckpt        = torch.load(MODEL_PATH, map_location=device)
    memory_bank = ckpt['memory_bank'].to(device)
    threshold   = ckpt['threshold']
    print_checkpoint_info(ckpt, device)

    resnet = build_backbone(device)
    buf    = []
    register_hooks(resnet, buf)

    if not os.path.exists(TEST_ROOT):
        print(f"\nTest folder '{TEST_ROOT}' not found — skipping evaluation and speed test.")
    else:
        items = collect_test_images(TEST_ROOT)
        print(f"\nFound {len(items)} test images across {len(set(c for _,_,c in items))} categories.")

        speed = measure_speed(items, resnet, device, memory_bank, buf)

        metrics = evaluate(items, resnet, device, memory_bank, buf, threshold)

        results = {
            "model_path": MODEL_PATH,
            "checkpoint": {
                "bank_patches": ckpt['memory_bank'].shape[0],
                "bank_dims":    ckpt['memory_bank'].shape[1],
                "threshold":    round(threshold, 4),
                "train_mu":     round(ckpt['train_mu'], 4),
                "train_sigma":  round(ckpt['train_sigma'], 4),
            },
            "speed": speed,
            "metrics": metrics,
        }
        with open(RESULTS_OUT, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Results saved to '{RESULTS_OUT}' — no need to re-run.")
        print(f"{'='*60}\n")
