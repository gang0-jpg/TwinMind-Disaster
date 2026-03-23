import os
import glob
import csv
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_case(npz_path: str):
    d = np.load(npz_path)
    flood = d["flood_gt"].astype(np.float32)
    mask = d["mask"].astype(np.float32)
    return flood, mask


def load_pred_map(
    output_dir: str,
    base_name: str,
    candidates: list[str],
) -> np.ndarray:
    for pat in candidates:
        path = os.path.join(output_dir, pat.format(base=base_name))
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    raise FileNotFoundError(
        f"prediction file not found for case={base_name} in {output_dir}\n"
        f"tried: {[pat.format(base=base_name) for pat in candidates]}"
    )


def masked_flatten(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask > 0
    return arr[valid]


def mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    p = masked_flatten(pred, mask)
    g = masked_flatten(gt, mask)
    return float(np.mean(np.abs(p - g)))


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    p = masked_flatten(pred, mask)
    g = masked_flatten(gt, mask)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def binarize_map(x: np.ndarray, thr: float, mask: np.ndarray) -> np.ndarray:
    y = (x >= thr).astype(np.uint8)
    y = y * (mask > 0).astype(np.uint8)
    return y


def iou_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin > 0, gt_bin > 0).sum()
    union = np.logical_or(pred_bin > 0, gt_bin > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def f1_score_binary(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    tp = np.logical_and(pred_bin > 0, gt_bin > 0).sum()
    fp = np.logical_and(pred_bin > 0, gt_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, gt_bin > 0).sum()
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 1.0
    return float((2 * tp) / denom)


def boundary_map(bin_map: np.ndarray) -> np.ndarray:
    """
    4-neighbor boundary extraction.
    """
    x = bin_map.astype(np.uint8)
    h, w = x.shape
    b = np.zeros_like(x, dtype=np.uint8)

    b[:-1, :] |= (x[:-1, :] != x[1:, :])
    b[1:, :] |= (x[:-1, :] != x[1:, :])
    b[:, :-1] |= (x[:, :-1] != x[:, 1:])
    b[:, 1:] |= (x[:, :-1] != x[:, 1:])

    return b


def dilate_binary(bin_map: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (bin_map > 0).astype(np.uint8)

    x = torch.tensor(bin_map[None, None, ...].astype(np.float32))
    k = 2 * radius + 1
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return (y[0, 0].numpy() > 0).astype(np.uint8)


def boundary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, tolerance: int = 2) -> float:
    pred_b = boundary_map(pred_bin)
    gt_b = boundary_map(gt_bin)

    pred_b_d = dilate_binary(pred_b, tolerance)
    gt_b_d = dilate_binary(gt_b, tolerance)

    matched_pred = np.logical_and(pred_b > 0, gt_b_d > 0).sum()
    matched_gt = np.logical_and(gt_b > 0, pred_b_d > 0).sum()

    pred_count = (pred_b > 0).sum()
    gt_count = (gt_b > 0).sum()

    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    precision = matched_pred / pred_count
    recall = matched_gt / gt_count

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def uncertainty_metrics(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    top_frac: float = 0.10,
) -> dict:
    valid = mask > 0

    abs_err = np.abs(pred_mean - gt)[valid]
    unc = pred_std[valid]

    out = {
        "mean_uncertainty": float(np.mean(unc)) if unc.size > 0 else float("nan"),
        "max_uncertainty": float(np.max(unc)) if unc.size > 0 else float("nan"),
        "corr_abs_error_uncertainty": pearson_corr(abs_err, unc),
        "high_uncertainty_mae": float("nan"),
        "low_uncertainty_mae": float("nan"),
        "high_low_error_ratio": float("nan"),
    }

    if unc.size == 0:
        return out

    n = unc.size
    k = max(1, int(n * top_frac))

    order = np.argsort(unc)
    low_idx = order[:k]
    high_idx = order[-k:]

    low_mae = float(np.mean(abs_err[low_idx]))
    high_mae = float(np.mean(abs_err[high_idx]))

    out["high_uncertainty_mae"] = high_mae
    out["low_uncertainty_mae"] = low_mae
    out["high_low_error_ratio"] = float(high_mae / (low_mae + 1e-12))
    return out


def pct_improvement(base_value: float, hybrid_value: float, lower_is_better: bool = True) -> float:
    if lower_is_better:
        if abs(base_value) < 1e-12:
            return 0.0
        return float((base_value - hybrid_value) / base_value * 100.0)
    else:
        denom = abs(base_value) if abs(base_value) > 1e-12 else 1.0
        return float((hybrid_value - base_value) / denom * 100.0)


def mean_ignore_nan(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return float("nan")
    return float(arr[mask].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_dir", default="data/cases")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--hybrid_dir", required=True)
    parser.add_argument("--std_dir", default=None, help="directory containing *_pred_std.npy; default=hybrid_dir")
    parser.add_argument("--output_csv", default="evaluation_summary.csv")
    parser.add_argument("--flood_threshold", type=float, default=0.001)
    parser.add_argument("--boundary_tolerance", type=int, default=2)
    parser.add_argument("--top_frac", type=float, default=0.10)
    args = parser.parse_args()

    std_dir = args.std_dir if args.std_dir is not None else args.hybrid_dir

    all_case_files = sorted(glob.glob(os.path.join(args.cases_dir, "*.npz")))
    if not all_case_files:
        raise RuntimeError(f"no case files found in {args.cases_dir}")

    case_files = []
    pred_candidates = [
        "{base}_pred_mean.npy",
        "{base}_pred.npy",
    ]
    std_candidates = [
        "{base}_pred_std.npy",
    ]

    for npz_path in all_case_files:
        base_name = Path(npz_path).stem

        try:
            _ = load_pred_map(args.base_dir, base_name, pred_candidates)
            _ = load_pred_map(args.hybrid_dir, base_name, pred_candidates)
            _ = load_pred_map(std_dir, base_name, std_candidates)
            case_files.append(npz_path)
        except FileNotFoundError:
            pass

    if not case_files:
        raise RuntimeError("no common cases found across base/hybrid/std outputs")

    print(f"[info] common cases used for evaluation = {len(case_files)}")

    rows = []

    agg = {
        "base_mae": [],
        "hybrid_mae": [],
        "base_rmse": [],
        "hybrid_rmse": [],
        "base_iou": [],
        "hybrid_iou": [],
        "base_f1": [],
        "hybrid_f1": [],
        "base_boundary_f1": [],
        "hybrid_boundary_f1": [],
        "mae_improve_pct": [],
        "rmse_improve_pct": [],
        "iou_improve_pct": [],
        "f1_improve_pct": [],
        "boundary_f1_improve_pct": [],
        "mean_uncertainty": [],
        "corr_abs_error_uncertainty": [],
        "high_low_error_ratio": [],
    }

    for npz_path in case_files:
        base_name = Path(npz_path).stem
        gt, mask = load_case(npz_path)

        base_pred = load_pred_map(args.base_dir, base_name, pred_candidates)
        hybrid_pred = load_pred_map(args.hybrid_dir, base_name, pred_candidates)
        hybrid_std = load_pred_map(std_dir, base_name, std_candidates)

        base_mae = mae(base_pred, gt, mask)
        hybrid_mae = mae(hybrid_pred, gt, mask)

        base_rmse = rmse(base_pred, gt, mask)
        hybrid_rmse = rmse(hybrid_pred, gt, mask)

        gt_bin = binarize_map(gt, args.flood_threshold, mask)
        base_bin = binarize_map(base_pred, args.flood_threshold, mask)
        hybrid_bin = binarize_map(hybrid_pred, args.flood_threshold, mask)

        base_iou = iou_score(base_bin, gt_bin)
        hybrid_iou = iou_score(hybrid_bin, gt_bin)

        base_f1 = f1_score_binary(base_bin, gt_bin)
        hybrid_f1 = f1_score_binary(hybrid_bin, gt_bin)

        base_bf1 = boundary_f1(base_bin, gt_bin, tolerance=args.boundary_tolerance)
        hybrid_bf1 = boundary_f1(hybrid_bin, gt_bin, tolerance=args.boundary_tolerance)

        unc = uncertainty_metrics(
            pred_mean=hybrid_pred,
            pred_std=hybrid_std,
            gt=gt,
            mask=mask,
            top_frac=args.top_frac,
        )

        row = {
            "case": base_name,
            "base_mae": base_mae,
            "hybrid_mae": hybrid_mae,
            "mae_improve_pct": pct_improvement(base_mae, hybrid_mae, lower_is_better=True),
            "base_rmse": base_rmse,
            "hybrid_rmse": hybrid_rmse,
            "rmse_improve_pct": pct_improvement(base_rmse, hybrid_rmse, lower_is_better=True),
            "base_iou": base_iou,
            "hybrid_iou": hybrid_iou,
            "iou_improve_pct": pct_improvement(base_iou, hybrid_iou, lower_is_better=False),
            "base_f1": base_f1,
            "hybrid_f1": hybrid_f1,
            "f1_improve_pct": pct_improvement(base_f1, hybrid_f1, lower_is_better=False),
            "base_boundary_f1": base_bf1,
            "hybrid_boundary_f1": hybrid_bf1,
            "boundary_f1_improve_pct": pct_improvement(base_bf1, hybrid_bf1, lower_is_better=False),
            "mean_uncertainty": unc["mean_uncertainty"],
            "max_uncertainty": unc["max_uncertainty"],
            "corr_abs_error_uncertainty": unc["corr_abs_error_uncertainty"],
            "high_uncertainty_mae": unc["high_uncertainty_mae"],
            "low_uncertainty_mae": unc["low_uncertainty_mae"],
            "high_low_error_ratio": unc["high_low_error_ratio"],
        }
        rows.append(row)

        for k in agg:
            agg[k].append(row[k])

    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("Per-case summary saved to:", args.output_csv)
    print("=" * 72)

    print("\n[Reservoir / Hybrid effect]")
    print(f"MAE              Base={mean_ignore_nan(agg['base_mae']):.6f}  Hybrid={mean_ignore_nan(agg['hybrid_mae']):.6f}  Improve={mean_ignore_nan(agg['mae_improve_pct']):.2f}%")
    print(f"RMSE             Base={mean_ignore_nan(agg['base_rmse']):.6f}  Hybrid={mean_ignore_nan(agg['hybrid_rmse']):.6f}  Improve={mean_ignore_nan(agg['rmse_improve_pct']):.2f}%")
    print(f"IoU              Base={mean_ignore_nan(agg['base_iou']):.6f}  Hybrid={mean_ignore_nan(agg['hybrid_iou']):.6f}  Improve={mean_ignore_nan(agg['iou_improve_pct']):.2f}%")
    print(f"F1               Base={mean_ignore_nan(agg['base_f1']):.6f}  Hybrid={mean_ignore_nan(agg['hybrid_f1']):.6f}  Improve={mean_ignore_nan(agg['f1_improve_pct']):.2f}%")
    print(f"Boundary F1      Base={mean_ignore_nan(agg['base_boundary_f1']):.6f}  Hybrid={mean_ignore_nan(agg['hybrid_boundary_f1']):.6f}  Improve={mean_ignore_nan(agg['boundary_f1_improve_pct']):.2f}%")

    print("\n[MC Dropout effect]")
    print(f"Mean uncertainty              = {mean_ignore_nan(agg['mean_uncertainty']):.6f}")
    print(f"Corr(abs error, uncertainty)  = {mean_ignore_nan(agg['corr_abs_error_uncertainty']):.6f}")
    print(f"High/Low uncertainty error ratio = {mean_ignore_nan(agg['high_low_error_ratio']):.3f}x")

    print("\nInterpretation guide:")
    print("- Hybrid is better if MAE/RMSE go down and IoU/F1/Boundary F1 go up.")
    print("- MC Dropout is meaningful if corr(abs error, uncertainty) is positive and")
    print("  high/low uncertainty error ratio is clearly > 1.0.")
    print("=" * 72)


if __name__ == "__main__":
    main()
