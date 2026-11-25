"""Downstream YOLOv8 detection evaluation and small-object size-stratified analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

DIOR_CLASSES: dict[int, str] = {
    0: "Expressway-Service-area",
    1: "Expressway-toll-station",
    2: "airplane",
    3: "airport",
    4: "baseballfield",
    5: "basketballcourt",
    6: "bridge",
    7: "chimney",
    8: "dam",
    9: "golffield",
    10: "groundtrackfield",
    11: "harbor",
    12: "overpass",
    13: "ship",
    14: "stadium",
    15: "storagetank",
    16: "tenniscourt",
    17: "trainstation",
    18: "vehicle",
    19: "windmill",
}

# Relative bounding-box area thresholds (box_area / image_area)
SIZE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "small": (0.0, 0.005),
    "medium": (0.005, 0.05),
    "large": (0.05, 1.01),
}


def categorize_box_size(rel_area: float) -> str:
    """Categorize a bounding box by its area relative to the image area."""
    if rel_area < SIZE_THRESHOLDS["small"][1]:
        return "small"
    if rel_area < SIZE_THRESHOLDS["medium"][1]:
        return "medium"
    return "large"


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two (N, 4) and (M, 4) xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized YOLO labels as (xyxy_pixels, class_ids)."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes: list[list[float]] = []
    classes: list[int] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        x1 = (xc - w / 2.0) * img_w
        y1 = (yc - h / 2.0) * img_h
        x2 = (xc + w / 2.0) * img_w
        y2 = (yc + h / 2.0) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(cls_id)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)


def create_dataset_yaml(dataset_dir: Path, yaml_path: Path) -> Path:
    """Write a YOLO dataset YAML pointing to images/ and labels/ under dataset_dir."""
    lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images",
        "val: images",
        "names:",
    ]
    for k, v in DIOR_CLASSES.items():
        lines.append(f"  {k}: {v}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def evaluate_yolo_dataset(
    yolo_weights: Path | str,
    dataset_dir: Path | str,
    yaml_path: Path | str,
    imgsz: int = 800,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run fixed-detector YOLOv8 validation and return overall + per-class metrics."""
    os.environ["YOLO_VERBOSE"] = "False"
    model = YOLO(str(yolo_weights))
    yaml_file = create_dataset_yaml(Path(dataset_dir), Path(yaml_path))

    results = model.val(
        data=str(yaml_file),
        imgsz=imgsz,
        split="val",
        device=device,
        verbose=False,
        plots=False,
    )
    metrics = results.results_dict

    per_class_ap50: dict[str, float] = {}
    if hasattr(results, "box") and hasattr(results.box, "ap50") and hasattr(results.box, "ap_class_index"):
        for idx, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = DIOR_CLASSES.get(int(cls_idx), str(int(cls_idx)))
            per_class_ap50[cls_name] = float(results.box.ap50[idx])

    return {
        "mAP50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
        "per_class_ap50": per_class_ap50,
    }


def analyze_small_objects(
    yolo_weights: Path | str,
    images_dir: Path | str,
    labels_dir: Path | str,
    conf: float = 0.25,
    iou_thresh: float = 0.5,
    imgsz: int = 800,
    device: str = "cpu",
) -> dict[str, dict[str, float | int]]:
    """Compute size-stratified recall, precision, TP, FP, FN across small/medium/large objects."""
    os.environ["YOLO_VERBOSE"] = "False"
    model = YOLO(str(yolo_weights))

    stats: dict[str, dict[str, int]] = {
        cat: {"gt_count": 0, "tp": 0, "fp": 0, "fn": 0}
        for cat in ("small", "medium", "large")
    }

    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_area = float(w * h)

        gt_boxes, gt_cls = _load_yolo_labels(lbl_dir / f"{img_path.stem}.txt", w, h)
        preds = model.predict(source=img, imgsz=imgsz, conf=conf, device=device, verbose=False)

        if preds and len(preds[0].boxes) > 0:
            pred_boxes = preds[0].boxes.xyxy.cpu().numpy()
            pred_cls = preds[0].boxes.cls.cpu().numpy().astype(np.int64)
            pred_conf = preds[0].boxes.conf.cpu().numpy()
            order = np.argsort(-pred_conf)
            pred_boxes = pred_boxes[order]
            pred_cls = pred_cls[order]
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_cls = np.zeros((0,), dtype=np.int64)

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        iou_mat = _compute_iou_matrix(pred_boxes, gt_boxes)

        for p_idx in range(len(pred_boxes)):
            p_area = max(0.0, (pred_boxes[p_idx, 2] - pred_boxes[p_idx, 0]) * (pred_boxes[p_idx, 3] - pred_boxes[p_idx, 1]))
            p_cat = categorize_box_size(p_area / img_area)

            best_gt = -1
            best_iou = iou_thresh
            for g_idx in range(len(gt_boxes)):
                if gt_matched[g_idx] or gt_cls[g_idx] != pred_cls[p_idx]:
                    continue
                if iou_mat[p_idx, g_idx] >= best_iou:
                    best_iou = float(iou_mat[p_idx, g_idx])
                    best_gt = g_idx

            if best_gt >= 0:
                gt_matched[best_gt] = True
                g_area = max(0.0, (gt_boxes[best_gt, 2] - gt_boxes[best_gt, 0]) * (gt_boxes[best_gt, 3] - gt_boxes[best_gt, 1]))
                g_cat = categorize_box_size(g_area / img_area)
                stats[g_cat]["tp"] += 1
            else:
                stats[p_cat]["fp"] += 1

        for g_idx in range(len(gt_boxes)):
            g_area = max(0.0, (gt_boxes[g_idx, 2] - gt_boxes[g_idx, 0]) * (gt_boxes[g_idx, 3] - gt_boxes[g_idx, 1]))
            g_cat = categorize_box_size(g_area / img_area)
            stats[g_cat]["gt_count"] += 1
            if not gt_matched[g_idx]:
                stats[g_cat]["fn"] += 1

    summary: dict[str, dict[str, float | int]] = {}
    for cat, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        summary[cat] = {
            **s,
            "recall": float(recall),
            "precision": float(precision),
        }
    return summary
