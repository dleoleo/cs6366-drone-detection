"""VisDrone-aware COCO evaluation for YOLO detections.

Why this exists: Ultralytics' VisDrone conversion deletes the dataset's "ignored regions"
(score 0: unlabelled crowds and the "others" class). Its built-in mAP therefore counts a
correct detection inside such a region as a false positive. It also reports no accuracy by
object size, although most VisDrone objects are tiny.

This module rebuilds ground truth from the raw VisDrone annotations. It marks ignored regions
as COCO crowd regions, so detections there count as neither true nor false positives, which
follows the official VisDrone protocol. It then runs pycocotools' COCOeval, allowing up to
500 detections per image as VisDrone does, and reports AP overall, by object size and by class.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
MAX_DETS = [10, 100, 500]  # COCOeval reports AP at the last value


def load_visdrone_gt(raw_split_dir):
    """Build a COCO-format ground-truth dict from one raw VisDrone-DET split directory.

    raw_split_dir contains images/*.jpg and annotations/*.txt. Each annotation line is:
    x, y, w, h, score, category, truncation, occlusion (pixels, top-left corner).
    """
    raw_split_dir = Path(raw_split_dir)
    images, annotations = [], []
    ann_files = sorted((raw_split_dir / "annotations").glob("*.txt"))
    if not ann_files:
        raise FileNotFoundError(f"No VisDrone annotation files in {raw_split_dir / 'annotations'}")
    for img_id, ann_file in enumerate(ann_files, start=1):
        width, height = Image.open(raw_split_dir / "images" / f"{ann_file.stem}.jpg").size
        images.append({"id": img_id, "file_name": f"{ann_file.stem}.jpg", "width": width, "height": height})
        for line in ann_file.read_text(encoding="utf-8").strip().splitlines():
            x, y, w, h, score, category = map(int, line.split(",")[:6])
            if w <= 0 or h <= 0:
                continue
            base = {"image_id": img_id, "bbox": [x, y, w, h], "area": w * h}
            if score == 1 and 1 <= category <= len(CLASS_NAMES):
                annotations.append({**base, "category_id": category, "iscrowd": 0})
            else:
                # Ignored region: COCO crowd regions are per class, so add one for every class
                # so that no detection inside it is penalised.
                for c in range(1, len(CLASS_NAMES) + 1):
                    annotations.append({**base, "category_id": c, "iscrowd": 1})
    for ann_id, ann in enumerate(annotations, start=1):
        ann["id"] = ann_id
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def load_ultralytics_predictions(pred_json, gt):
    """Read the predictions.json written by model.val(save_json=True) and match image ids to gt.

    Ultralytics writes image_id as the file stem and category_id as class index + 1, with boxes
    in original-image pixels, which matches load_visdrone_gt.
    """
    stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}
    detections, unmatched = [], set()
    for det in json.loads(Path(pred_json).read_text()):
        stem = str(det["image_id"])
        if stem not in stem_to_id:
            unmatched.add(stem)
            continue
        detections.append({
            "image_id": stem_to_id[stem],
            "category_id": int(det["category_id"]),
            "bbox": det["bbox"],
            "score": det["score"],
        })
    if unmatched:
        raise ValueError(f"{len(unmatched)} predicted images are not in the ground truth, e.g. {sorted(unmatched)[:3]}")
    return detections


def evaluate(gt, detections, verbose=True):
    """Run COCOeval and return overall, size-stratified and per-class AP (each as a fraction)."""
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    if not detections:
        raise ValueError("No detections to evaluate")
    coco_dt = coco_gt.loadRes(detections)
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.params.maxDets = MAX_DETS
    E.evaluate()
    E.accumulate()
    if verbose:
        E.summarize()
    else:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            E.summarize()

    # precision has shape [IoU thresholds, recall points, classes, area ranges, maxDets]
    precision = E.eval["precision"][:, :, :, 0, -1]
    per_class = {}
    for k, name in enumerate(CLASS_NAMES):
        p = precision[:, :, k]
        per_class[name] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")

    s = E.stats
    return {
        "AP50-95": s[0], "AP50": s[1], "AP75": s[2],
        "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
        f"AR@{MAX_DETS[-1]}": s[8],
        "per_class_AP50-95": per_class,
    }
