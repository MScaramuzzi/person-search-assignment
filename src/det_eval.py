"""Detector evaluation: recall at IoU 0.5, and the detection cache.

**Why recall and not COCO mAP.**  The person-search metric matches a detection
to a ground-truth box at ``iou_thresh = min(0.5, w*h / ((w+10)*(h+10)))`` --
never stricter than 0.5, and *more lenient* for small boxes -- and it multiplies
each query's average precision by the detector's recall on that identity.  A
detection the detector never produces is a recall loss the ranking can never
repair, while box tightness beyond IoU 0.5 buys nothing.  Selecting the detector
checkpoint on COCO mAP (or worse, AP@0.75) optimises something the final metric
cannot see; a prior project did exactly that.

Primary selection is recall at detection score >= 0.5; recall at >= 0.05 is
logged too, as the ceiling the cached inference pass can ever reach.
"""

from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of xyxy boxes: (N,4), (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)).astype(np.float32)


def match_greedy(dets: np.ndarray, gts: np.ndarray, iou_thr: float = 0.5) -> int:
    """Number of GT boxes matched one-to-one by detections, best IoU first.

    ``dets`` is (n,5) xyxy+score, already score-filtered; ``gts`` is (m,4) xyxy.
    One detection can claim at most one GT and vice versa, so duplicate boxes on
    the same person cannot inflate the count.
    """
    if len(dets) == 0 or len(gts) == 0:
        return 0
    ious = box_iou(dets[:, :4], gts)
    matched = 0
    used_gt = np.zeros(len(gts), dtype=bool)
    used_det = np.zeros(len(dets), dtype=bool)
    order = np.dstack(np.unravel_index(np.argsort(ious, axis=None)[::-1], ious.shape))[0]
    for di, gi in order:
        if ious[di, gi] < iou_thr:
            break
        if used_det[di] or used_gt[gi]:
            continue
        used_det[di] = used_gt[gi] = True
        matched += 1
    return matched


def recall_at_iou(
    dets_per_frame: list[np.ndarray],
    gts_per_frame: list[np.ndarray],
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
) -> dict:
    """Detection recall over a set of frames -- the checkpoint criterion.

    Returns recall plus the raw counts, and the mean number of detections kept
    per frame (how many crops the re-ID stage would have to embed).
    """
    assert len(dets_per_frame) == len(gts_per_frame)
    n_matched = n_gt = n_det = 0
    for dets, gts in zip(dets_per_frame, gts_per_frame):
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 5)
        gts = np.asarray(gts, dtype=np.float32).reshape(-1, 4)
        kept = dets[dets[:, 4] >= score_thr]
        n_matched += match_greedy(kept, gts, iou_thr)
        n_gt += len(gts)
        n_det += len(kept)
    return {
        "recall": n_matched / n_gt if n_gt else 0.0,
        "matched": n_matched,
        "n_gt": n_gt,
        "n_det": n_det,
        "det_per_frame": n_det / max(len(dets_per_frame), 1),
        "score_thr": score_thr,
        "iou_thr": iou_thr,
    }


def recall_table(
    dets_per_frame: list[np.ndarray],
    gts_per_frame: list[np.ndarray],
    score_thrs=(0.05, 0.5),
    iou_thr: float = 0.5,
) -> list[dict]:
    """Recall at several score thresholds, from one cached detection pass."""
    return [recall_at_iou(dets_per_frame, gts_per_frame, iou_thr, s) for s in score_thrs]


def print_recall_table(rows: list[dict], title: str = "") -> None:
    if title:
        print(title)
    print(f"  {'score>=':>8}  {'recall':>7}  {'matched/GT':>14}  {'det/frame':>9}")
    for r in rows:
        counts = "{}/{}".format(r["matched"], r["n_gt"])
        print(
            f"  {r['score_thr']:>8.2f}  {r['recall']:>7.4f}  {counts:>14}  "
            f"{r['det_per_frame']:>9.2f}"
        )
