"""Pedestrian detector: torchvision Faster R-CNN R50-FPN fine-tuned on PRW.

Design choices and their reasons (they are what the oral exam asks about):

* **Faster R-CNN R50-FPN, COCO weights.**  The backbone the person-search
  literature standardises on (SeqNet, NAE, COAT all build on it), so our numbers
  are comparable, and it adds no dependency.
* **min_size=900, max_size=1500, aspect preserving.**  1500x900 is the standard
  PRW inference size.  Squashing frames into a square destroys small pedestrians,
  which are exactly the boxes the gallery is full of.
* **Two classes, background + person.**  Every annotated box is a pedestrian,
  including the ``pid = -2`` ones: "identity unknown" is a re-ID statement, not a
  detection statement, and dropping those boxes would teach the detector that
  real people are background.
* **trainable_backbone_layers=3** (layer2-4 unfrozen, stem+layer1 frozen,
  frozen batch-norm throughout, as torchvision does by default for detection).

Inference configuration is frozen for the whole project: ``box_score_thresh
=0.05``, ``box_detections_per_img=50``.  One pass over gallery and query frames
is cached and shared by every experiment row, because detections are a property
of the detector alone -- the ranking never sees detection scores unless we
deliberately let it in (the CWS ablation).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)
from torchvision.transforms import functional as TF

# Class index of "person" in the two-class fine-tuned model and in COCO alike:
# torchvision's COCO categories are ['__background__', 'person', ...].
PERSON_CLASS = 1

INFER_SCORE_THRESH = 0.05
INFER_MAX_DETS = 50


class PRWDetectionDataset(torch.utils.data.Dataset):
    """Frames + all their boxes as class ``person``.

    Boxes stay in original image coordinates; torchvision's internal
    ``GeneralizedRCNNTransform`` rescales images and targets together, so no
    manual resizing happens here.  The only training augmentation is a
    horizontal flip, applied to image and boxes consistently.
    """

    def __init__(self, root: str, anns: list, train: bool = False, hflip_prob: float = 0.5):
        self.root = root
        self.anns = anns
        self.train = train
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int):
        ann = self.anns[idx]
        img = Image.open(os.path.join(self.root, "frames", ann.img_name)).convert("RGB")
        boxes = torch.as_tensor(np.asarray(ann.boxes, dtype=np.float32)).reshape(-1, 4)

        if self.train and torch.rand(1).item() < self.hflip_prob:
            img = TF.hflip(img)
            width = img.width
            boxes = boxes.clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        target = {
            "boxes": boxes,
            "labels": torch.full((len(boxes),), PERSON_CLASS, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return TF.to_tensor(img), target


def collate_fn(batch):
    """Frames have different sizes, so they are kept as a list, not stacked."""
    return tuple(zip(*batch))


def build_detector(
    num_classes: int = 2,
    min_size: int = 900,
    max_size: int = 1500,
    trainable_backbone_layers: int = 3,
    pretrained: bool = True,
    score_thresh: float = INFER_SCORE_THRESH,
    max_dets: int = INFER_MAX_DETS,
):
    """Faster R-CNN R50-FPN with a two-class box predictor.

    ``num_classes=91`` keeps the original COCO head, which is how the
    no-fine-tuning baseline is evaluated (person is class 1 there too).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=score_thresh,
        box_detections_per_img=max_dets,
    )
    if num_classes != 91:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def predict(
    model,
    dataset: PRWDetectionDataset,
    device,
    batch_size: int = 4,
    num_workers: int = 2,
    keep_label: int = PERSON_CLASS,
    amp: bool = True,
    progress_every: int = 0,
) -> list[np.ndarray]:
    """Run the detector over a dataset -> one (n,5) xyxy+score array per frame.

    Frames keep the dataset's order, so the arrays align with ``dataset.anns``.
    """
    model.eval().to(device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    out: list[np.ndarray] = []
    use_amp = amp and device.type == "cuda"
    # inference_mode, not just eval(): this same function runs the end-of-epoch
    # validation pass while the optimiser state is still resident on the GPU, and
    # building an autograd graph for 1500x900 frames there is how a T4 runs out
    # of memory eight minutes into epoch 1.
    with torch.inference_mode():
        for i, (images, _) in enumerate(loader, 1):
            images = [im.to(device, non_blocking=True) for im in images]
            with torch.autocast("cuda", enabled=use_amp):
                preds = model(images)
            for p in preds:
                keep = p["labels"] == keep_label
                boxes = p["boxes"][keep].float().cpu().numpy()
                scores = p["scores"][keep].float().cpu().numpy()[:, None]
                out.append(np.hstack([boxes, scores]).astype(np.float32))
            if progress_every and i % progress_every == 0:
                print(f"  {i * batch_size}/{len(dataset)} frames", flush=True)
    return out


def save_detections(dets: list[np.ndarray], names: list[str], path: str) -> None:
    """Cache a detection pass as one compressed ``.npz`` keyed by frame name."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = np.array([len(d) for d in dets], dtype=np.int64)
    np.savez_compressed(
        path,
        names=np.array(names),
        offsets=np.concatenate([[0], np.cumsum(counts)]),
        dets=np.concatenate(dets).astype(np.float32) if dets else np.zeros((0, 5), np.float32),
    )


def load_detections(path: str) -> dict[str, np.ndarray]:
    """Frame name -> (n,5) detections, as written by :func:`save_detections`."""
    with np.load(path, allow_pickle=False) as z:
        off, dets = z["offsets"], z["dets"]
        return {str(n): dets[off[i]: off[i + 1]] for i, n in enumerate(z["names"])}


def load_checkpoint(path: str, map_location=None) -> dict:
    """``torch.load`` that works on both the local torch 1.12 and Colab's 2.x.

    torch >= 2.6 defaults to ``weights_only=True``, which refuses the
    ``collections.Counter`` inside ``MultiStepLR.state_dict()`` -- resuming a run
    would die on the scheduler, not the weights.  torch 1.12 has no such kwarg
    and forwards it to pickle, hence the fallback.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _grad_scaler():
    """AMP scaler across torch versions (``torch.cuda.amp`` is deprecated in 2.4+)."""
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def train_one_epoch(model, loader, optimizer, device, epoch: int, scaler=None,
                    print_every: int = 100, clip_norm: float | None = 10.0) -> dict:
    """One pass over the training frames; returns the mean of each loss term."""
    model.train()
    sums: dict[str, float] = {}
    n = 0
    t0 = time.time()
    for it, (images, targets) in enumerate(loader, 1):
        images = [im.to(device, non_blocking=True) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast("cuda", enabled=scaler is not None):
            losses = model(images, targets)
            loss = sum(losses.values())

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        n += 1
        sums["loss"] = sums.get("loss", 0.0) + loss.detach().item()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + v.detach().item()
        if print_every and it % print_every == 0:
            print(
                f"  ep {epoch} it {it}/{len(loader)}  loss {sums['loss'] / n:.4f}  "
                f"({(time.time() - t0) / n:.2f}s/it)",
                flush=True,
            )
    return {k: v / max(n, 1) for k, v in sums.items()}


def overfit_smoke(model, dataset, device, n_frames: int = 8, iters: int = 20, lr: float = 0.005):
    """Pre-gate before spending GPU-hours: can the loss drop on 8 frames?

    A loop that cannot overfit a handful of frames has a bug in the targets, the
    coordinate convention, or the optimiser -- and it is far cheaper to find that
    here than after a 12-epoch run.
    """
    subset = torch.utils.data.Subset(dataset, list(range(n_frames)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    model.train().to(device)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=5e-4
    )
    history = []
    it = 0
    while it < iters:
        for images, targets in loader:
            if it >= iters:
                break
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append(loss.detach().item())
            it += 1
    return history


def train_detector(
    root: str,
    train_anns: list,
    val_anns: list,
    out_dir: str = "checkpoints/detector",
    epochs: int = 12,
    batch_size: int = 4,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    milestones=(8,),
    num_workers: int = 2,
    device=None,
    amp: bool = True,
    resume: bool = True,
    seed: int = 42,
) -> str:
    """Fine-tune the detector, checkpointing and logging every epoch.

    The learning rate is torchvision's reference detection recipe (0.02 at batch
    16) rescaled linearly to batch 4.  Selection is by validation recall at IoU
    0.5, score >= 0.5 -- never COCO mAP (see ``det_eval``).  Colab sessions die,
    so every epoch writes a checkpoint and a CSV row, and ``resume`` picks up
    from the last one.
    """
    # Works whether the caller put ``src/`` on sys.path or imports ``src.detector``.
    try:
        from seed import loader_generator, set_seed, worker_init_fn
        from det_eval import recall_at_iou
    except ImportError:
        from src.seed import loader_generator, set_seed, worker_init_fn
        from src.det_eval import recall_at_iou

    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "log.csv")
    last_path = os.path.join(out_dir, "last.pt")
    best_path = os.path.join(out_dir, "best.pt")

    train_ds = PRWDetectionDataset(root, train_anns, train=True)
    val_ds = PRWDetectionDataset(root, val_anns, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
        generator=loader_generator(seed), worker_init_fn=worker_init_fn,
    )

    model = build_detector().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=0.1)
    scaler = _grad_scaler() if (amp and device.type == "cuda") else None

    start_epoch, best_recall = 1, -1.0
    if resume and os.path.exists(last_path):
        ckpt = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("best_recall", -1.0)
        print(f"resumed from {last_path} at epoch {start_epoch} (best recall {best_recall:.4f})")

    val_gts = [a.boxes for a in val_anns]
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        losses = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
        scheduler.step()

        val_dets = predict(model, val_ds, device, batch_size=batch_size, num_workers=num_workers)
        r50 = recall_at_iou(val_dets, val_gts, score_thr=0.5)
        r05 = recall_at_iou(val_dets, val_gts, score_thr=0.05)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{k: round(v, 5) for k, v in losses.items()},
            "val_recall@0.5": round(r50["recall"], 5),
            "val_recall@0.05": round(r05["recall"], 5),
            "val_det_per_frame@0.5": round(r50["det_per_frame"], 3),
            "minutes": round((time.time() - t0) / 60, 2),
        }
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_recall": max(best_recall, r50["recall"]),
        }
        torch.save(state, last_path)
        if r50["recall"] > best_recall:
            best_recall = r50["recall"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "recall": best_recall}, best_path)
            print(f"  new best: recall@0.5 = {best_recall:.4f} (epoch {epoch})")

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "epochs": epochs, "batch_size": batch_size, "lr": lr,
                "milestones": list(milestones), "seed": seed,
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "best_recall@0.5": best_recall,
            },
            fh, indent=1,
        )
    print(f"done. best val recall@0.5 = {best_recall:.4f} -> {best_path}")
    return best_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="split_v1.json")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="checkpoints/detector")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from prw_dataset import load_frame_anns, load_split

    with open(args.split, encoding="utf-8") as fh:
        split = json.load(fh)
    names = load_split(os.path.join(args.root, "frame_train.mat"))
    anns, _, _ = load_frame_anns(
        args.root, names, cache_path=os.path.join(args.cache_dir, "train_anns.npz")
    )
    by_name = {a.img_name: a for a in anns}
    train_anns = [by_name[n] for n in split["train"]]
    val_anns = [by_name[n] for n in split["val"]]
    print(f"train frames {len(train_anns)} | val frames {len(val_anns)}")

    train_detector(
        args.root, train_anns, val_anns, out_dir=args.out_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
        amp=not args.no_amp, resume=not args.no_resume, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
