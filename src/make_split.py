"""Build the self-made train/validation split of the PRW training frames.

PRW ships no validation split -- the assignment's FAQ confirms the 570-frame
split named in the brief does not exist in the published data -- so we build one
and freeze it in ``split_v1.json``.  Every later component (detector training,
re-ID training, checkpoint selection) loads that file; the split is never
regenerated.

Two properties, both deliberate:

* **frame-level, not crop-level.**  Splitting crops would put different crops of
  the same person, often from the same scene, on both sides: the validation
  score would then measure memorisation rather than generalisation.
* **identity-coverage-aware.**  A frame may move to validation only if every
  labeled identity it contains keeps at least one frame on the training side.
  A plain random draw can strand a rare identity entirely in validation, which
  removes it from the OIM lookup table and from the cross-entropy classifier.
  Scarce identities therefore have priority on the training side.

Identities appearing in exactly one training frame can never cross, so the
validation set covers slightly fewer identities than the training set.

Usage::

    python src/make_split.py --root data --out split_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from prw_dataset import load_frame_anns, load_split

SPLIT_RULE = "id_coverage_v1"


def labeled_ids(ann) -> set[int]:
    """Labeled identities present in a frame (unlabeled ``-2`` crops ignored)."""
    return {int(p) for p in ann.pids if p > 0}


def make_split(anns, val_size: int = 570, seed: int = 42) -> tuple[list[str], list[str]]:
    """Split frame names into (train, val) under the identity-coverage rule.

    A seeded shuffle decides the order in which frames are *offered* to the
    validation set; a frame is accepted only if every identity it contains still
    has another frame left on the training side.
    """
    remaining = Counter()
    for ann in anns:
        for pid in labeled_ids(ann):
            remaining[pid] += 1

    order = list(range(len(anns)))
    random.Random(seed).shuffle(order)

    val_idx: set[int] = set()
    for i in order:
        if len(val_idx) >= val_size:
            break
        ids = labeled_ids(anns[i])
        if all(remaining[pid] > 1 for pid in ids):
            val_idx.add(i)
            for pid in ids:
                remaining[pid] -= 1

    train = [a.img_name for i, a in enumerate(anns) if i not in val_idx]
    val = [a.img_name for i, a in enumerate(anns) if i in val_idx]
    return train, val


def summarize(anns, train: list[str], val: list[str]) -> dict:
    by_name = {a.img_name: a for a in anns}
    train_ids = set().union(*(labeled_ids(by_name[n]) for n in train))
    val_ids = set().union(*(labeled_ids(by_name[n]) for n in val)) if val else set()
    train_crops = sum(int((by_name[n].pids > 0).sum()) for n in train)
    val_crops = sum(int((by_name[n].pids > 0).sum()) for n in val)
    train_unlab = sum(int((by_name[n].pids == -2).sum()) for n in train)
    val_unlab = sum(int((by_name[n].pids == -2).sum()) for n in val)
    return {
        "train_frames": len(train),
        "val_frames": len(val),
        "train_ids": len(train_ids),
        "val_ids": len(val_ids),
        "ids_only_in_train": len(train_ids - val_ids),
        "ids_only_in_val": len(val_ids - train_ids),
        "train_labeled_crops": train_crops,
        "val_labeled_crops": val_crops,
        "train_unlabeled_crops": train_unlab,
        "val_unlabeled_crops": val_unlab,
        "val_ids_with_ge2_crops": sum(
            1
            for pid in val_ids
            if sum(int((by_name[n].pids == pid).sum()) for n in val) >= 2
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="split_v1.json")
    ap.add_argument("--cache", default="cache/train_anns.npz")
    ap.add_argument("--val-size", type=int, default=570)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} already exists -- the split is frozen by design; use --force to redo.")
        return 0

    train_names = load_split(os.path.join(args.root, "frame_train.mat"))
    anns, _, _ = load_frame_anns(args.root, train_names, cache_path=args.cache, progress_every=2000)

    train, val = make_split(anns, val_size=args.val_size, seed=args.seed)
    stats = summarize(anns, train, val)

    # Asserts: the split is only useful if these hold.
    assert len(train) + len(val) == len(anns)
    assert not (set(train) & set(val)), "a frame ended up on both sides"
    assert stats["ids_only_in_val"] == 0, "an identity has no training frame left"
    assert stats["val_frames"] == args.val_size, f"val has {stats['val_frames']} frames"

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": args.seed,
                "rule": SPLIT_RULE,
                "val_size": args.val_size,
                "train": train,
                "val": val,
            },
            fh,
            indent=1,
        )

    width = max(len(k) for k in stats)
    for key, value in stats.items():
        print(f"  {key:<{width}}  {value}")
    print(f"\nwrote {args.out}  (rule={SPLIT_RULE}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
