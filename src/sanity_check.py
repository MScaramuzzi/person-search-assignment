"""Dataset gate: parse the whole PRW distribution once, assert it, report it.

Run this before any training.  It fails loudly on structural problems (fatal
asserts) and prints a stats table for everything that is merely *expected*
(actual vs. the brief's numbers), because the brief's own FAQ states that
table-vs-data discrepancies exist and must be handled by the student.

Every step is exposed as a function so the notebook can call the same code that
the command line runs -- :func:`parse_dataset`, :func:`check_dataset`,
:func:`print_distributions`, :func:`render_sample_frames` -- with no logic
duplicated between the two.  The printed table is the notebook's
dataset-exploration section, and the sample-frame figure is a static matplotlib
figure (no ipywidgets anywhere in graded output).

Usage::

    python src/sanity_check.py --root data --fig-dir figures
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from prw_dataset import (
    UNLABELED_PID,
    FrameAnn,
    Query,
    build_id_map,
    load_frame_anns,
    load_id_list,
    load_queries,
    load_split,
    save_id_map,
)

# Nominal statistics from the assignment brief / dataset readme, for comparison.
BRIEF = {
    "frames": 11816,
    "boxes": 43110,
    "ids": 932,
    "train_frames": 5704,
    "train_ids": 482,
    "test_frames": 6112,
    "queries": 2057,
    "unlabeled_train_crops": 2827,
    "test_boxes_with_id": 19127,
    "test_boxes_without_id": 5935,
}

MIN_CROP_HEIGHT = 24  # training crops are kept down to this height (decision #12)


@dataclass
class ParsedPRW:
    """Everything the dataset layer produces, parsed once and reused."""

    root: str
    train_names: list[str]
    test_names: list[str]
    train_anns: list[FrameAnn]
    test_anns: list[FrameAnn]
    queries: list[Query]
    id_train: np.ndarray
    id_test: np.ndarray
    key_counts: Counter = field(default_factory=Counter)
    res_counts: Counter = field(default_factory=Counter)

    @property
    def all_anns(self) -> list[FrameAnn]:
        return self.train_anns + self.test_anns


class Report:
    """Collects fatal checks and statistics, then prints them as one table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.failures: list[str] = []

    def fatal(self, name: str, ok: bool, detail: str) -> None:
        if not ok:
            self.failures.append(f"{name}: {detail}")
        self.rows.append(("FATAL" if not ok else "ok", name, detail, ""))

    def stat(self, name: str, actual, expected=None) -> None:
        exp = "" if expected is None else str(expected)
        if expected is None:
            status = "--"
        else:
            status = "ok" if str(actual) == exp else "DIFFERS"
        self.rows.append((status, name, str(actual), exp))

    @property
    def ok(self) -> bool:
        return not self.failures

    def print(self) -> None:
        w_name = max(len(r[1]) for r in self.rows)
        w_act = max(len(r[2]) for r in self.rows)
        header = f"{'':>7}  {'check':<{w_name}}  {'actual':<{w_act}}  brief"
        print(header)
        print("-" * len(header))
        for status, name, actual, expected in self.rows:
            print(f"{status:>7}  {name:<{w_name}}  {actual:<{w_act}}  {expected}")
        print("-" * len(header))
        if self.failures:
            print(f"FAILED {len(self.failures)} fatal check(s):")
            for f in self.failures:
                print("  -", f)
        else:
            print("all fatal checks passed")


def parse_dataset(root: str = "data", cache_dir: str | None = "cache", verbose: bool = True) -> ParsedPRW:
    """Read splits, all 11,816 annotations and the queries.

    The first run parses the raw ``.mat`` files (~2.5 min) and writes an ``.npz``
    cache under ``cache_dir``; later runs load the cache in a second.
    """
    t0 = time.time()
    train_names = load_split(os.path.join(root, "frame_train.mat"))
    test_names = load_split(os.path.join(root, "frame_test.mat"))
    if verbose:
        print(f"parsing {len(train_names) + len(test_names)} frames ...", flush=True)

    def cache_path(name: str) -> str | None:
        return os.path.join(cache_dir, name) if cache_dir else None

    train_anns, train_keys, train_res = load_frame_anns(
        root, train_names, cache_path=cache_path("train_anns.npz"),
        progress_every=2000 if verbose else 0,
    )
    test_anns, test_keys, test_res = load_frame_anns(
        root, test_names, cache_path=cache_path("test_anns.npz"),
        progress_every=2000 if verbose else 0,
    )
    parsed = ParsedPRW(
        root=root,
        train_names=train_names,
        test_names=test_names,
        train_anns=train_anns,
        test_anns=test_anns,
        queries=load_queries(os.path.join(root, "query_info.txt")),
        id_train=load_id_list(os.path.join(root, "ID_train.mat")),
        id_test=load_id_list(os.path.join(root, "ID_test.mat")),
        key_counts=train_keys + test_keys,
        res_counts=train_res + test_res,
    )
    if verbose:
        print(f"parsed in {time.time() - t0:.1f}s")
    return parsed


def crop_heights(anns: list[FrameAnn], labeled: bool) -> np.ndarray:
    """Heights of labeled (or unlabeled) crops, in original-frame pixels."""
    hs = []
    for ann in anns:
        sel = ann.pids > 0 if labeled else ann.pids == UNLABELED_PID
        if sel.any():
            b = ann.boxes[sel]
            hs.append(b[:, 3] - b[:, 1])
    return np.concatenate(hs) if hs else np.zeros(0, dtype=np.float32)


def check_dataset(d: ParsedPRW) -> Report:
    """Fatal structural checks + the actual-vs-brief statistics table."""
    rep = Report()
    root = d.root

    # ---------------------------------------------------------------- layout
    for sub in ("frames", "annotations", "query_box"):
        rep.fatal(f"dir {sub}/ exists", os.path.isdir(os.path.join(root, sub)), sub)
    n_anno = len([f for f in os.listdir(os.path.join(root, "annotations")) if f.endswith(".mat")])
    n_frames = len([f for f in os.listdir(os.path.join(root, "frames")) if f.endswith(".jpg")])
    rep.fatal("annotation files == 11816", n_anno == BRIEF["frames"], str(n_anno))
    rep.fatal("frame files == 11816", n_frames == BRIEF["frames"], str(n_frames))

    # ---------------------------------------------------------------- splits
    rep.fatal("len(train) == 5704", len(d.train_names) == 5704, str(len(d.train_names)))
    rep.fatal("len(test) == 6112", len(d.test_names) == 6112, str(len(d.test_names)))
    overlap = set(d.train_names) & set(d.test_names)
    rep.fatal("train and test are disjoint", not overlap, f"{len(overlap)} shared frames")
    rep.fatal(
        "train + test == all frames",
        len(set(d.train_names) | set(d.test_names)) == n_frames,
        str(len(set(d.train_names) | set(d.test_names))),
    )

    # --------------------------------------------------------------- queries
    rep.fatal("len(queries) == 2057", len(d.queries) == 2057, str(len(d.queries)))
    test_set = set(d.test_names)
    missing = [q.img_name for q in d.queries if q.img_name not in test_set]
    rep.fatal("every query frame is a test frame", not missing, f"{len(missing)} missing")
    on_disk = [
        q.img_name for q in d.queries
        if not os.path.exists(os.path.join(root, "frames", q.img_name))
    ]
    rep.fatal("every query frame exists on disk", not on_disk, f"{len(on_disk)} missing")
    rep.fatal(
        "every query pid > 0",
        all(q.pid > 0 for q in d.queries),
        f"min pid {min(q.pid for q in d.queries)}",
    )
    rep.fatal(
        "every query box has positive area",
        all((q.box[2] > q.box[0]) and (q.box[3] > q.box[1]) for q in d.queries),
        "w > 0 and h > 0",
    )

    # ------------------------------------------------------------ statistics
    train_boxes = sum(a.n_boxes for a in d.train_anns)
    test_boxes = sum(a.n_boxes for a in d.test_anns)
    train_pids = np.concatenate([a.pids for a in d.train_anns])
    test_pids = np.concatenate([a.pids for a in d.test_anns])
    train_ids = sorted({int(p) for p in train_pids if p > 0})
    test_ids = sorted({int(p) for p in test_pids if p > 0})
    query_ids = sorted({q.pid for q in d.queries})
    lab_h = crop_heights(d.train_anns, labeled=True)
    unlab_h = crop_heights(d.train_anns, labeled=False)

    rep.stat("total boxes (train+test)", train_boxes + test_boxes, BRIEF["boxes"])
    rep.stat("train frames", len(d.train_anns), BRIEF["train_frames"])
    rep.stat("test / gallery frames", len(d.test_anns), BRIEF["test_frames"])
    rep.stat("train boxes", train_boxes)
    rep.stat("test boxes", test_boxes)
    rep.stat("mean boxes / frame", f"{(train_boxes + test_boxes) / n_frames:.2f}")
    rep.stat("labeled train identities (L)", len(train_ids), BRIEF["train_ids"])
    rep.stat("labeled train crops", int(lab_h.size))
    rep.stat(f"labeled train crops h >= {MIN_CROP_HEIGHT}", int((lab_h >= MIN_CROP_HEIGHT).sum()))
    rep.stat("unlabeled train crops (pid=-2)", int(unlab_h.size), BRIEF["unlabeled_train_crops"])
    rep.stat(
        f"unlabeled train crops h >= {MIN_CROP_HEIGHT}", int((unlab_h >= MIN_CROP_HEIGHT).sum())
    )
    rep.stat("gallery identities (test, pid>0)", len(test_ids))
    rep.stat("test boxes with id", int((test_pids > 0).sum()), BRIEF["test_boxes_with_id"])
    rep.stat(
        "test boxes without id (-2)",
        int((test_pids == UNLABELED_PID).sum()),
        BRIEF["test_boxes_without_id"],
    )
    rep.stat("queries", len(d.queries), BRIEF["queries"])
    rep.stat("query identities", len(query_ids))
    rep.stat("query ids seen in train", len(set(query_ids) & set(train_ids)), 0)
    rep.stat("max raw pid in dataset", int(max(train_pids.max(), test_pids.max())), BRIEF["ids"])
    rep.stat("min labeled crop height (train, px)", f"{lab_h.min():.1f}" if lab_h.size else "-")

    # ID_*.mat are informational, but a mismatch would signal a corrupt download.
    rep.stat("ID_train.mat entries", d.id_train.size, len(train_ids))
    rep.stat("ID_test.mat entries", d.id_test.size, len(query_ids))
    rep.stat("ID_train.mat == parsed train ids", sorted(d.id_train.tolist()) == train_ids, True)
    rep.stat("ID_test.mat == query ids", sorted(d.id_test.tolist()) == query_ids, True)
    return rep


def print_distributions(d: ParsedPRW) -> None:
    """The dataset quirks worth one sentence each in the notebook."""
    print("box-key distribution (the key genuinely varies per file):")
    for key, count in d.key_counts.most_common():
        print(f"  {key:<14} {count:>6}")

    print("\nframe-resolution distribution:")
    for (w, h), count in d.res_counts.most_common():
        print(f"  {f'{w}x{h}':<14} {count:>6}")

    print("\nboxes per frame, percentiles (all frames):")
    per_frame = np.array([a.n_boxes for a in d.all_anns])
    for q in (0, 25, 50, 75, 100):
        print(f"  p{q:<3} {np.percentile(per_frame, q):.0f}")

    print("\ncrops per labeled train identity:")
    train_pids = np.concatenate([a.pids for a in d.train_anns])
    counts = np.array(list(Counter(int(p) for p in train_pids if p > 0).values()))
    print(f"  min {counts.min()}  median {np.median(counts):.0f}  max {counts.max()}")
    print(f"  identities with < 4 crops (PK sampling with K=4): {(counts < 4).sum()}")


def render_sample_frames(
    root: str,
    anns: list[FrameAnn],
    out_path: str | None = None,
    n: int = 3,
    seed: int = 42,
):
    """Draw GT boxes on n random frames: labeled green, unlabeled (pid=-2) orange.

    This is the visual check that the xywh -> xyxy conversion is correct.
    Returns the matplotlib figure so a notebook can display it inline.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    picks = random.Random(seed).sample([a for a in anns if a.n_boxes >= 2], n)

    fig, axes = plt.subplots(n, 1, figsize=(12, 6.5 * n))
    for ax, ann in zip(np.atleast_1d(axes), picks):
        img = Image.open(os.path.join(root, "frames", ann.img_name))
        ax.imshow(img)
        for box, pid in zip(ann.boxes, ann.pids):
            labeled = pid > 0
            color = "lime" if labeled else "orange"
            x1, y1, x2, y2 = box
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
                )
            )
            ax.text(
                x1,
                max(y1 - 4, 8),
                f"id {int(pid)}" if labeled else "unlabeled",
                color="black",
                fontsize=8,
                bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "edgecolor": "none"},
            )
        ax.set_title(
            f"{ann.img_name}  |  {img.size[0]}x{img.size[1]}  |  "
            f"{ann.n_boxes} boxes  |  key={ann.box_key}",
            fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data", help="PRW root (frames/, annotations/, ...)")
    ap.add_argument("--id-map", default="id_map.json", help="where to write the id map")
    ap.add_argument("--cache-dir", default="cache", help="parsed-annotation cache (.npz)")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="do not write id_map.json")
    args = ap.parse_args()

    parsed = parse_dataset(args.root, cache_dir=args.cache_dir)
    rep = check_dataset(parsed)
    rep.print()
    print()
    print_distributions(parsed)

    id_map = build_id_map(parsed.train_anns)
    print(f"\nid map: {len(id_map)} labeled train identities -> [0, {len(id_map) - 1}]")
    if not args.no_write:
        save_id_map(id_map, args.id_map)
        print(f"wrote {args.id_map}")

    if not args.no_figure:
        import matplotlib

        matplotlib.use("Agg")
        out = os.path.join(args.fig_dir, "sample_frames_gt.png")
        render_sample_frames(args.root, parsed.train_anns, out_path=out, seed=args.seed)
        print(f"wrote {out}  (labeled = green, pid=-2 = orange)")

    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
