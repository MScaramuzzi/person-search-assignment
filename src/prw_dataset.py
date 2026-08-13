"""PRW dataset conversion layer.

Single source of truth for reading the raw PRW distribution (MATLAB ``.mat``
annotations + ``query_info.txt``) into plain numpy structures.  Every other
module in this project reads the dataset through this file and nowhere else.

Two identity spaces are kept strictly separate (see the handoff plan, section 4.2):

* **RAW pids** -- the values stored in the dataset, including the ``-2``
  "ambiguous person" sentinel.  These are what the evaluation function compares,
  so anything that touches evaluation uses raw pids.
* **Contiguous ids** -- ``[0 .. L-1]`` plus ``-1`` for unlabeled crops, produced
  by :func:`build_id_map` and persisted to ``id_map.json``.  These are training
  only (cross-entropy targets, OIM lookup-table rows).

Verified facts about this copy of the data (measured, not assumed):

* the per-frame box key varies: ``box_new`` (11,792 files), ``anno_file`` (23),
  ``anno_previous`` (1) -- hence the per-file fallback chain, never a pinned key;
* box rows are ``[pid, x, y, w, h]`` with float coordinates;
* negative ``x`` coordinates exist (35 files, minimum -4.04), so coordinates are
  clipped at 0 before the xywh -> xyxy conversion;
* split ``.mat`` files store double-nested frame names without extension;
* ``ID_test.mat`` stores its array under the key ``ID_test2``;
* ``query_info.txt`` uses CRLF line endings and frame names without extension.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.io import loadmat

# Fallback chain for the per-frame box array, in resolution order.
BOX_KEYS = ("box_new", "anno_file", "anno_previous")

# Raw pid used by PRW for "person present, identity unknown".
UNLABELED_PID = -2

# Contiguous label assigned to unlabeled crops in the training id space.
UNLABELED_LABEL = -1

_CAM_RE = re.compile(r"^c(\d+)s")


@dataclass
class FrameAnn:
    """Ground-truth annotation of one full scene frame."""

    img_name: str  # "c1s1_000151.jpg"
    cam_id: int  # parsed from the file name; -1 if unparsable
    boxes: np.ndarray  # (N, 4) float32, xyxy, clipped to the image bounds
    pids: np.ndarray  # (N,) int32, RAW ids (-2 preserved)
    box_key: str = ""  # which key this frame's boxes came from (for reporting)
    img_wh: tuple[int, int] = (0, 0)  # PRW mixes 1920x1080 and 720x576

    @property
    def n_boxes(self) -> int:
        return int(self.boxes.shape[0])


@dataclass
class Query:
    """One line of ``query_info.txt``."""

    pid: int  # RAW id
    box: np.ndarray  # (4,) float32, xyxy
    img_name: str  # frame name parsed verbatim from the txt, plus ".jpg"


def parse_cam_id(img_name: str) -> int:
    """Camera id from a PRW frame name (``c1s1_000151.jpg`` -> 1)."""
    m = _CAM_RE.match(img_name)
    return int(m.group(1)) if m else -1


def image_size(img_path: str) -> tuple[int, int]:
    """(width, height) of an image, read from the header only.

    PRW mixes resolutions (1920x1080 and 720x576), so sizes are always read
    per image and never assumed.
    """
    with Image.open(img_path) as im:
        return im.size


def load_split(mat_path: str) -> list[str]:
    """Frame names of a split file (``frame_train.mat`` / ``frame_test.mat``).

    The names are stored double-nested in a MATLAB cell array and carry no
    extension; ``.jpg`` is appended when missing.
    """
    mat = loadmat(mat_path)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{mat_path}: expected one data key, found {keys}")
    names: list[str] = []
    for entry in np.asarray(mat[keys[0]]).ravel():
        name = str(np.asarray(entry).ravel()[0])
        names.append(name if name.lower().endswith(".jpg") else name + ".jpg")
    return names


def load_id_list(mat_path: str) -> np.ndarray:
    """Identity list of ``ID_train.mat`` / ``ID_test.mat`` (informational).

    The key differs between the two files (``ID_train`` vs ``ID_test2``), so it
    is resolved dynamically rather than hardcoded.
    """
    mat = loadmat(mat_path)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{mat_path}: expected one data key, found {keys}")
    return np.asarray(mat[keys[0]]).ravel().astype(np.int32)


def resolve_box_key(anno: dict) -> str:
    """Which of :data:`BOX_KEYS` this annotation file uses.

    The key genuinely varies per file, so it is resolved per file and never
    pinned once for the whole dataset.
    """
    for key in BOX_KEYS:
        if key in anno:
            return key
    present = [k for k in anno if not k.startswith("__")]
    raise KeyError(f"no known box key among {BOX_KEYS}; file has {present}")


def parse_frame_annotation(mat_path: str, img_wh: tuple[int, int]) -> FrameAnn:
    """Read one ``<frame>.jpg.mat`` into a :class:`FrameAnn`.

    Conversion order matters and follows the two working reference
    implementations: clip coordinates at 0 first (negative x values exist in the
    raw data), then convert xywh -> xyxy, then clip against the image bounds,
    then drop degenerate boxes.
    """
    width, height = img_wh
    anno = loadmat(mat_path)
    box_key = resolve_box_key(anno)

    rois = np.asarray(anno[box_key], dtype=np.float32)
    if rois.ndim == 1:  # a single-box file may come back flat
        rois = rois.reshape(1, -1)
    if rois.shape[1] < 5:
        raise ValueError(f"{mat_path}: expected 5 columns [pid x y w h], got {rois.shape}")

    pids = rois[:, 0].astype(np.int32)
    boxes = rois[:, 1:5].copy()

    np.clip(boxes, 0, None, out=boxes)  # negative coordinates exist in the raw data
    boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)

    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    img_name = os.path.basename(mat_path)[: -len(".mat")]
    return FrameAnn(
        img_name=img_name,
        cam_id=parse_cam_id(img_name),
        boxes=np.ascontiguousarray(boxes[keep], dtype=np.float32),
        pids=np.ascontiguousarray(pids[keep], dtype=np.int32),
        box_key=box_key,
        img_wh=(int(width), int(height)),
    )


def parse_frames(
    root: str, frame_names: list[str], progress_every: int = 0
) -> list[FrameAnn]:
    """Parse a list of frames from the raw ``.mat`` files (~13 ms/frame)."""
    anns: list[FrameAnn] = []
    for i, name in enumerate(frame_names, 1):
        wh = image_size(os.path.join(root, "frames", name))
        anns.append(
            parse_frame_annotation(os.path.join(root, "annotations", name + ".mat"), wh)
        )
        if progress_every and i % progress_every == 0:
            print(f"  parsed {i}/{len(frame_names)} frames", flush=True)
    return anns


def save_frame_anns(anns: list[FrameAnn], path: str) -> None:
    """Cache parsed annotations to a compressed ``.npz`` (ragged -> offsets)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = np.array([a.n_boxes for a in anns], dtype=np.int64)
    np.savez_compressed(
        path,
        names=np.array([a.img_name for a in anns]),
        cam_ids=np.array([a.cam_id for a in anns], dtype=np.int32),
        box_keys=np.array([a.box_key for a in anns]),
        img_wh=np.array([a.img_wh for a in anns], dtype=np.int32),
        offsets=np.concatenate([[0], np.cumsum(counts)]),
        boxes=np.concatenate([a.boxes for a in anns]).astype(np.float32),
        pids=np.concatenate([a.pids for a in anns]).astype(np.int32),
    )


def _read_frame_anns(path: str) -> list[FrameAnn]:
    with np.load(path, allow_pickle=False) as z:
        off, boxes, pids = z["offsets"], z["boxes"], z["pids"]
        return [
            FrameAnn(
                img_name=str(name),
                cam_id=int(cam),
                boxes=boxes[off[i] : off[i + 1]],
                pids=pids[off[i] : off[i + 1]],
                box_key=str(key),
                img_wh=(int(wh[0]), int(wh[1])),
            )
            for i, (name, cam, key, wh) in enumerate(
                zip(z["names"], z["cam_ids"], z["box_keys"], z["img_wh"])
            )
        ]


def load_frame_anns(
    root: str,
    frame_names: list[str],
    cache_path: str | None = None,
    progress_every: int = 0,
) -> tuple[list[FrameAnn], Counter, Counter]:
    """Annotations for ``frame_names``, plus box-key and resolution counters.

    Both distributions are reported in the notebook's dataset section.  When
    ``cache_path`` is given, a valid cache (same frames, same order) is reused
    and otherwise rebuilt -- parsing all 11,816 frames takes ~2.5 minutes.
    """
    anns: list[FrameAnn] | None = None
    if cache_path and os.path.exists(cache_path):
        cached = _read_frame_anns(cache_path)
        if [a.img_name for a in cached] == frame_names:
            anns = cached
        else:
            print(f"  cache {cache_path} does not match the requested frames; reparsing")
    if anns is None:
        anns = parse_frames(root, frame_names, progress_every=progress_every)
        if cache_path:
            save_frame_anns(anns, cache_path)

    key_counts = Counter(a.box_key for a in anns)
    res_counts = Counter(a.img_wh for a in anns)
    return anns, key_counts, res_counts


def load_queries(txt_path: str) -> list[Query]:
    """Parse ``query_info.txt`` -- exactly 2,057 entries.

    Line format: ``pid x y w h frame_name``.  The frame name is taken verbatim
    from the line and never reconstructed from the camera id (pitfall P8); the
    file uses CRLF endings and omits the ``.jpg`` extension.
    """
    queries: list[Query] = []
    with open(txt_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"{txt_path}:{lineno}: expected 6 fields, got {len(parts)}")
            pid = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
            name = parts[5]
            if not name.lower().endswith(".jpg"):
                name += ".jpg"
            x, y = max(x, 0.0), max(y, 0.0)  # same clip-low rule as the frame boxes
            queries.append(
                Query(
                    pid=pid,
                    box=np.array([x, y, x + w, y + h], dtype=np.float32),
                    img_name=name,
                )
            )
    return queries


def build_id_map(train_frames: list[FrameAnn]) -> dict[int, int]:
    """RAW labeled pid -> contiguous label in ``[0, L-1]``.

    Sorted by raw pid so the mapping is deterministic.  Sized by the *count* of
    distinct labeled pids, never by the maximum pid value: PRW's raw ids are
    sparse (they reach 933 here while only 483 identities exist in train), and
    sizing by the maximum is the bug that silently pushes identities out of the
    OIM lookup table.  Unlabeled crops are not in the map; they get
    :data:`UNLABELED_LABEL`.
    """
    pids = sorted({int(p) for ann in train_frames for p in ann.pids if p > 0})
    return {pid: i for i, pid in enumerate(pids)}


def save_id_map(id_map: dict[int, int], path: str) -> None:
    """Persist the id map (JSON keys are strings by definition)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in sorted(id_map.items())}, fh, indent=1)


def load_id_map(path: str) -> dict[int, int]:
    """Load the id map written by :func:`save_id_map`, keys back to ``int``."""
    with open(path, "r", encoding="utf-8") as fh:
        return {int(k): int(v) for k, v in json.load(fh).items()}


def to_contiguous(pids: np.ndarray, id_map: dict[int, int]) -> np.ndarray:
    """Map raw pids to training labels, unlabeled -> :data:`UNLABELED_LABEL`."""
    return np.array(
        [id_map.get(int(p), UNLABELED_LABEL) for p in np.asarray(pids).ravel()],
        dtype=np.int64,
    )
