"""Pack PRW plus the parsed-annotation cache into one archive for Colab.

Google Drive charges a round trip per file.  ``data/`` holds 25,689 of them --
11,816 frames (2.7 GB), 11,816 annotation ``.mat`` files (3.5 MB in total, i.e.
~300 bytes each) and 2,057 query crops -- so uploading the folder file by file
takes hours, and reading it back through the Colab FUSE mount is worse: at
~10-100 ms of latency per open, the 300-byte annotation files alone would cost
more wall clock than parsing them ever did.  One archive uploads as a single
stream and unpacks onto the Colab local SSD in a couple of minutes.

Two consequences are baked in here:

* **Stored, not deflated.**  JPEGs are already compressed; deflating 2.7 GB of
  them burns minutes of CPU on both ends to save low single-digit percent.
* **The ``cache/*.npz`` goes in too.**  Those two files (512 KB) are the parsed
  form of all 11,816 ``.mat`` files, they contain no absolute paths, and with
  them present the notebook never opens ``data/annotations/`` at all.

Paths inside the archive are relative to the repo root (``data/...``,
``cache/...``), so unzipping into a fresh clone puts everything exactly where
the notebook's default ``ROOT="data"`` / ``CACHE_DIR="cache"`` expect it.

    python tools/pack_data.py --out PRW.zip
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile


def iter_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def pack(out_path: str, root: str = "data", cache_dir: str = "cache") -> None:
    if not os.path.isdir(root):
        raise SystemExit(f"{root!r} not found -- run this from the repo root")

    members: list[str] = list(iter_files(root))
    if os.path.isdir(cache_dir):
        members += [p for p in iter_files(cache_dir) if p.endswith(".npz")]
    else:
        print(f"note: no {cache_dir}/ yet -- run the notebook's parse cell first "
              "to avoid re-parsing 11,816 .mat files on Colab")

    total = sum(os.path.getsize(p) for p in members)
    print(f"{len(members)} files, {total / 1024 ** 3:.2f} GB -> {out_path}")

    start = time.time()
    # ZIP64 is required: the archive is well past the 4 GB / 65k-entry limits of
    # the original format in file count, and near it in size.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, path in enumerate(members, 1):
            zf.write(path, arcname=path.replace(os.sep, "/"))
            if i % 2000 == 0:
                print(f"  {i}/{len(members)}", flush=True)

    size = os.path.getsize(out_path) / 1024 ** 3
    print(f"wrote {out_path}: {size:.2f} GB in {time.time() - start:.0f}s")
    print("upload it to Drive (one file) and point DRIVE_DIR in the notebook at it")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="PRW.zip")
    args = ap.parse_args()
    pack(args.out, args.root, args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
