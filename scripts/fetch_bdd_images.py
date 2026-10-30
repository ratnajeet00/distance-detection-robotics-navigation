#!/usr/bin/env python
"""Fetch only the images we need from a remote zip, without the whole file.

Reads entries over HTTP Range requests (remotezip) with parallel workers --
range servers can be slow per-stream (~250KB/s on dl.yf.io), so several
concurrent reads restore the aggregate speed.

    python scripts/fetch_bdd_images.py \
        --url http://dl.yf.io/bdd100k/bdd100k_images_100k.zip \
        --names data/bdd/wanted_train.txt \
        --out data/bdd/images/train \
        [--limit N] [--workers 12] [--resize 800]
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


class ZipPool:
    """remotezip instances are not thread-safe; give each thread its own."""

    def __init__(self, url: str):
        self.url = url
        self._tl = threading.local()

    def get(self):
        if not hasattr(self._tl, "z"):
            import remotezip
            self._tl.z = remotezip.RemoteZip(self.url)
        return self._tl.z

    def close_all(self) -> None:
        pass  # GC closes them; RemoteZip sessions die with the thread


def fetch_one(pool: ZipPool, entry: str, dst: Path, resize: int) -> int:
    for attempt in range(4):
        try:
            data = pool.get().read(entry)
            break
        except Exception as e:  # rate limits (429) / transient network
            time.sleep(1.5 * (attempt + 1))
            if hasattr(pool._tl, "z"):
                del pool._tl.z  # rebuild the connection next attempt
    else:
        raise RuntimeError(f"could not fetch {entry}: {e}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if resize and dst.suffix.lower() in (".jpg", ".jpeg", ".png"):
        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None and max(img.shape[:2]) > resize:
                s = resize / max(img.shape[:2])
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                                 interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(dst.suffix, img,
                                       [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    data = buf.tobytes()
        except Exception:
            pass
    dst.write_bytes(data)
    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--names", required=True, help="txt with one image name per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resize", type=int, default=0,
                    help="downscale the long edge to N px before saving")
    ap.add_argument("--log-every", type=int, default=250)
    args = ap.parse_args()

    from remotezip import RemoteZip

    wanted = [l.strip() for l in Path(args.names).read_text().splitlines() if l.strip()]
    if args.limit:
        wanted = wanted[: args.limit]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {len(wanted)} images wanted -> {args.out}", flush=True)

    t0 = time.time()
    saved = missing = bytes_total = 0
    with RemoteZip(args.url) as z_main:
        by_base = {Path(n).name: n for n in z_main.namelist()}
        by_full = {n: n for n in by_base.values()}
        pool = ZipPool(args.url)
        tasks = []
        for base in wanted:
            stem = Path(base).stem
            cands = [base, f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png",
                     Path(base).name, f"{Path(base).stem}.jpg"]
            entry = next((by_full.get(c) or by_base.get(c) for c in cands), None)
            if entry is None:
                missing += 1
                continue
            tasks.append((entry, Path(args.out) / Path(entry).name))

        def work(task):
            entry, dst = task
            try:
                n = fetch_one(pool, entry, dst, args.resize)
                return entry, dst, n
            except Exception as exc:  # keep going; report at the end
                return entry, dst, -1

        failed = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futures = [ex.submit(work, t) for t in tasks]
            for f in as_completed(futures):
                entry, dst, n = f.result()
                if n < 0:
                    failed += 1
                    print(f"[fetch] FAILED {entry}", flush=True)
                    continue
                saved += 1
                bytes_total += n
                if saved % args.log_every == 0:
                    rate = saved / max(time.time() - t0, 1)
                    print(f"[fetch] {saved}/{len(tasks)} ({rate:.1f} img/s, "
                          f"{bytes_total/1e6:.0f} MB), missing so far {missing}",
                          flush=True)
    print(f"[fetch] done: {saved} saved ({bytes_total/1e6:.0f} MB), "
          f"{missing} not found, {failed} failed")


if __name__ == "__main__":
    main()
