"""Download a specific set of Ego4D clips directly via boto3, bypassing the
official ``ego4d`` CLI's broken ``--video_uid_file``/``--video_uids`` filter
for non-``full_scale``/``clips`` datasets (e.g. ``clip_256ss``).

Why this exists: ``ego4d`` CLI 1.7.3's ``DATASETS_VIDEO`` list
(``ego4d/cli/config.py``) only contains ``["full_scale", "clips",
"components/videos", "video_540ss"]``. For any other dataset name --
``clip_256ss`` in particular, which this repo uses for its much smaller
download footprint -- ``list_videos_for_download`` in ``ego4d/cli/download.py``
silently ignores the uid filter and returns the *entire* unfiltered catalog
(14,026 clips as of Ego4D v2) instead of the requested subset. See
``develop_report/2026-07-16_ego4d-data-download-handoff.md`` for the full
story of how this was discovered.

Prerequisite: the ``clip_256ss`` manifest.csv must already be present (it is
fetched automatically by any ``ego4d --datasets clip_256ss ...`` invocation,
even a filtered one that will pull too much -- run it once with e.g. a bogus
``--video_uids nonexistent`` filter to fetch just the manifest cheaply, or
let this script's first real run download it via the CLI yourself).

Usage:
    python scripts/step1/ego4d_lta/download_ego4d_clips.py \
        --manifest data/Ego4D/v2/clip_256ss/manifest.csv \
        --out-dir data/Ego4D/v2/clip_256ss \
        --uid-list outputs/ego4d_lta/index_full/full_clip_uids.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config

# The Ego4D-issued IAM user lacks s3:GetBucketLocation, so per-bucket region
# auto-detection isn't possible -- but every known Ego4D bucket lives in
# us-west-2 (confirmed empirically; the official CLI hardcodes the same).
EGO4D_S3_REGION = "us-west-2"


def load_manifest(manifest_path: Path) -> dict[str, str]:
    mapping = {}
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            mapping[row["video_uid"]] = row["s3_path"]
    return mapping


def _bucket_and_key(s3_path: str) -> tuple[str, str]:
    match = re.match(r"s3://([^/]+)/(.+)", s3_path)
    return match.groups()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to the dataset's manifest.csv (video_uid,type,s3_path)")
    parser.add_argument("--out-dir", required=True, help="Directory to save <uid>.mp4 files into")
    parser.add_argument("--uid-list", required=True, help="Newline-delimited file of uids to download")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.manifest))
    with open(args.uid_list) as f:
        needed = [line.strip() for line in f if line.strip()]

    todo = []
    missing_from_manifest = []
    for uid in needed:
        s3_path = manifest.get(uid)
        if s3_path is None:
            missing_from_manifest.append(uid)
            continue
        dest = out_dir / f"{uid}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        todo.append((uid, s3_path))

    print(
        f"total needed={len(needed)} already_have={len(needed) - len(todo) - len(missing_from_manifest)} "
        f"to_download={len(todo)} not_in_manifest={len(missing_from_manifest)}",
        flush=True,
    )
    if missing_from_manifest:
        print(f"WARNING: not present in manifest (can't be downloaded from this dataset): {missing_from_manifest}", flush=True)

    client = boto3.client("s3", config=Config(region_name=EGO4D_S3_REGION, signature_version="s3v4"))

    def download_one(item: tuple[str, str]) -> tuple[str, bool, str | None]:
        uid, s3_path = item
        bucket, key = _bucket_and_key(s3_path)
        dest = out_dir / f"{uid}.mp4"
        tmp = out_dir / f"{uid}.mp4.part"
        try:
            client.download_file(bucket, key, str(tmp))
            tmp.rename(dest)
            return (uid, True, None)
        except Exception as e:  # noqa: BLE001 - report and continue, don't kill the whole batch
            return (uid, False, str(e))

    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, item) for item in todo]
        for fut in as_completed(futures):
            uid, ok, err = fut.result()
            done += 1
            if not ok:
                failed.append((uid, err))
                print(f"FAIL {uid}: {err}", flush=True)
            if done % 25 == 0 or done == len(todo):
                print(f"progress {done}/{len(todo)} (failed={len(failed)})", flush=True)

    print(f"DONE. downloaded_ok={len(todo) - len(failed)} failed={len(failed)}", flush=True)
    if failed:
        print("failed uids:", [uid for uid, _ in failed], flush=True)


if __name__ == "__main__":
    main()
