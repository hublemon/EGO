"""Download the full videos backing the GoalStep Z=1 index, via boto3.

GoalStep timestamps are **video-relative** (there is no clip layer), so unlike
FHO-LTA -- which uses the pre-trimmed ``clip_256ss`` clips -- this pipeline
needs whole parent videos, resolved as ``<video_root>/<video_uid>.mp4``
(``dataset.video_source: full_scale``).

Manifest tiering (measured 2026-07-19, 701 index video_uids):
  * ``video_540ss`` (v2/v2_1) covers **529/701** at ~300 MB/video.
  * the remaining **172** GoalStep-only videos exist only in ``full_scale``
    (v2_1 manifest) at ~750 MB/video.
So the default is "540ss where available, full_scale as fallback" (~290 GB
total) rather than all-full_scale (~530 GB). Mixed source resolution is
harmless: the V-JEPA2 transform resizes/crops to ``dataset.resolution`` anyway.

Fetch the manifests first (cheap -- the ``ego4d`` CLI writes manifest.csv even
when the uid filter matches nothing):

    ego4d --datasets video_540ss --version v2_1 --video_uids 00000000-0000-0000-0000-000000000000 -o data/Ego4D -y
    ego4d --datasets full_scale  --version v2_1 --video_uids 00000000-0000-0000-0000-000000000000 -o data/Ego4D -y

Then:

    python scripts/step1/goalstep/download_goalstep_videos.py \
        --uid-list outputs/goalstep/index/video_uids.txt \
        --manifest data/Ego4D/v2/video_540ss/manifest.csv \
        --manifest data/Ego4D/v2/full_scale/manifest.csv \
        --out-dir data/Ego4D/v2/goalstep_videos

``--dry-run`` prints the resolved per-manifest split and the exact byte total
(via S3 ``head_object``) without transferring anything.

Credentials come from ``~/.aws/credentials`` (the Ego4D-issued IAM user); the
Ego4D buckets all live in us-west-2 and the IAM user lacks
``s3:GetBucketLocation``, so the region is hardcoded exactly as in
``scripts/step1/ego4d_lta/download_ego4d_clips.py``.
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config

EGO4D_S3_REGION = "us-west-2"
# manifest.csv column holding the object URI: video_540ss/clip_256ss manifests
# use `s3_path`; the full_scale manifest uses `path` (== canonical_s3_location).
S3_PATH_COLUMNS = ("s3_path", "path", "canonical_s3_location")


def load_manifest(manifest_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        col = next((c for c in S3_PATH_COLUMNS if c in (reader.fieldnames or [])), None)
        if col is None:
            raise ValueError(f"{manifest_path}: none of {S3_PATH_COLUMNS} present in header")
        for row in reader:
            if row.get(col):
                mapping[row["video_uid"]] = row[col]
    return mapping


def _bucket_and_key(s3_path: str) -> tuple[str, str]:
    match = re.match(r"s3://([^/]+)/(.+)", s3_path)
    if match is None:
        raise ValueError(f"Not an s3:// URI: {s3_path!r}")
    return match.groups()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True,
                        help="manifest.csv path; repeatable, earlier manifests win (preference order)")
    parser.add_argument("--uid-list", required=True, help="Newline-delimited video_uids (index/video_uids.txt)")
    parser.add_argument("--out-dir", required=True, help="Directory to save <video_uid>.mp4 into")
    parser.add_argument("--limit", type=int, default=None, help="Only take the first N uids (smoke tests)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Resolve + size everything, download nothing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = [line.strip() for line in Path(args.uid_list).read_text().splitlines() if line.strip()]
    if args.limit:
        needed = needed[: args.limit]

    # First manifest that has a uid wins, so callers control the size/quality tier.
    resolved: dict[str, tuple[str, str]] = {}
    for manifest_path in args.manifest:
        mapping = load_manifest(Path(manifest_path))
        hits = 0
        for uid in needed:
            if uid not in resolved and uid in mapping:
                resolved[uid] = (mapping[uid], manifest_path)
                hits += 1
        print(f"{manifest_path}: {len(mapping)} entries, resolved {hits} new uids", flush=True)

    missing = [uid for uid in needed if uid not in resolved]
    todo, have = [], 0
    for uid in needed:
        if uid not in resolved:
            continue
        dest = out_dir / f"{uid}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            have += 1
            continue
        todo.append((uid, resolved[uid][0]))

    print(f"total needed={len(needed)} already_have={have} to_download={len(todo)} "
          f"not_in_any_manifest={len(missing)}", flush=True)
    if missing:
        print(f"WARNING: no manifest entry for {len(missing)} uids: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}", flush=True)

    client = boto3.client("s3", config=Config(region_name=EGO4D_S3_REGION, signature_version="s3v4"))

    if args.dry_run:
        total = 0
        for uid, s3_path in todo:
            bucket, key = _bucket_and_key(s3_path)
            total += client.head_object(Bucket=bucket, Key=key)["ContentLength"]
        print(f"DRY RUN: {len(todo)} objects, {total / 1e9:.1f} GB to transfer", flush=True)
        return

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
            tmp.unlink(missing_ok=True)
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
            if done % 10 == 0 or done == len(todo):
                print(f"progress {done}/{len(todo)} (failed={len(failed)})", flush=True)

    print(f"DONE. downloaded_ok={len(todo) - len(failed)} failed={len(failed)}", flush=True)
    if failed:
        print("failed uids:", [uid for uid, _ in failed], flush=True)


if __name__ == "__main__":
    main()
