"""VPA v2 — Frontier VLM(vision) 러너. 8프레임을 실제로 **본다**.

기존 `run_frontier_baseline.py` / `frontier_select_eval.py`는 텍스트만 보냈다(게이트웨이
text-only 가정). 여기서는 OpenAI 호환 멀티모달 포맷으로 이미지를 실어 보내고, 게이트웨이가
이미지를 받지 못하면 **즉시 중단하고 그 사실을 기록**한다(조용한 text-only 퇴화 방지).

비용 통제·재개 (사용자 지시):
  · `--subset` 으로 고정 500샘플만 호출 (seed 고정 → 재개 시 동일 집합)
  · `records.jsonl` 에 한 줄씩 append. 재실행 시 **성공한 sample_id 만** 건너뛴다.
    실패(429 등) 행은 남겨두되 재시도 대상으로 되살린다 — 과거 `resume_select_429.sh` 가
    api_error 행을 지워 "완료"로 오판정한 사고의 재발 방지(handoff §6-3).
  · `--max-calls` 로 호출 상한을 강제. 상한 도달 시 중단하고 재개 지점을 출력.

보고 규약: 커버리지 100% 미만이면 metrics 의 `reportable=false` 로 남고, 부분 결과를
논문 표에 올리지 않는다(handoff §2-7 의 972건 실패 사고 교훈).

사용:
  export FRONTIER_API_KEY=...   # 하드코딩 금지
  export FRONTIER_BASE_URL=https://gw.letsur.ai/v1
  export FRONTIER_MODEL=gemini-2.5-pro
  PYTHONPATH=src python -m ego.step3_results.vpa.v2.run_frontier \
      --gt runs/vpa_v2/vpa_v2_T3.json --subset runs/vpa_v2/frontier_subset_T3.json \
      --out runs/vpa_v2/preds/frontier_T3
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

from ego.step3_results.vpa.v2 import common as C
from ego.step3_results.vpa.v2 import frames as F


def encode_frames(paths: list[Path]) -> list[str]:
    out = []
    for p in paths:
        out.append("data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode())
    return out


def build_payload(model: str, system: str, user: str, data_uris: list[str]) -> dict:
    content: list[dict] = [{"type": "image_url", "image_url": {"url": u}} for u in data_uris]
    content.append({"type": "text", "text": user})
    return {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}]}


def call_api(base_url: str, api_key: str, payload: dict, max_retries: int, timeout: int):
    """(content, error, status) — 재시도는 429/5xx 에만. 4xx(요청 자체 오류)는 즉시 반환."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last, status = None, None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            status = r.status_code
            if status == 429 or status >= 500:
                last = f"HTTP {status}"
                time.sleep(min(2 ** attempt, 16))
                continue
            if status >= 400:  # 이미지 미지원 등 — 재시도해도 소용없다
                return None, f"HTTP {status}: {r.text[:200]}", status
            return r.json()["choices"][0]["message"]["content"], None, status
        except Exception as e:  # noqa: BLE001
            last = str(e)[:200]
            time.sleep(min(2 ** attempt, 16))
    return None, last, status


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--vocab", default="runs/vpa_v2/vocab.json")
    p.add_argument("--subset", default=None)
    p.add_argument("--out", required=True, help="출력 prefix (…​.records.jsonl / ….json 생성)")
    p.add_argument("--frames-dir", default="runs/vpa_v2")
    p.add_argument("--base-url", default=os.environ.get("FRONTIER_BASE_URL", "https://gw.letsur.ai/v1"))
    p.add_argument("--model", default=os.environ.get("FRONTIER_MODEL", "gemini-2.5-pro"))
    p.add_argument("--max-calls", type=int, default=500, help="이번 실행의 API 호출 상한")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--sleep", type=float, default=0.5, help="호출 간 간격(rate limit 완화)")
    p.add_argument("--probe", action="store_true", help="1샘플만 호출해 vision 지원 여부만 확인")
    args = p.parse_args()

    api_key = os.environ.get("FRONTIER_API_KEY")
    if not api_key:
        sys.exit("ERROR: FRONTIER_API_KEY 미설정 — 환경변수로만 주입한다(하드코딩 금지).")

    samples = C.load_json(args.gt)
    T = samples[0]["horizon"]
    vocab = C.load_json(args.vocab)["labels"]
    if args.subset:
        keep = set(C.load_json(args.subset)["sample_ids"])
        samples = [s for s in samples if s["sample_id"] in keep]

    cache_root = Path(args.frames_dir) / F.cache_dirname()
    out_prefix = Path(args.out)
    rec_path = out_prefix.with_suffix(".records.jsonl")
    rec_path.parent.mkdir(parents=True, exist_ok=True)

    done_ok = {r["sample_id"] for r in C.read_jsonl(rec_path) if r.get("ok")}
    todo = []
    for s in samples:
        if s["sample_id"] in done_ok:
            continue
        paths = F.frame_paths(cache_root, s)
        if all(p.is_file() for p in paths):
            todo.append(s)
    n_no_frames = len(samples) - len(done_ok) - len(todo)

    print(f"[info] model={args.model} endpoint={args.base_url}")
    print(f"[info] subset={len(samples)} · done={len(done_ok)} · todo={len(todo)} "
          f"· no_frames={n_no_frames} · max_calls={args.max_calls}")
    if args.probe:
        todo = todo[:1]
    if not todo:
        print("[info] 호출할 샘플 없음 (완료했거나 프레임 미추출).")

    n_call = n_ok = n_fail = 0
    with open(rec_path, "a") as fh:
        for s in todo:
            if n_call >= args.max_calls:
                print(f"[stop] 호출 상한 {args.max_calls} 도달 — 재실행하면 이어서 진행합니다.")
                break
            system, user = C.build_prompt(s, vocab, T, with_frames=True)
            uris = encode_frames(F.frame_paths(cache_root, s))
            payload = build_payload(args.model, system, user, uris)
            content, err, status = call_api(args.base_url, api_key, payload, args.max_retries, args.timeout)
            n_call += 1
            pred = C.parse_prediction(content, T) if content else []
            ok = bool(pred)
            n_ok += ok
            n_fail += (not ok)
            fh.write(json.dumps({"sample_id": s["sample_id"], "video_uid": s["video_uid"],
                                 "ok": ok, "pred": pred, "error": err, "status": status,
                                 "n_images": len(uris)}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{n_call}/{min(len(todo), args.max_calls)}] {s['sample_id']}: "
                  f"{pred if ok else 'FAIL ' + str(err)[:120]}")
            if err and status is not None and 400 <= status < 500 and status != 429:
                print("\n[ABORT] 게이트웨이가 요청을 거부했습니다(4xx). 이미지 파트 미지원 가능성이 높습니다.\n"
                      "        vision 지원 모델/엔드포인트 확인 후 재실행하세요. 부분 결과는 보고 금지.")
                break
            time.sleep(args.sleep)

    # preds json 재생성 — 성공 행만 모은다(실패를 오답으로 세지 않는다)
    preds = {r["sample_id"]: r["pred"] for r in C.read_jsonl(rec_path) if r.get("ok")}
    C.dump_json(out_prefix.with_suffix(".json"), preds)
    C.dump_json(out_prefix.parent / (out_prefix.stem + ".status.json"), {
        "model": args.model, "endpoint": args.base_url, "horizon": T,
        "subset_n": len(samples), "predicted": len(preds),
        "remaining": len(samples) - len(preds), "calls_this_run": n_call,
        "ok_this_run": n_ok, "fail_this_run": n_fail,
        "complete": len(preds) == len(samples),
        "note": "complete=false 이면 논문 표에 넣지 말 것 (부분 보고 금지)",
    })
    print(f"\nwrote {out_prefix.with_suffix('.json')}  ({len(preds)}/{len(samples)} 완료)")
    if len(preds) < len(samples):
        print(f"[resume] 남은 {len(samples) - len(preds)}건 — 동일 명령을 다시 실행하면 이어서 호출합니다.")


if __name__ == "__main__":
    main()
