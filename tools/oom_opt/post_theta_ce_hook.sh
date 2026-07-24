#!/usr/bin/env bash
# post-theta_ce 무인 훅 (핸드오프 §5) — theta_ce 완료 마커를 폴링하다 뜨면 최적화 체인을 잇는다.
# 실행 중 orchestrator 와 '협조'만 한다: 스킵은 marker touch 로(orchestrator 가 done 판정→건너뜀),
# GPU 잡은 orchestrator 와 겹치지 않게만 띄운다(strip-eval 은 체인 완료 후 = GPU 여유).
#
# 절대 규칙 준수:
#   - theta_ce/supervisor/orchestrator/watchdog 를 절대 죽이거나 재시작하지 않는다.
#   - 프레임 추출(CPU+RAM)은 theta_ce 완료 후에만, RAM 60G 게이트로 self-throttle.
#   - strip-eval(GPU)은 CESFT_V2_CHAIN_DONE 이후에만 (동시 GPU 잡 금지 = OOM 방지).
# 멱등: 각 단계 marker 확인 후 실행. setsid 분리라 세션이 꺼져도 지속.
set -uo pipefail
REPO="${REPO:-/mnt/nvme/migration/jihun/EGO_jihun3}"
cd "$REPO"
PY="${PY:-/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python}"
RENDER_PY="/opt/conda/bin/python3"
RUN="$REPO/runs/cesft_v2"
MK="$RUN/markers"
LOG="$RUN/logs/chain.log"
HLOG="$RUN/logs/post_theta_hook.log"
export RETRO3_RUNS="runs/cesft_v2" PYTHONPATH="src" HF_HOME="/mnt/nvme/cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"
export RETRO_NEXT_GAP_TEXT="after the current action ends"

mkdir -p "$RUN/logs" "$MK"
hlog(){ echo "[post_theta_hook $(date '+%F %T')] $*" | tee -a "$HLOG" >> "$LOG"; }
have(){ [ -f "$MK/$1" ]; }
touch_marker(){ # touch_marker <NAME> <reason-json-payload>
  if ! have "$1"; then
    printf '{"ts":%d,"skipped_by":"post_theta_hook","reason":%s}\n' "$(date +%s)" "$2" > "$MK/$1"
    hlog "touch $1 — $2"
  fi
}
render(){ "$RENDER_PY" "$REPO/tools/oom_opt/render_progress.py" "$RUN" >/dev/null 2>&1 || true; }
wait_marker(){ # wait_marker <NAME>  (CHAIN_STUCK/DONE 시 사유 반환 1)
  while ! have "$1"; do
    have CHAIN_STUCK && { hlog "CHAIN_STUCK 감지 — '$1' 대기 중단"; return 1; }
    sleep 20
  done
  return 0
}

# 중복 기동 방지
LOCK="$MK/POST_THETA_HOOK_RUNNING"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  hlog "이미 실행 중(pid $(cat "$LOCK")) — 종료"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
hlog "시작 (pid $$) — S_CE_THETA_CE_DONE 폴링"

# ── Phase 0: theta_ce 완료 대기 (그 전엔 아무 무거운 작업도 안 함) ──
wait_marker S_CE_THETA_CE_DONE || exit 0
hlog "S_CE_THETA_CE_DONE 감지 — 최적화 체인 개시"

# ── Phase 1: marker-touch 스킵 (즉시·안전, orchestrator 가 cand_free 에 도달하기 전) ──
# B 스킵 (핸드오프 §2) — cand_free/no_history 는 EGO_jihun 확정결과+strip 으로 대체.
# eval_* 마커도 touch (adapter 없는 arm 을 orchestrator 가 배터리 돌리다 실패하지 않게).
touch_marker S_CE_CAND_FREE_DONE      '"EGO_jihun 성립부등식 확정결과로 대체(§2)"'
touch_marker S7_EVAL_CAND_FREE_DONE   '"cand_free 스킵 동반(§2)"'
touch_marker S_CE_NO_HISTORY_DONE     '"strip-eval(같은 θ_CE paired)로 대체(§2)"'
touch_marker S7_EVAL_NO_HISTORY_DONE  '"no_history 스킵 동반(§2)"'
# WiSE α∈{.25,.75} 스킵 → α=0.5 한 점만 (핸드오프 §2). merge/eval/harden 마커 모두 touch.
for a in A025 A075; do
  touch_marker "MERGED_WISE_${a}"     '"WiSE α=0.5 한 점만(§2)"'
  touch_marker "S7_EVAL_WISE_${a}_DONE" '"WiSE α=0.5 한 점만(§2)"'
  touch_marker "S3H_WISE_${a}_DONE"   '"WiSE α=0.5 한 점만(§2)"'
done
hlog "B 스킵 + WiSE 축소 마커 완료 (예상 절약 ~7.1h + 조건부 r30 2.5h)"
render

# ── vlm.py 가드 + 캐시 로더 분기 적용 (theta_ce 끝난 뒤라 안전; 이후 스테이지가 패치본 import) ──
"$PY" "$REPO/tools/oom_opt/apply_vlm_guards.py" 2>&1 | tee -a "$HLOG" >> "$LOG" || hlog "apply_vlm_guards 비정상(무시 가능 — 폴백 on-the-fly)"

# ── 프레임 사전추출 시작 (CPU-only, RAM 60G 게이트로 self-throttle; theta_ce 이미 종료) ──
EXLOCK="$MK/FRAME_EXTRACT_RUNNING"
if [ ! -f "$RUN/frame_cache/EXTRACT_DONE" ] && { [ ! -f "$EXLOCK" ] || ! kill -0 "$(cat "$EXLOCK" 2>/dev/null)" 2>/dev/null; }; then
  setsid "$PY" "$REPO/tools/oom_opt/frame_extractor.py" --pool both \
    </dev/null >"$RUN/logs/frame_extract_stdout.log" 2>&1 &
  echo $! > "$EXLOCK"
  hlog "프레임 추출 기동 (pid $!, RAM<60G 시 자동 일시정지)"
fi

# ── Phase 2: r15 G-NH 판정 후 조건부 r30 스킵 ──
if wait_marker S3H_SFT_R15_DONE; then
  # orchestrator CPU 게이트가 paired G-NH json 을 쓸 때까지 잠깐 대기
  GNH="$RUN/eval/paired_G-NH_sft_r15_vs_theta_ce.json"
  for _ in $(seq 1 60); do [ -f "$GNH" ] && break; have CHAIN_STUCK && break; sleep 20; done
  if [ -f "$GNH" ]; then
    VERDICT=$("$RENDER_PY" -c "import json,sys;print('PASS' if json.load(open(sys.argv[1])).get('pass') else 'FAIL')" "$GNH" 2>/dev/null || echo "UNKNOWN")
    hlog "G-NH(r15) = $VERDICT"
    if [ "$VERDICT" = "PASS" ]; then
      touch_marker S6_SFT_R30_DONE     '"r15 G-NH PASS — fallback r30 불필요(§3)"'
      touch_marker S7_EVAL_SFT_R30_DONE '"r15 G-NH PASS — r30 스킵 동반(§3)"'
      touch_marker S3H_SFT_R30_DONE    '"r15 G-NH PASS — r30 스킵 동반(§3)"'
      hlog "r30 스킵 (추가 절약 ~2.5h)"
    else
      hlog "G-NH FAIL/미결 — r30 는 orchestrator 가 정상 수행 (스킵 안 함)"
    fi
  else
    hlog "G-NH json 미생성 — r30 스킵 보류(안전측: orchestrator 가 r30 수행)"
  fi
  render
fi

# ── Phase 3: 전체 체인 완료 후 strip-eval (GPU 여유 확보 상태) ──
if wait_marker CESFT_V2_CHAIN_DONE; then
  hlog "CESFT_V2_CHAIN_DONE — strip-eval 시작 (GPU 여유)"
  if ! have S_STRIP_THETA_CE_DONE; then
    if [ -d "$REPO/outputs/step2_retrospection/cesft_v2/theta_ce/adapter" ]; then
      "$PY" "$REPO/tools/oom_opt/strip_eval.py" 2>&1 | tee -a "$HLOG" >> "$LOG" || hlog "strip_eval 실패 — 로그 확인"
    else
      hlog "θ_CE adapter 없음 — strip-eval 생략"
    fi
  fi
  render
  # 절약 합계 보고
  "$RENDER_PY" - "$RUN" <<'PYEOF' 2>&1 | tee -a "$HLOG" >> "$LOG" || true
import json,sys
d=json.load(open(sys.argv[1]+"/optim_progress.json"))
sec=d["saved_sec_applied"]; h=sec//3600; m=(sec%3600)//60
print(f"[요약] 적용 절약 {h}h{m:02d}m · gates={d.get('gates')}")
PYEOF
  hlog "완료 — 최적화 체인 종료"
fi
