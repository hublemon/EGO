"""cesft_v2 병렬 오케스트레이터 — DAG 스케줄러 (로직 동일, 최대 병렬).

직렬 cesft_v2_chain.sh 를 대체. 같은 학습/평가 명령을 의존성 순서로 돌리되,
GPU 메모리 여유가 있을 때 동시에 여러 잡을 띄운다.

OOM 안전:
  - MAX_PARALLEL 하드캡 (기본 1 — 비디오 arm 동시 실행 시 호스트 RAM 폭증 방지)
  - 새 GPU 잡은 free memory > MIN_FREE_MB (기본 60GB) 일 때만 기동
  - 새 잡은 cgroup 여유 RAM > RAM_FLOOR_GB (기본 100GB) 일 때만 기동
    (admission control은 기동 시점만 보호 — 실행 중 arm의 스파이크는 MAX_PARALLEL=1로 격리)
  - 트레이너 자체 per-sample OOM-skip 이 2차 안전망 (일시 peak 시 샘플만 건너뜀)

멱등/무인:
  - 완료는 marker 파일로 판정 (재시작 시 done skip)
  - 진행중 잡은 pids/<id>.pid 로 추적 → 오케스트레이터 재시작해도 재접속(중복기동 방지)
  - 자식은 setsid 분리 (오케스트레이터 죽어도 학습 지속)
  - 각 스테이지 후 artifact.html 재굽기

사용: RETRO3_RUNS=runs/cesft_v2 python tools/parallel_orchestrator.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
PY = os.environ.get("PYTHON_BIN", "/mnt/nvme/migration/jihun/envs/miniforge3/envs/eve-cu124/bin/python")
RUNS = Path(os.environ.get("RETRO3_RUNS", "runs/cesft_v2"))
CFG = "configs/step2_retrospection/cesft_v2.yaml"
ADAPT = "outputs/step2_retrospection/cesft_v2"
MK = RUNS / "markers"
LOG = RUNS / "logs" / "chain.log"
PIDS = RUNS / "pids"
for d in (MK, PIDS, RUNS / "logs", RUNS / "eval"):
    d.mkdir(parents=True, exist_ok=True)

MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "1"))
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "60000"))
# 호스트 RAM admission control: cgroup 여유(limit-current)가 이 값 미만이면 새 잡 보류.
# 2026-07-24 cesft_v2 OOM-kill(240GiB cgroup, theta_ce+cand_free 동시 208G→SIGTERM) 재발 방지.
RAM_FLOOR_GB = int(os.environ.get("RAM_FLOOR_GB", "100"))
EVAL_N = os.environ.get("EVAL_N", "1000")
IV_N = os.environ.get("IV_N", "800")
CE_EPOCHS = os.environ.get("CE_EPOCHS", "1")
SFT_EPOCHS = os.environ.get("SFT_EPOCHS", "1")
TAU = os.environ.get("TAU", "1.0")
CSTACK_STEPS = os.environ.get("CSTACK_STEPS", "150")
BEST_R = os.environ.get("BEST_R", "sft_r15")
RUN_BASELINE_EXTRA = os.environ.get("RUN_BASELINE_EXTRA", "1") == "1"
RUN_APPENDIX_A = os.environ.get("RUN_APPENDIX_A", "1") == "1"

ENV = {**os.environ, "PYTHONPATH": "src", "HF_HOME": "/mnt/nvme/cache",
       "RETRO3_RUNS": str(RUNS), "TOKENIZERS_PARALLELISM": "false",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "RETRO_NEXT_GAP_TEXT": "after the current action ends",
       "LD_LIBRARY_PATH": "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}


def log(msg: str):
    line = f"[orch {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def done(marker: str) -> bool:
    return (MK / marker).is_file()


def gpu_free_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0


# cgroup (limit, memory.stat, current) — v2 우선, v1 폴백.
_CG_PATHS = (("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory.stat",
              "/sys/fs/cgroup/memory.current"),
             ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
              "/sys/fs/cgroup/memory/memory.stat",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
# 회수 불가(= 실제로 압박을 만드는) 항목. dirty/writeback 은 flush 전까지 못 비우므로 포함.
_HARD_KEYS_V2 = ("anon", "unevictable", "slab_unreclaimable", "file_dirty", "file_writeback")
_HARD_KEYS_V1 = ("total_rss", "total_unevictable", "total_dirty", "total_writeback")


def _read_mem_stat(path: str) -> dict:
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                k, _, v = ln.strip().partition(" ")
                try:
                    out[k] = int(v)
                except ValueError:
                    continue
    except Exception:
        pass
    return out


def _hard_used_bytes(stat: dict, current: int) -> int:
    """current(= page cache 포함) 대신 회수 불가 사용량만 합산."""
    if not stat:
        return current  # stat 을 못 읽으면 종전대로 보수적 동작
    keys = _HARD_KEYS_V2 if "anon" in stat else _HARD_KEYS_V1
    hard = sum(stat.get(k, 0) for k in keys)
    if hard <= 0:
        return current
    return min(hard, current) if current > 0 else hard


def cgroup_ram_free_gb() -> float:
    """이 프로세스가 묶인 cgroup의 여유 RAM(GB) = limit − '회수 불가' 사용량.
    psutil.virtual_memory()는 호스트 전체(2TB)를 봐서 240GiB cgroup 한도에 무력 —
    반드시 cgroup 값을 읽어야 게이트가 실제로 작동한다. v2 우선, 실패 시 v1,
    한도 미설정('max')/읽기 실패면 큰 값 반환(교착 방지 — GPU 게이트에 위임).

    limit − memory.current 는 쓰지 않는다: current 는 page cache(file) 를 포함해서
    frame_cache JPEG 쓰기만으로도 수백 G 가 쌓이고(2026-07-24 20:23 관측:
    anon 4.2G / file 227.7G / current 232.9G), 회수 가능한 캐시 때문에 admission
    게이트가 free=0G 로 굳어 잡이 영구 대기했다. page cache 는 제외하고 센다.
    (frame_extractor.py 와 동일 로직 — 한쪽만 고치면 두 게이트가 어긋난다.)"""
    for mx, st, cur in _CG_PATHS:
        try:
            lim = Path(mx).read_text().strip()
            if lim == "max":
                break
            limit = int(lim)
            if limit <= 0 or limit >= (1 << 62):  # 사실상 무제한
                break
            try:
                used = int(Path(cur).read_text().strip())
            except Exception:
                used = 0
            hard = _hard_used_bytes(_read_mem_stat(st), used)
            free = max(0.0, (limit - hard) / 1e9)
            # 호스트가 실제로 더 빡빡하면 그쪽이 진짜 상한.
            try:
                import psutil
                free = min(free, psutil.virtual_memory().available / 1e9)
            except Exception:
                pass
            return free
        except Exception:
            continue
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 1e9


def bake():
    try:
        subprocess.run([PY, "tools/cesft_v2_artifact.py", "--run", str(RUNS),
                        "--out", str(RUNS / "artifact.html"), "--now",
                        time.strftime("%Y-%m-%dT%H:%M:%S")],
                       env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except Exception:
        pass


def sel_ce(run_name, arm, extra=None):
    return [PY, "-m", "ego.step2_retrospection.train.select_ce", "--config", CFG,
            "--run_name", run_name, "--arm", arm, "--tau", TAU, "--epochs", CE_EPOCHS] + (extra or [])


def sft(run_name, rho):
    return [PY, "-m", "ego.step2_retrospection.train.sft_v2", "--config", CFG,
            "--run_name", run_name, "--init_adapter", f"{ADAPT}/theta_ce/adapter",
            "--ce_replay_rho", rho, "--ce_tau", TAU, "--epochs", SFT_EPOCHS]


def battery(arm):
    return [PY, "-m", "ego.step2_retrospection.eval.battery", "--config", CFG,
            "--arm", arm, "--adapter", f"{ADAPT}/{arm}/adapter", "--eval_n", EVAL_N]


def harden(arm):
    return [PY, "-m", "ego.step2_retrospection.eval.harden_s3", "--config", CFG,
            "--arm", arm, "--adapter", f"{ADAPT}/{arm}/adapter", "--n", IV_N]


# ── GPU 태스크 DAG: (id, marker, deps[markers], argv) ──
GPU_TASKS = []


def add(tid, marker, deps, cmd):
    GPU_TASKS.append({"id": tid, "marker": marker, "deps": deps, "cmd": cmd})


# ── core=sft_r15 우선 재배치 (2026-07-24) ──
# MAX_PARALLEL=1 에서 launch 는 이 리스트 순서대로 ready task 를 집으므로,
# 순서 자체가 우선순위. 결과값은 불변, "언제 알게 되는가"만 최적화(headline fail-fast).
#
# Phase A — core spine: θ_CE → G-ACC1 → sft_r15 → belief-only U_g / G-NH
add("theta_ce", "S_CE_THETA_CE_DONE", [], sel_ce("theta_ce", "wm_cand"))
add("eval_theta_ce", "S7_EVAL_THETA_CE_DONE", ["S_CE_THETA_CE_DONE"], battery("theta_ce"))
add("sft_r15", "S6_SFT_R15_DONE", ["S_CE_THETA_CE_DONE"], sft("sft_r15", "0.15"))
add("eval_sft_r15", "S7_EVAL_SFT_R15_DONE", ["S6_SFT_R15_DONE"], battery("sft_r15"))
add("harden_sft_r15", "S3H_SFT_R15_DONE", ["S7_EVAL_SFT_R15_DONE"], harden("sft_r15"))
#
# Phase B — 귀속 ablation: 성립부등식(wm>cand_free) · no_history
#   baseline 은 cand_free/no_history 만 (random_cand·no_video 제거 — WM prior 후보가
#   데이터에 이미 shuffle 저장되어 위치/객관식 confound 가 wm_cand 안에서 통제됨).
add("cand_free", "S_CE_CAND_FREE_DONE", [], sel_ce("cand_free", "cand_free"))
add("eval_cand_free", "S7_EVAL_CAND_FREE_DONE", ["S_CE_CAND_FREE_DONE"], battery("cand_free"))
if RUN_BASELINE_EXTRA:
    add("no_history", "S_CE_NO_HISTORY_DONE", [], sel_ce("no_history", "no_history"))
    add("eval_no_history", "S7_EVAL_NO_HISTORY_DONE", ["S_CE_NO_HISTORY_DONE"], battery("no_history"))
#
# Phase C — ρ ablation/fallback: r0(replay 없음) · r30(G-NH fallback) + robustness
for r, rho in (("sft_r0", "0.0"), ("sft_r30", "0.30")):
    R = r.upper()
    add(r, f"S6_{R}_DONE", ["S_CE_THETA_CE_DONE"], sft(r, rho))
    add(f"eval_{r}", f"S7_EVAL_{R}_DONE", [f"S6_{R}_DONE"], battery(r))
    add(f"harden_{r}", f"S3H_{R}_DONE", [f"S7_EVAL_{R}_DONE"], harden(r))
# ④ WiSE-FT (θ_CE + BEST_R 의존) — merge 는 CPU inline, eval/harden 은 GPU
for a in ("025", "050", "075"):
    tag = f"wise_a{a}"
    add(f"eval_{tag}", f"S7_EVAL_{tag.upper()}_DONE",
        [f"MERGED_{tag.upper()}", "S_CE_THETA_CE_DONE", f"S6_{BEST_R.upper()}_DONE"], battery(tag))
    add(f"harden_{tag}", f"S3H_{tag.upper()}_DONE", [f"S7_EVAL_{tag.upper()}_DONE"], harden(tag))

# 조건부(P-UTIL) 부록A 태스크는 putil 통과 후 동적 추가
APPENDIX_ADDED = [False]


def maybe_add_appendix():
    if not RUN_APPENDIX_A or APPENDIX_ADDED[0]:
        return
    if not done(f"S3H_{BEST_R.upper()}_DONE"):
        return
    # P-UTIL 판정 (inline)
    import json
    h = RUNS / "eval" / f"{BEST_R}.harden_s3.json"
    putil = "MISSING"
    try:
        putil = "PASS" if json.loads(h.read_text())["utility_belief_only_ci"]["lo"] > 0 else "FAIL"
    except Exception:
        pass
    (MK / f"P_UTIL_{putil}").write_text(f'{{"best_r":"{BEST_R}"}}')
    log(f"P-UTIL({BEST_R}) = {putil}")
    APPENDIX_ADDED[0] = True
    if putil == "PASS":
        add("c_stack", "S_CE_C_STACK_DONE", [f"S6_{BEST_R.upper()}_DONE"],
            sel_ce("c_stack", "wm_cand", ["--init_adapter", f"{ADAPT}/{BEST_R}/adapter", "--max_steps", CSTACK_STEPS]))
        add("c_ctrl", "S_CE_C_CTRL_DONE", ["S_CE_THETA_CE_DONE"],
            sel_ce("c_ctrl", "wm_cand", ["--init_adapter", f"{ADAPT}/theta_ce/adapter", "--max_steps", CSTACK_STEPS]))
        add("eval_c_stack", "S7_EVAL_C_STACK_DONE", ["S_CE_C_STACK_DONE"], battery("c_stack"))
        add("eval_c_ctrl", "S7_EVAL_C_CTRL_DONE", ["S_CE_C_CTRL_DONE"], battery("c_ctrl"))
        add("harden_c_stack", "S3H_C_STACK_DONE", ["S7_EVAL_C_STACK_DONE"], harden("c_stack"))
    else:
        for m in ("S_CE_C_STACK_DONE", "S_CE_C_CTRL_DONE", "S7_EVAL_C_CTRL_DONE"):
            (MK / m).write_text(f'{{"skipped":"P-UTIL={putil}"}}')


# ── CPU inline 후처리 (deps 충족 시 1회 실행) ──
CPU_DONE = set()


def gate(args, out):
    subprocess.run([PY, "tools/paired_boot.py", "--run", str(RUNS)] + args + ["--out", str(RUNS / "eval" / out)],
                   env=ENV, stdout=open(LOG, "a"), stderr=subprocess.STDOUT)


def run_cpu_tasks():
    E = RUNS / "eval"
    # WiSE merges (θ_CE + BEST_R 준비되면)
    if done("S_CE_THETA_CE_DONE") and done(f"S6_{BEST_R.upper()}_DONE"):
        for a, av in (("025", "0.25"), ("050", "0.50"), ("075", "0.75")):
            tag = f"wise_a{a}"
            if not done(f"MERGED_{tag.upper()}") and not Path(f"{ADAPT}/{tag}/adapter").is_dir():
                subprocess.run([PY, "tools/merge_adapters.py", "--adapter_a", f"{ADAPT}/theta_ce/adapter",
                                "--adapter_b", f"{ADAPT}/{BEST_R}/adapter", "--alpha", av,
                                "--out", f"{ADAPT}/{tag}/adapter"], env=ENV,
                               stdout=open(LOG, "a"), stderr=subprocess.STDOUT)
            if Path(f"{ADAPT}/{tag}/adapter/adapter_model.safetensors").is_file():
                (MK / f"MERGED_{tag.upper()}").write_text("{}")
    # 게이트들 (deps 충족 & 1회)
    def once(key, cond, fn):
        if key not in CPU_DONE and cond:
            fn(); CPU_DONE.add(key)
    once("g_acc1", done("S7_EVAL_THETA_CE_DONE"),
         lambda: gate(["--arm_a", "theta_ce", "--gate", "G-ACC1"], "paired_G-ACC1_theta_ce.json"))
    once("g_delta_cf", done("S7_EVAL_THETA_CE_DONE") and done("S7_EVAL_CAND_FREE_DONE"),
         lambda: gate(["--arm_a", "theta_ce", "--arm_b", "cand_free", "--gate", "G-DELTA", "--metric", "SelAcc"],
                      "paired_G-DELTA_theta_ce_vs_cand_free.json"))
    for r in ("sft_r0", "sft_r15", "sft_r30"):
        once(f"gnh_{r}", done(f"S7_EVAL_{r.upper()}_DONE") and done("S7_EVAL_THETA_CE_DONE"),
             lambda r=r: gate(["--arm_a", r, "--arm_b", "theta_ce", "--gate", "G-NH"],
                              f"paired_G-NH_{r}_vs_theta_ce.json"))
    # WiSE frontier (모든 wise eval+harden & theta_ce eval 완료 시)
    wise_ok = all(done(f"S7_EVAL_WISE_A{a}_DONE") for a in ("025", "050", "075"))
    once("wise_frontier", wise_ok and done("S7_EVAL_THETA_CE_DONE") and not done("S_WISE_DONE"),
         lambda: _frontier())
    # 부록A T-ACC
    once("tacc_b0", done("S7_EVAL_C_STACK_DONE") and done(f"S7_EVAL_{BEST_R.upper()}_DONE"),
         lambda: gate(["--arm_a", "c_stack", "--arm_b", BEST_R, "--gate", "G-DELTA", "--metric", "SelAcc"],
                      "paired_TACC_cstack_vs_b0.json"))
    once("tacc_ctrl", done("S7_EVAL_C_STACK_DONE") and done("S7_EVAL_C_CTRL_DONE"),
         lambda: gate(["--arm_a", "c_stack", "--arm_b", "c_ctrl", "--gate", "G-DELTA", "--metric", "SelAcc"],
                      "paired_TACC_cstack_vs_cctrl.json"))


def _frontier():
    import json
    E = RUNS / "eval"
    pts = []
    def load(a):
        b = E / f"{a}.json"
        if not b.is_file():
            return None
        bj = json.loads(b.read_text())
        h = E / f"{a}.harden_s3.json"
        hj = json.loads(h.read_text()) if h.is_file() else {}
        cs = (hj.get("causal_sensitivity_ci", {}).get("both", {}) or {}).get("point")
        ug = (hj.get("utility_belief_only_ci", {}) or {}).get("point")
        return {"SelAcc": bj.get("acc"), "GADR": bj.get("G2_correction"),
                "causal_sensitivity": cs, "U_g": ug}
    for a, arm in [(0.0, "theta_ce"), (0.25, "wise_a025"), (0.5, "wise_a050"),
                   (0.75, "wise_a075"), (1.0, BEST_R)]:
        d = load(arm)
        if d:
            pts.append({"alpha": a, "arm": arm, **d})
    (E / "wise_ft_frontier.json").write_text(json.dumps(pts, indent=1, ensure_ascii=False))
    (MK / "S_WISE_DONE").write_text("{}")
    log(f"WiSE frontier: {len(pts)} pts")


# ── 스케줄러 ──
def pid_alive(pid: int) -> bool:
    """좀비(Z)는 죽은 것으로 취급 — os.kill(0)은 좀비에도 성공하므로 /proc 상태 확인."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().split(") ", 1)[1].split(" ", 1)[0]
        return state != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError, OSError):
        return False


def launch(task):
    f = open(LOG, "a")
    f.write(f"\n[orch] ==== LAUNCH {task['id']} ====\n")
    f.flush()
    p = subprocess.Popen(task["cmd"], env=ENV, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    (PIDS / f"{task['id']}.pid").write_text(str(p.pid))
    return {"id": task["id"], "pid": p.pid, "proc": p, "marker": task["marker"],
            "retries": task.get("retries", 0)}


def adopt_running():
    """오케스트레이터 재시작 시 이미 도는 잡 재접속 (pidfile)."""
    inflight = {}
    for task in GPU_TASKS:
        pf = PIDS / f"{task['id']}.pid"
        if done(task["marker"]) or not pf.is_file():
            continue
        try:
            pid = int(pf.read_text().strip())
        except Exception:
            continue
        if pid_alive(pid):
            inflight[task["id"]] = {"id": task["id"], "pid": pid, "proc": None,
                                    "marker": task["marker"], "retries": 0}
            log(f"재접속 in-flight: {task['id']} (pid {pid})")
    return inflight


def main():
    log(f"오케스트레이터 시작 (MAX_PARALLEL={MAX_PARALLEL}, MIN_FREE={MIN_FREE_MB}MB)")
    inflight = adopt_running()
    last_bake = 0
    stuck = 0
    while True:
        maybe_add_appendix()
        run_cpu_tasks()
        # reap
        for tid in list(inflight):
            info = inflight[tid]
            alive = pid_alive(info["pid"])
            if done(info["marker"]):
                log(f"완료: {tid}")
                (PIDS / f"{tid}.pid").unlink(missing_ok=True)
                del inflight[tid]
            elif not alive:  # 죽었는데 marker 없음 → 실패
                info["retries"] += 1
                (PIDS / f"{tid}.pid").unlink(missing_ok=True)
                del inflight[tid]
                if info["retries"] <= 1:
                    log(f"실패 {tid} (marker 없음) — 재시도 예정")
                    for t in GPU_TASKS:
                        if t["id"] == tid:
                            t["retries"] = info["retries"]
                else:
                    log(f"실패 {tid} 반복 — 건너뜀 (CHAIN_STUCK 후보)")
                    (MK / f"FAILED_{tid.upper()}").write_text("{}")
        # launch ready
        ready = [t for t in GPU_TASKS
                 if not done(t["marker"]) and t["id"] not in inflight
                 and not (MK / f"FAILED_{t['id'].upper()}").is_file()
                 and all(done(d) for d in t["deps"])]
        for t in ready:
            if len(inflight) >= MAX_PARALLEL:
                break
            free = gpu_free_mb()
            if free < MIN_FREE_MB and inflight:
                break  # GPU 메모리 부족 — 기존 잡 끝날 때까지 대기
            ram_free = cgroup_ram_free_gb()
            if ram_free < RAM_FLOOR_GB and inflight:
                log(f"RAM 대기: cgroup free {ram_free:.0f}G < floor {RAM_FLOOR_GB}G — 기존 잡 대기")
                break  # 호스트 RAM(cgroup) 부족 — OOM-kill(SIGTERM) 방지
            inflight[t["id"]] = launch(t)
            log(f"기동: {t['id']} (동시 {len(inflight)}, gpu_free {free}MB, ram_free {ram_free:.0f}G)")
            time.sleep(20)  # 모델 로드 겹침 완화 (동시 로드 메모리 스파이크 방지)
        # bake
        if time.time() - last_bake > 90:
            bake(); last_bake = time.time()
        # 완료 판정
        pending = [t for t in GPU_TASKS if not done(t["marker"])
                   and not (MK / f"FAILED_{t['id'].upper()}").is_file()]
        if not pending and not inflight and APPENDIX_ADDED[0]:
            run_cpu_tasks()
            bake()
            (MK / "CESFT_V2_CHAIN_DONE").write_text('{"ts":%d}' % int(time.time()))
            (MK / "RETRO3_CHAIN_DONE").write_text('{"ts":%d}' % int(time.time()))
            log("전체 완료")
            return
        if not inflight and not ready:
            stuck += 1
            if stuck > 20:
                log("교착 — runnable 없음, in-flight 없음")
                (MK / "CHAIN_STUCK").write_text('{"reason":"deadlock"}')
                return
        else:
            stuck = 0
        time.sleep(15)


if __name__ == "__main__":
    main()
