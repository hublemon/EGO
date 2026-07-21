#!/usr/bin/env bash
# 무인 9시간 ViT-L 실험 (GPU 0 전용) — 2026-07-20 야간
#
# 왜 ViT-g 가 아니라 ViT-L 인가:
#   샘플러 결함(ScenarioStratifiedSampler, 샘플별 epoch 반복 0.09~138.7배)을 오늘 저녁에야
#   발견했다. 그 이전에 내린 진단 — "프로브 용량 무관", "weight decay 무효", "영상 다양성
#   무관", "데이터를 줄이면 나빠진다" — 은 전부 망가진 측정 위에서 나온 것이라 무효다.
#   ViT-g 는 9시간으로 train 19,000건까지만 뽑히므로 ViT-L(30,374건)과 비교하면 백본
#   차이와 데이터량 차이가 섞인다. 큰 모델일수록 데이터를 더 요구하니 ViT-g 에 불리하게
#   나올 위험이 크고, 그 잘못된 결론으로 11.6시간 투자를 접게 된다.
#
#   ViT-g 투자 여부를 결정하는 것은 Phase 2 의 **데이터 스케일링 곡선**이다:
#     30,000건에서 아직 가파르면 -> 병목은 데이터, ViT-g 는 나중
#     30,000건에서 포화됐으면   -> 병목은 피처 품질, ViT-g 가 최우선
#
# 고쳐진 베이스라인(s6_d1_rand, sampler=random)이 알려준 것:
#     top5  ep3 24.80 (정점) -> ep12 19.55 (하락),  train_loss 1.18 -> 0.0115
#   즉 random 으로 고쳐도 과적합은 여전하고, 정점이 epoch 3 으로 앞당겨졌다.
#   이제서야 정규화 실험이 의미를 갖는다(Phase 3).
#
# 공통 고정: sampler=random, seed 42, val_subset 2000/seed42, bf16, lr 3e-4, 동일 캐시.
# 각 스테이지는 독립 실패해도 다음으로 넘어간다. 완료된 런은 재실행 시 건너뛴다.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
GPU=0
OUT="outputs/goalstep/night"; mkdir -p "$OUT"
LOG="$OUT/chain.log"
CFG="configs/step1/goalstep/night"; mkdir -p "$CFG"
INDEX_NEW="src/ego/step1_action_anticipation/goalstep/index_lobs4"
CACHE_NEW="../datasets/Ego4D/goalstep_feature_cache_lobs4"
ANNOT="../datasets/Ego4D/v2/annotations"

say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

train(){
  local name="$1" dir="$OUT/$1"
  [ -f "$dir/final_metrics.json" ] && { say "SKIP $name (완료됨)"; return 0; }
  mkdir -p "$dir/logs"; say "START $name"
  CUDA_VISIBLE_DEVICES=$GPU python src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py \
    --config "$CFG/$name.yaml" > "$dir/logs/train.log" 2>&1
  say "END   $name rc=$?"
}

say "===== ViT-L 야간 체인 시작 (GPU $GPU) ====="

# s6_d1_rand 가 아직 돌고 있으면 GPU 를 비켜준다 (그것이 Phase 1 의 베이스라인이다).
while pgrep -f "sweep/s6_d1_rand.yaml" >/dev/null; do sleep 60; done
say "s6_d1_rand(베이스라인) 종료 확인"

# 설정 생성 ------------------------------------------------------------------
python - <<'PY' 2>&1 | tee -a "$LOG"
import yaml, copy
base = yaml.safe_load(open('configs/step1/goalstep/z1_jihun2.yaml'))
base['training']['sampler'] = 'random'          # 결함 1 수정 — 전 런 공통
CFG = 'configs/step1/goalstep/night/'
def w(name, **kw):
    c = copy.deepcopy(base)
    c['experiment']['name'] = 'goalstep_night_' + name
    c['experiment']['output_dir'] = 'outputs/goalstep/night/' + name
    c['model']['classifier']['num_probe_blocks'] = kw.get('depth', 1)
    c['training']['train_heads'] = kw.get('heads', ['action'])
    c['training']['epochs'] = kw.get('epochs', 15)
    c['training']['weight_decay'] = kw.get('wd', 0.0001)
    c['training']['final_weight_decay'] = kw.get('fwd', 0.0001)
    if 'n' in kw: c['training']['max_train_samples'] = kw['n']
    yaml.safe_dump(c, open(CFG + name + '.yaml','w'), sort_keys=False, allow_unicode=True)
    print(f'  {name}: depth={kw.get("depth",1)} heads={kw.get("heads",["action"])} '
          f'epochs={kw.get("epochs",15)} wd={kw.get("wd",0.0001)} n={kw.get("n","full")}')

# Phase 1 — 고쳐진 베이스라인 위에서 구조 재검증
w('b2_vna',  heads=['verb','noun','action'])   # verb top-5 상한 + 다중헤드 보조감독 효과
w('b3_d4',   depth=4)                          # "용량 무관" 결론 재검증
# Phase 2 — 데이터 스케일링 (ViT-g 투자 여부를 결정한다)
for n in (5000, 10000, 20000):
    w(f'n{n//1000:02d}k', n=n)
# Phase 3 — 과적합 대응 (정점이 epoch 3 이라 이제 의미가 있다)
w('r_wd',    wd=0.04, fwd=0.4)                 # weight decay 재검증
w('r_ep40',  epochs=40)                        # 정점 이후 곡선 전체 형태
PY

# Phase 1 ---------------------------------------------------------------------
say "--- Phase 1: 고쳐진 베이스라인 위 구조 재검증 ---"
train b2_vna
train b3_d4

# Phase 2 ---------------------------------------------------------------------
say "--- Phase 2: 데이터 스케일링 곡선 (ViT-g 판단 근거) ---"
train n05k
train n10k
train n20k

# Phase 3 ---------------------------------------------------------------------
say "--- Phase 3: 과적합 대응 재검증 ---"
train r_wd
train r_ep40

# Phase 4 ---------------------------------------------------------------------
# 결함 2(fps) 수정: 관측창 3.496s 에 32프레임 = 실제 9.15fps 인데 백본에는 8fps 로
# 전달되어 anticipation_steps = 4 -> 0.874s 를 예측하고 있다(tau=1.0 요청). l_obs=4.0
# 이면 32/4.0 = 정확히 8fps 가 되어 4스텝 = 1.0s 가 맞는다. .pt 에 구워지므로 재추출 필요.
say "--- Phase 4: fps 수정 (l_obs=4.0 인덱스 재생성 + 재추출) ---"
if [ ! -f "$INDEX_NEW/train.parquet" ]; then
  python src/ego/step1_action_anticipation/goalstep/build_goalstep_z1_index.py \
    --annotations-dir "$ANNOT" --tau-a 1.0 --l-obs 4.0 --output-dir "$INDEX_NEW" \
    > "$OUT/build_index.log" 2>&1
  say "인덱스 재생성 rc=$?"
fi
if [ -f "$INDEX_NEW/train.parquet" ]; then
  python - <<PY 2>&1 | tee -a "$LOG"
import yaml
c = yaml.safe_load(open('configs/step1/goalstep/night/b3_d4.yaml'))
c['model']['classifier']['num_probe_blocks'] = 1
c['dataset']['index_dir'] = '$INDEX_NEW'
c['dataset']['feature_cache_dir'] = '$CACHE_NEW'
c['dataset']['l_obs'] = 4.0
c['experiment']['name'] = 'goalstep_night_f_fps'
c['experiment']['output_dir'] = 'outputs/goalstep/night/f_fps'
yaml.safe_dump(c, open('configs/step1/goalstep/night/f_fps.yaml','w'),
               sort_keys=False, allow_unicode=True)
print('  f_fps 설정 작성 (l_obs=4.0 -> 정확히 8fps)')
PY
  for split in val train; do
    say "START extract $split (l_obs=4.0)"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/step1/ego4d_lta/extract_features.py \
      --config "$CFG/f_fps.yaml" --split $split > "$OUT/extract_$split.log" 2>&1
    say "END   extract $split rc=$?"
  done
  python scripts/step1/goalstep/verify_cache_labels.py --config "$CFG/f_fps.yaml" \
    --samples 3000 > "$OUT/verify_labels.log" 2>&1
  say "라벨 검증 rc=$? -> $OUT/verify_labels.log"
  train f_fps
else
  say "SKIP Phase 4 — 인덱스 재생성 실패 ($OUT/build_index.log)"
fi

# 마무리 ----------------------------------------------------------------------
say "--- train-val 격차 측정 ---"
RUNS=$(for d in $OUT/*/; do [ -f "$d/best.pt" ] && echo "$d"; done)
[ -n "$RUNS" ] && CUDA_VISIBLE_DEVICES=$GPU python scripts/step1/goalstep/measure_gap.py $RUNS \
  > "$OUT/gap.txt" 2>&1 && say "격차 -> $OUT/gap.txt"
say "===== ViT-L 야간 체인 종료 ====="
