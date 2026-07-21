#!/usr/bin/env bash
# 무인 9시간 체인 (GPU 0 전용) — 2026-07-20 야간
#
# 오늘 확인된 두 개의 설계 결함을 고치고 그 위에서 다시 측정한다.
#   결함 1 (치명적) ScenarioStratifiedSampler
#       73개 시나리오를 라운드로빈하며 각 416회씩 뽑는다. 시나리오 크기가 3~4,494개라
#       샘플별 epoch당 반복이 0.09배 ~ 138.7배 (1,498배 격차). "1 epoch"이 데이터 1회
#       통과가 아니고, 학습 노출은 시나리오 균등인데 val은 자연분포다(COOKING_GENERAL
#       val 11.9% vs 학습 1.4%). 증거: sampler=random 이 epoch 2에 top5 22.20 으로
#       scenario_stratified 전 런의 15 epoch 최고치(19.45)를 이미 넘겼다.
#       -> 이 체인의 모든 런은 sampler: random 을 쓴다.
#   결함 2 fps/관측창 불일치
#       인덱스 관측창이 3.496s인데 32프레임을 균등 추출하므로 실제 9.15fps다.
#       백본에는 8fps로 전달되어 anticipation_steps = 1.0*8/2 = 4스텝 = 8프레임 =
#       0.874초. 즉 tau=1.0s 를 요청하고 0.87s 를 예측한다(13% 오차).
#       -> Stage B에서 l_obs=4.0 으로 인덱스를 다시 만든다(32/4.0 = 정확히 8fps).
#          .pt 에 예측 시점이 구워지므로 피처 재추출이 필요하다.
#
# ViT-g/384 는 의도적으로 제외했다. encoder 1,012M(ViT-L 304M의 3.3배) + 토큰 2.25배라
# 1 GPU 추출만 7시간 이상으로 추정되어 9시간 안에 학습까지 넣을 수 없다. 별도 세션에서
# 범위를 좁혀(val + train 일부) 판정할 것.
#
# 각 스테이지는 독립적으로 실패해도 다음으로 넘어간다. 앞 스테이지 결과만으로도
# 읽을 수 있게 설계했다.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
REPO="$PWD"
GPU=0
OUT="outputs/goalstep/overnight"
LOG="$OUT/chain.log"
mkdir -p "$OUT"

INDEX_OLD="src/ego/step1_action_anticipation/goalstep/index"
INDEX_NEW="src/ego/step1_action_anticipation/goalstep/index_lobs4"
CACHE_NEW="../datasets/Ego4D/goalstep_feature_cache_lobs4"
ANNOT="../datasets/Ego4D/v2/annotations"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# train_goalstep_z1.py 한 번 실행. 실패해도 체인은 계속된다.
train() {
  local name="$1" cfg="configs/step1/goalstep/overnight/$name.yaml"
  local dir="$OUT/$name"
  [ -f "$dir/final_metrics.json" ] && { say "SKIP $name (이미 완료)"; return 0; }
  mkdir -p "$dir/logs"
  say "START train $name"
  CUDA_VISIBLE_DEVICES=$GPU python src/ego/step1_action_anticipation/goalstep/train_goalstep_z1.py \
    --config "$cfg" > "$dir/logs/train.log" 2>&1
  say "END   train $name rc=$?"
}

say "===== 무인 체인 시작 (GPU $GPU) ====="

# ---------------------------------------------------------------- Stage A
# 기존 캐시 위에서 sampler=random 으로 재측정. 결함 1만 고친 상태.
# 재추출을 기다리지 않고 바로 읽을 수 있는 결과를 먼저 확보한다.
say "--- Stage A: sampler=random 재측정 (기존 캐시) ---"
python - <<'PY' 2>&1 | tee -a "$LOG"
import yaml, copy, pathlib
base = yaml.safe_load(open('configs/step1/goalstep/z1_jihun2.yaml'))
pathlib.Path('configs/step1/goalstep/overnight').mkdir(parents=True, exist_ok=True)
# (이름, depth, weight_decay, final_wd, heads)
V = [
    ('a1_d1_vna',  1, 0.0001, 0.0001, ['verb','noun','action']),
    ('a2_d4_act',  4, 0.0001, 0.0001, ['action']),
    ('a3_d1_wd',   1, 0.04,   0.4,    ['action']),
]
for name, depth, wd, fwd, heads in V:
    c = copy.deepcopy(base)
    c['experiment']['name'] = 'goalstep_overnight_' + name
    c['experiment']['output_dir'] = 'outputs/goalstep/overnight/' + name
    c['model']['classifier']['num_probe_blocks'] = depth
    c['training']['sampler'] = 'random'      # 결함 1 수정
    c['training']['weight_decay'] = wd
    c['training']['final_weight_decay'] = fwd
    c['training']['train_heads'] = heads
    yaml.safe_dump(c, open(f'configs/step1/goalstep/overnight/{name}.yaml','w'),
                   sort_keys=False, allow_unicode=True)
    print(f'  wrote {name}: depth={depth} wd={wd}->{fwd} heads={heads} sampler=random')
PY

train a1_d1_vna     # verb+noun+action 이 action 단독보다 나은가 (s6_d1_rand 가 대조군)
train a2_d4_act     # 샘플러를 고친 뒤에도 depth 가 무의미한가
train a3_d1_wd      # 샘플러를 고친 뒤 weight decay 가 의미를 갖는가

# ---------------------------------------------------------------- Stage B
# 결함 2 수정: l_obs=4.0 인덱스 재생성 + 피처 재추출.
say "--- Stage B: l_obs=4.0 인덱스 재생성 + 재추출 (결함 2 수정) ---"
if [ -f "$INDEX_NEW/train.parquet" ]; then
  say "SKIP 인덱스 재생성 (이미 존재)"
else
  python src/ego/step1_action_anticipation/goalstep/build_goalstep_z1_index.py \
    --annotations-dir "$ANNOT" --tau-a 1.0 --l-obs 4.0 \
    --output-dir "$INDEX_NEW" > "$OUT/build_index.log" 2>&1
  say "인덱스 재생성 rc=$? -> $INDEX_NEW"
fi

if [ -f "$INDEX_NEW/train.parquet" ]; then
  python - <<PY 2>&1 | tee -a "$LOG"
import yaml, copy
c = yaml.safe_load(open('configs/step1/goalstep/z1_jihun2.yaml'))
c['dataset']['index_dir'] = '$INDEX_NEW'
c['dataset']['feature_cache_dir'] = '$CACHE_NEW'
c['dataset']['l_obs'] = 4.0          # 32 frames / 4.0s = 정확히 8 fps
c['training']['sampler'] = 'random'
c['model']['classifier']['num_probe_blocks'] = 1
c['experiment']['name'] = 'goalstep_overnight_fixed'
c['experiment']['output_dir'] = 'outputs/goalstep/overnight/c1_fixed_act'
yaml.safe_dump(c, open('configs/step1/goalstep/overnight/_extract.yaml','w'),
               sort_keys=False, allow_unicode=True)
print('  extract config 작성 (l_obs=4.0, 8fps 정합)')
PY
  for split in val train; do
    say "START extract $split"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/step1/ego4d_lta/extract_features.py \
      --config configs/step1/goalstep/overnight/_extract.yaml --split $split \
      > "$OUT/extract_$split.log" 2>&1
    say "END   extract $split rc=$?"
  done
  # 라벨 정합성은 학습 전에 반드시 확인한다 (taxonomy/인덱스가 바뀌었으므로).
  python scripts/step1/goalstep/verify_cache_labels.py \
    --config configs/step1/goalstep/overnight/_extract.yaml --samples 3000 \
    > "$OUT/verify_labels.log" 2>&1
  say "라벨 검증 rc=$? (상세: $OUT/verify_labels.log)"
else
  say "SKIP Stage B 재추출 — 인덱스 재생성 실패. $OUT/build_index.log 확인"
fi

# ---------------------------------------------------------------- Stage C
# 두 결함을 모두 고친 캐시 위에서 최종 측정.
say "--- Stage C: 수정된 캐시로 최종 측정 ---"
if [ -f "$CACHE_NEW/train/$(ls $CACHE_NEW/train 2>/dev/null | head -1)" ] 2>/dev/null; then
  python - <<PY 2>&1 | tee -a "$LOG"
import yaml, copy
b = yaml.safe_load(open('configs/step1/goalstep/overnight/_extract.yaml'))
for name, heads in [('c1_fixed_act', ['action']), ('c2_fixed_vna', ['verb','noun','action'])]:
    c = copy.deepcopy(b)
    c['experiment']['name'] = 'goalstep_overnight_' + name
    c['experiment']['output_dir'] = 'outputs/goalstep/overnight/' + name
    c['training']['train_heads'] = heads
    yaml.safe_dump(c, open(f'configs/step1/goalstep/overnight/{name}.yaml','w'),
                   sort_keys=False, allow_unicode=True)
    print(f'  wrote {name}: heads={heads}')
PY
  train c1_fixed_act
  train c2_fixed_vna
else
  say "SKIP Stage C — 수정된 캐시가 없다"
fi

# ---------------------------------------------------------------- 요약
say "--- 일반화 격차 측정 ---"
RUNS=$(ls -d $OUT/*/ 2>/dev/null | while read d; do [ -f "$d/best.pt" ] && echo "$d"; done)
if [ -n "$RUNS" ]; then
  CUDA_VISIBLE_DEVICES=$GPU python scripts/step1/goalstep/measure_gap.py $RUNS \
    > "$OUT/gap.txt" 2>&1
  say "격차 측정 완료 -> $OUT/gap.txt"
fi
say "===== 무인 체인 종료 ====="
