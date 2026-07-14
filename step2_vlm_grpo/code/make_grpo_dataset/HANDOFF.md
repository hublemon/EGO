# GRPO 데이터셋 생성 파이프라인 — 세션 핸드오프

> 작성: 2026-05-28. 다음 세션이 그대로 이어받기 위한 자족적(self-contained) 문서.
> 명세 원본: [docs/GRPO_DATASET_SPEC.md](../docs/GRPO_DATASET_SPEC.md)

## 한 줄 요약
`make_grpo_dataset/` 의 스크립트 ①~⑥ 중 **①②③④ 완료, ⑤⑥ 작성 완료·미실행**.
다음 세션은 **⑤ assemble → ⑥ analyze 2줄만 실행**하면 데이터셋 완성 (①~④ 산출물 디스크에 존재).

## 환경
```bash
source ~/work/jihun/bootstrap/activate.sh      # conda env: eve-cu124
cd ~/work/jihun/EGO
# API 키는 절대 파일에 하드코딩 금지 (이 파이프라인은 API 불필요, V-JEPA2 로컬 추론만)
```

## 진행 상태 (스크립트별)

| # | 스크립트 | 상태 | 출력 | 비고 |
|---|---|---|---|---|
| ① | `select_train.py` | ✅ 완료 | `data/grpo_dataset/selected_train.jsonl` (**1,348줄**) | P01 12개 비디오. spec target 5,000 > 가용 1,348 → 전체 사용 |
| ② | `vjepa_infer_train.py` | ✅ 완료 | `data/grpo_dataset/predictions_train.jsonl` (**1,347줄**) | Top-5 verb/noun/action + softmax likelihood. 1건 에러(P01_09_880, trigger_frame 영상길이 초과 IndexError) → 드롭 |
| ③ | `extract_frame_train.py` | ✅ 완료 | `data/grpo_dataset/frames/{sample_id}.jpg` (**1,348장**) + `frames_manifest.jsonl` (1,348줄) | trigger_frame JPEG(short-side 768). video_id별 batch open으로 최적화. 에러 0 |
| ④ | `extract_memory_train.py` | ✅ 완료 | `data/grpo_dataset/memory_train.jsonl` (**1,348줄**) | task_history(평균 9.7, max 10) + temporal_proximity(0.5/1.0/2.0s, 평균 2.0 non-null) |
| ⑤ | `assemble_train.py` | 📝 작성·미실행 | `data/grpo_dataset/grpo_dataset.jsonl` | sample_id 조인. prediction/frame 없는 샘플 드롭. gt_in_top5 플래그 계산 |
| ⑥ | `analyze_train.py` | 📝 작성·미실행 | stdout + `data/grpo_dataset/stats/hit_rate.json` | GT in Top-5 hit rate, rank-1==GT, likelihood 분포 |

## 다음 세션 실행 순서 (복붙용)

```bash
source ~/work/jihun/bootstrap/activate.sh && cd ~/work/jihun/EGO

# 1) ⑤ 병합 (①~④ 산출물 전부 디스크에 있음)
python make_grpo_dataset/assemble_train.py        # → grpo_dataset.jsonl (~1,347)

# 2) ⑥ 통계 (이게 나오면 파이프라인 완료)
python make_grpo_dataset/analyze_train.py         # → stats/hit_rate.json
```
> ③ 산출 jpg가 손상/누락 의심되면 재생성: `python make_grpo_dataset/extract_frame_train.py --resume`

## 최종 산출물 (완료 시)
```
data/grpo_dataset/
├── selected_train.jsonl       # ① 1,348 ✅
├── predictions_train.jsonl    # ② 1,347 ✅
├── frames/{sample_id}.jpg     # ③ 1,348 ✅
├── frames_manifest.jsonl      # ③ 1,348 ✅
├── memory_train.jsonl         # ④ 1,348 ✅
├── grpo_dataset.jsonl         # ⑤ 메인 출력 (~1,347, no-pred 1건 드롭) — 미생성
└── stats/hit_rate.json        # ⑥ — 미생성
```

## grpo_dataset.jsonl 레코드 포맷 (⑤ 출력)
```json
{
  "sample_id": "P01_02_78",
  "split": "train",
  "video_id": "P01_02",
  "narration_id": "P01_02_78",
  "trigger_frame": 3540,
  "trigger_timestamp": "00:01:59.00",
  "frame_path": "data/grpo_dataset/frames/P01_02_78.jpg",
  "task_goal": "<video 첫 narration verb noun>",
  "gt_label": {"action","verb","noun","verb_class","noun_class"},
  "wm_output": {
    "top5_verb":[{"rank","verb","verb_class","likelihood"}],
    "top5_noun":[{"rank","noun","noun_class","likelihood"}],
    "top5_action":[{"rank","action","verb_class","noun_class","action_class","likelihood"}],
    "gt_in_top5_verb": bool, "gt_in_top5_noun": bool, "gt_in_top5_action": bool
  },
  "memory_context": {"task_history":[...], "temporal_proximity":{"t-0.5s","t-1.0s","t-2.0s"}}
}
```

## 핵심 설계 결정 / 주의사항 (다음 세션이 알아야 할 것)
- **anchor = trigger_frame = `stop_frame - int(1.0*fps)`** (action 종료 1초 전). ②③ 동일 시점 사용 → V-JEPA2 클립 끝과 VLM 프레임이 같은 프레임.
- **likelihood = softmax 후 top-k 확률** (null 아님). spec 2.215는 logits 노출되어 추출 가능 확인됨.
- **디스크 제약**: EK100 비디오 P01만 보유 → train 가용 샘플 1,348개로 한정. 추가 비디오(P02~) 확보 시 `select_train.py --target 5000` 재실행으로 확장 가능.
- **task_goal은 대리 정의**: video_id별 첫 narration의 "verb noun" (EK100엔 전체 task goal 없음). 데모 시 수동 오버라이드 가능.
- **task_history vs EK100**: EK100은 명확한 task 목표가 없어 task_history가 VLM 정확도에 악영향(-20pp) 가능 → 데이터셋엔 **저장만** 하고 EK100 평가/학습엔 비활성. 다른 데이터셋용으로 보존.
- ② 1건 에러(P01_09_880)는 영상 길이 < trigger_frame 인 annotation/disk 불일치. ③은 클램프 처리해 프레임은 생성될 수 있으나 prediction이 없어 ⑤에서 드롭됨 → 최종 ~1,346.

## 이 파이프라인 다음 단계 (범위 밖)
- 실제 GRPO 강화학습(Step 2): 위 `grpo_dataset.jsonl` 을 reward signal로 Qwen2.5-VL-7B 파인튜닝. WM(V-JEPA2) Top-5 likelihood가 reward 설계의 핵심 입력.
- Step 3 최종 평가는 EK100 **validation** set으로 (train은 GRPO 학습용으로 사용했으므로 오염 방지).
```
