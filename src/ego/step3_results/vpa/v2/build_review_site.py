"""VPA v2 T4 full-history 결과를 사람이 비교 판정하는 정적 사이트를 만든다.

이 빌더는 모델을 다시 실행하지 않는다. 이미 저장된 네 arm의 4-step 예측, T4 GT,
그리고 VPA 프레임 캐시만 읽어 다음을 생성한다.

  site/index.html
      54개 영상별 탐색, 정량 지표 요약, 평가 진행률, JSON import/export
  site/video_<video_uid>.html
      각 VPA 샘플의 목표, 행동 히스토리, 관측 8프레임, 네 모델의 계획과 평가 위젯
  site/strips/<video_uid>/<sample_id>.jpg
      빠른 목록 표시용 관측 스트립
  site/zoom_strips_obs_n8_h336/<video_uid>/<sample_id>.jpg
      클릭 확대 뷰어용 관측 스트립

GT는 ``details`` 안에 가려 두며, 열었다는 사실을 ratings JSON에 기록한다. 평가는
브라우저 localStorage에 저장되므로 일부만 판정한 상태에서도 내보낼 수 있다.

사용:
  PYTHONPATH=src python -m ego.step3_results.vpa.v2.build_review_site
  python -m http.server 8898 --directory runs/vpa_v2/review_t4_full/site
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ego.step3_results.dynamic import build_review_site as D

N_FRAMES = 8
HORIZON = 4
ARM_SPECS = (
    (
        "ours_full",
        "EGO",
        "EGO adapter + 전체 action history",
    ),
    (
        "ours_wm1st",
        "EGO + WM",
        "EGO adapter + 전체 history + 첫 행동 WM 후보 제약",
    ),
    (
        "qwen_backbone",
        "Qwen backbone",
        "동일 backbone, EGO adapter 없음",
    ),
    (
        "frontier",
        "Frontier",
        "Gemini 2.5 Pro 기반 비교군",
    ),
)

VPA_CSS = D.CSS + r"""
button{background:#21262d;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:6px 11px;cursor:pointer;font:inherit}
button:hover{border-color:#58a6ff88}
.hero{background:linear-gradient(135deg,#17233a,#171a21 58%);border-color:#36537b}
.hero h2{font-size:23px;margin:0 0 8px}
.hero p{color:#c5cbd6;margin:6px 0}
.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px}
.metric-card{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--accent);border-radius:9px;padding:12px}
.metric-name{font-weight:750;margin-bottom:6px}
.metric-values{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
.metric-values div{background:#0d1117;border-radius:5px;padding:6px;text-align:center}
.metric-values b{display:block;font-size:16px}
.metric-values span{color:var(--dim);font-size:10px}
.overall-progress{display:flex;align-items:center;gap:8px}
.overall-progress .prog{width:180px}
.sample-card{scroll-margin-top:90px}
.context-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(260px,.8fr);gap:14px;margin-top:12px}
.context-box{background:#0d1117;border:1px solid var(--line);border-radius:7px;padding:10px}
.history-tail{display:flex;gap:6px;flex-wrap:wrap}
.action-chip{border:1px solid #364154;background:#202633;border-radius:99px;padding:3px 8px;font-size:12.5px}
.action-chip::before{content:attr(data-n);color:var(--dim);margin-right:5px}
.history-full{max-height:270px;overflow:auto;margin-top:7px}
.history-full ol{margin:5px 0;padding-left:27px}
.wm-list{columns:2;margin:5px 0;padding-left:24px}
.plans{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}
.plan{background:#11151c;border:1px solid var(--line);border-radius:9px;padding:11px;min-width:0}
.plan-head{display:flex;align-items:flex-start;gap:8px}
.plan-head strong{font-size:15px}
.plan-sub{color:var(--dim);font-size:11.5px;line-height:1.35;min-height:31px}
.arm-badge{margin-left:auto;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:#8cc8ff;border:1px solid #1f6feb66;border-radius:99px;padding:2px 6px}
.plan-seq,.gt-seq{list-style:none;padding:0;margin:9px 0;counter-reset:step}
.plan-seq li,.gt-seq li{counter-increment:step;display:flex;align-items:center;gap:8px;background:#1a202a;border-left:3px solid #58a6ff;padding:6px 8px;margin:4px 0;border-radius:3px;font-weight:600}
.plan-seq li::before,.gt-seq li::before{content:counter(step);display:grid;place-items:center;width:20px;height:20px;flex:0 0 20px;border-radius:50%;background:#303a49;color:#c9d1d9;font-size:11px}
.plan-rate{border-top:1px dashed var(--line);padding-top:9px}
.plan-rate .question{color:var(--dim);font-size:12px;margin-bottom:6px}
.rate-buttons{display:flex;gap:6px;flex-wrap:wrap}
.rate-buttons button{padding:5px 10px;font-size:13px}
.rate-buttons button.sel[data-v="ok"]{background:#238636;border-color:#3fb950}
.rate-buttons button.sel[data-v="mid"]{background:#9e6a03;border-color:#d29922}
.rate-buttons button.sel[data-v="no"]{background:#8b2c2c;border-color:#f85149}
.plan-rate input{width:100%;margin-top:7px;font-size:12.5px}
.gtbox{margin-top:14px;border:1px solid #d2992266;background:#d299220b;border-radius:8px;padding:9px 11px}
.gtbox>summary{color:#f0c56c;font-weight:650}
.gt-content{padding-top:8px}
.gt-seq li{border-left-color:var(--warn);background:#201c14}
.gt-compare{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}
.gt-score{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:7px 9px;font-size:12.5px}
.gt-score b{display:block;margin-bottom:2px}
.oktxt{color:var(--ok)}.badtxt{color:var(--bad)}
.sample-nav{display:flex;gap:7px;align-items:center}
.sample-nav a{text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:4px 9px}
.filter-row{display:flex;gap:8px;align-items:center;margin-bottom:10px}
.filter-row input{min-width:280px;flex:1}
.method-note{color:var(--dim);font-size:12px}
.method-note code{color:#c9d1d9}
.sticky-tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.no-results{display:none;padding:20px;text-align:center;color:var(--dim)}
@media(max-width:900px){
  .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .plans{grid-template-columns:1fr}
}
@media(max-width:700px){
  .context-grid{grid-template-columns:1fr}
  .gt-compare{grid-template-columns:1fr}
  .metric-grid{grid-template-columns:1fr}
  .wm-list{columns:1}
  .filter-row input{min-width:0}
}
"""

JS_COMMON = r"""
const RATING_KEY='vpaT4FullRatingsV1';
const RATER_KEY='vpaT4FullRaterV1';
const TASK_ID='vpa_t4_full_history_review';
const ARMS=['ours_full','ours_wm1st','qwen_backbone','frontier'];
function rater(){return localStorage.getItem(RATER_KEY)||''}
function setRater(v){localStorage.setItem(RATER_KEY,v)}
function allRatings(){
  try{return JSON.parse(localStorage.getItem(RATING_KEY)||'{}')}
  catch(_){return {}}
}
function saveRatings(v){localStorage.setItem(RATING_KEY,JSON.stringify(v))}
function bucket(){
  const all=allRatings(), name=rater()||'(unnamed)';
  all[name]=all[name]||{plans:{},samples:{}};
  all[name].plans=all[name].plans||{};
  all[name].samples=all[name].samples||{};
  return [all,all[name]];
}
function putRating(fn){
  const [all,current]=bucket();
  fn(current);
  saveRatings(all);
}
function safeName(v){return (v||'unnamed').trim().replace(/[^\p{L}\p{N}_.-]+/gu,'_')}
function exportJSON(){
  const [,current]=bucket();
  const payload={
    schema_version:1,
    task:TASK_ID,
    condition:{horizon:4,history:'full',with_frames:true,arms:ARMS},
    rater:rater(),
    exported_at:new Date().toISOString(),
    plans:current.plans||{},
    samples:current.samples||{}
  };
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob), link=document.createElement('a');
  link.href=url;
  link.download='vpa_t4_full_ratings_'+safeName(rater())+'.json';
  link.click();
  URL.revokeObjectURL(url);
}
function importJSON(event){
  const file=event.target.files?.[0];
  if(!file)return;
  const reader=new FileReader();
  reader.onload=()=>{
    try{
      const data=JSON.parse(reader.result);
      if(data.task&&data.task!==TASK_ID)throw new Error('다른 평가 과제의 JSON입니다.');
      const name=data.rater||rater()||'(unnamed)';
      setRater(data.rater||rater());
      const all=allRatings();
      all[name]={plans:data.plans||{},samples:data.samples||{}};
      saveRatings(all);
      location.reload();
    }catch(error){
      alert('평가 JSON을 불러오지 못했습니다: '+error.message);
    }
  };
  reader.readAsText(file);
}
function bindRater(onChange){
  const input=document.getElementById('rater');
  if(!input)return;
  input.value=rater();
  input.addEventListener('input',()=>{
    setRater(input.value);
    onChange();
  });
}
"""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_time(seconds: float) -> str:
    minutes = int(seconds) // 60
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:04.1f}"


def _sequence(actions: list[str], class_name: str) -> str:
    return (
        f'<ol class="{class_name}">'
        + "".join(f"<li>{_escape(action)}</li>" for action in actions)
        + "</ol>"
    )


def _frame_labels(sample: dict) -> str:
    start, end = float(sample["obs_start_sec"]), float(sample["obs_end_sec"])
    target = float(sample["target_start_sec"])
    labels = [
        f"t{(start + (end - start) * index / (N_FRAMES - 1) - target):+.2f}s"
        for index in range(N_FRAMES)
    ]
    return _escape(json.dumps(labels, ensure_ascii=False))


def _pred_file(phase_dir: Path, arm: str) -> Path:
    return phase_dir / "preds" / f"{arm}_T4.json"


def load_contract(
    gt_path: Path,
    subset_path: Path,
    phase_dir: Path,
) -> tuple[
    list[dict],
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, dict]],
    dict,
]:
    """Load and fail closed if any sample or prediction is absent/malformed."""
    gt = _load_json(gt_path)
    subset = _load_json(subset_path)
    summary = _load_json(phase_dir / "summary.json")
    if not isinstance(gt, list):
        raise ValueError(f"GT must be a list: {gt_path}")
    sample_ids = subset.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"invalid/duplicate sample_ids: {subset_path}")
    by_id = {row["sample_id"]: row for row in gt}
    missing_gt = [sid for sid in sample_ids if sid not in by_id]
    if missing_gt:
        raise ValueError(f"{len(missing_gt)} subset samples missing from GT: {missing_gt[:3]}")
    samples = [by_id[sid] for sid in sample_ids]
    if any(int(row.get("horizon", -1)) != HORIZON for row in samples):
        raise ValueError("non-T4 sample found in selected dataset")

    predictions: dict[str, dict[str, list[str]]] = {}
    scored_records: dict[str, dict[str, dict]] = {}
    expected = set(sample_ids)
    for arm, _, _ in ARM_SPECS:
        path = _pred_file(phase_dir, arm)
        raw = _load_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"predictions must be a dict: {path}")
        got = set(raw)
        if got != expected:
            raise ValueError(
                f"{arm} prediction coverage mismatch: "
                f"missing={len(expected - got)}, extra={len(got - expected)}"
            )
        bad = [sid for sid, plan in raw.items() if not isinstance(plan, list) or len(plan) != HORIZON]
        if bad:
            raise ValueError(f"{arm}: {len(bad)} predictions are not length {HORIZON}: {bad[:3]}")
        predictions[arm] = raw
        records_path = (
            phase_dir
            / "metrics"
            / f"records_{arm}_T4_frames_subset_T4.json"
        )
        records = _load_json(records_path)
        if not isinstance(records, list):
            raise ValueError(f"metric records must be a list: {records_path}")
        records_by_id = {row["sample_id"]: row for row in records}
        if len(records_by_id) != len(records) or set(records_by_id) != expected:
            raise ValueError(f"{arm} official metric record coverage/uniqueness mismatch")
        malformed = [
            sid
            for sid, row in records_by_id.items()
            if (
                not isinstance(row.get("pred"), list)
                or len(row["pred"]) != HORIZON
                or not isinstance(row.get("gt"), list)
                or len(row["gt"]) != HORIZON
            )
        ]
        if malformed:
            raise ValueError(f"{arm}: malformed official records: {malformed[:3]}")
        scored_records[arm] = records_by_id

    if (
        summary.get("horizon") != HORIZON
        or summary.get("history") != "full"
        or summary.get("with_frames") is not True
    ):
        raise ValueError("summary does not describe T4 full-history with-frames")
    for arm, _, _ in ARM_SPECS:
        metric = summary.get("metrics", {}).get(arm)
        if not metric or metric.get("n") != len(samples) or metric.get("coverage") != 100.0:
            raise ValueError(f"summary metric is incomplete for {arm}")
    return samples, predictions, scored_records, summary


def _plan_block(
    sample: dict,
    arm: str,
    label: str,
    description: str,
    prediction: list[str],
) -> str:
    sid = _escape(sample["sample_id"])
    video_uid = _escape(sample["video_uid"])
    return f"""
<section class="plan" data-arm="{arm}">
  <div class="plan-head">
    <div>
      <strong>{_escape(label)}</strong>
      <div class="plan-sub">{_escape(description)}</div>
    </div>
    <span class="arm-badge">{arm}</span>
  </div>
  {_sequence(prediction, "plan-seq")}
  <div class="plan-rate" data-sid="{sid}" data-arm="{arm}" data-video="{video_uid}">
    <div class="question">관측·목표·이력만 볼 때 이 4-step 계획이 타당한가?</div>
    <div class="rate-buttons">
      <button type="button" data-v="ok" aria-pressed="false">타당</button>
      <button type="button" data-v="mid" aria-pressed="false">애매</button>
      <button type="button" data-v="no" aria-pressed="false">부적절</button>
    </div>
    <input class="memo" aria-label="{_escape(label)} 판정 메모" placeholder="이 계획에 대한 메모 (선택)">
  </div>
</section>"""


def _gt_block(
    sample: dict,
    predictions: dict[str, dict[str, list[str]]],
    scored_records: dict[str, dict[str, dict]],
) -> str:
    gt = sample["future_actions"]
    rows = []
    for arm, label, _ in ARM_SPECS:
        sid = sample["sample_id"]
        raw_prediction = predictions[arm][sid]
        official = scored_records[arm][sid]
        exact_sequence = bool(official["success"])
        mapped_note = ""
        if official["pred"] != raw_prediction:
            mapped_note = (
                '<details><summary>공식 vocabulary mapping 결과 보기</summary>'
                f'{_sequence(official["pred"], "plan-seq")}</details>'
            )
        exact_class = "oktxt" if exact_sequence else "badtxt"
        rows.append(
            f'<div class="gt-score"><b>{_escape(label)}</b>'
            f'위치 일치 <strong>{int(official["correct"])}/{HORIZON}</strong> · '
            f'집합 IoU <strong>{100 * float(official["iou"]):.1f}%</strong> · '
            f'<span class="{exact_class}">'
            f'{"전체 일치" if exact_sequence else "전체 불일치"}</span>'
            f'{mapped_note}</div>'
        )
    return f"""
<details class="gtbox" data-sid="{_escape(sample['sample_id'])}">
  <summary>정답(GT) 4-step 보기 — 네 계획을 판정한 후에 열 것</summary>
  <div class="gt-content">
    <div class="lbl">실제 미래 행동 순서</div>
    {_sequence(gt, "gt-seq")}
    <div class="lbl">저장된 정량 판정</div>
    <div class="gt-compare">{''.join(rows)}</div>
    <p class="t">본문 계획은 모델의 raw parsed output이다. 위 정량 판정은 공식 vocabulary mapping 뒤의
    evaluation record를 그대로 사용한다. mapping으로 본문과 달라진 경우에만 변환 결과 토글이 나타난다.
    GT를 연 사실은 내보내는 평가 JSON에 기록된다.</p>
  </div>
</details>"""


def _sample_card(
    sample: dict,
    sample_number: int,
    predictions: dict[str, dict[str, list[str]]],
    scored_records: dict[str, dict[str, dict]],
    strip_path: str,
    zoom_path: str,
) -> str:
    observed = sample["observed_actions"]
    tail_start = max(0, len(observed) - 8)
    tail = "".join(
        f'<span class="action-chip" data-n="{index + 1}">{_escape(action)}</span>'
        for index, action in enumerate(observed[tail_start:], start=tail_start)
    )
    full_history = "".join(
        f"<li>{_escape(action)}</li>" for action in observed
    )
    candidates = "".join(
        f"<li>{_escape(candidate)}</li>" for candidate in sample.get("wm_candidates", [])
    )
    plan_blocks = "".join(
        _plan_block(
            sample,
            arm,
            label,
            description,
            predictions[arm][sample["sample_id"]],
        )
        for arm, label, description in ARM_SPECS
    )
    target = float(sample["target_start_sec"])
    obs_start = float(sample["obs_start_sec"])
    obs_end = float(sample["obs_end_sec"])
    return f"""
<article class="card sample-card" id="sample-{_escape(sample['sample_id'])}"
         data-sample="{_escape(sample['sample_id'])}">
  <div class="step-head">
    <span class="k">#{sample_number}</span>
    <span class="t">target {_fmt_time(target)} · 관측 [{obs_start:.2f}s → {obs_end:.2f}s] ·
      target까지 {target - obs_end:.2f}s gap</span>
    <span class="pill">{_escape(sample['scenario'])}</span>
  </div>
  <div class="strip-caption">
    <span class="lbl" style="margin:0">모델 입력 관측 · 8 frames / 4초</span>
    <span class="zoom-hint">프레임 클릭: 확대 · 좌우 버튼/키로 이동</span>
  </div>
  <img class="strip frame-strip" loading="lazy"
       src="{_escape(strip_path)}" data-viewer-src="{_escape(zoom_path)}"
       data-frame-count="{N_FRAMES}" data-frame-gap="{D.STRIP_GAP}"
       data-frame-kind="VPA 모델 입력 관측" data-frame-labels="{_frame_labels(sample)}"
       role="button" tabindex="0" aria-label="모델 입력 관측 프레임 확대 보기"
       alt="VPA 모델 입력 관측 8프레임">
  <div class="context-grid">
    <div class="context-box">
      <div class="lbl" style="margin-top:0">모델 입력: 완료된 action history · {len(observed)} actions</div>
      <div class="history-tail">{tail}</div>
      <details class="history-full">
        <summary>전체 이력 {len(observed)}개 보기</summary>
        <ol>{full_history}</ol>
      </details>
    </div>
    <div class="context-box">
      <div class="lbl" style="margin-top:0">WM 다음 행동 후보 · 10개</div>
      <details>
        <summary>후보 목록 보기</summary>
        <ol class="wm-list">{candidates}</ol>
      </details>
    </div>
  </div>
  <div class="plans">{plan_blocks}</div>
  {_gt_block(sample, predictions, scored_records)}
</article>"""


def video_page(
    video_uid: str,
    samples: list[dict],
    predictions: dict[str, dict[str, list[str]]],
    scored_records: dict[str, dict[str, dict]],
) -> str:
    cards = []
    for number, sample in enumerate(samples, 1):
        cards.append(
            _sample_card(
                sample,
                number,
                predictions,
                scored_records,
                f"strips/{video_uid}/{sample['sample_id']}.jpg",
                f"{D.OBS_ZOOM_DIRNAME}/{video_uid}/{sample['sample_id']}.jpg",
            )
        )
    goal_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        goal_counts[sample["goal_text"]] += 1
    goal_summary = " · ".join(
        f"{goal} ({count})" if count > 1 else goal
        for goal, count in goal_counts.items()
    )
    sid_list = json.dumps([sample["sample_id"] for sample in samples])
    video_json = json.dumps(video_uid)
    viewer_html = D.FRAME_VIEWER_HTML.replace("</dialog>\n</dialog>", "</dialog>")
    page_js = r"""
const VIDEO_UID=__VIDEO__;
const SAMPLE_IDS=__SAMPLES__;
const TOTAL=SAMPLE_IDS.length*ARMS.length;
function planKey(sid,arm){return sid+'::'+arm}
function render(){
  const [,current]=bucket();
  let rated=0;
  document.querySelectorAll('.plan-rate').forEach(box=>{
    const key=planKey(box.dataset.sid,box.dataset.arm);
    const record=(current.plans||{})[key];
    box.querySelectorAll('button[data-v]').forEach(button=>{
      const selected=!!record&&record.v===button.dataset.v;
      button.classList.toggle('sel',selected);
      button.setAttribute('aria-pressed',selected?'true':'false');
    });
    box.querySelector('.memo').value=record?.memo||'';
    if(record?.v)rated++;
  });
  document.getElementById('count').textContent=rated+'/'+TOTAL;
  document.getElementById('bar').style.width=(TOTAL?100*rated/TOTAL:0)+'%';
}
bindRater(render);
document.querySelectorAll('.plan-rate').forEach(box=>{
  const sid=box.dataset.sid,arm=box.dataset.arm,video=box.dataset.video;
  box.querySelectorAll('button[data-v]').forEach(button=>{
    button.addEventListener('click',()=>{
      putRating(current=>{
        const key=planKey(sid,arm),old=current.plans[key]||{};
        current.plans[key]={...old,sid,arm,video_uid:video,v:button.dataset.v,ts:Date.now()};
      });
      render();
    });
  });
  box.querySelector('.memo').addEventListener('input',event=>{
    putRating(current=>{
      const key=planKey(sid,arm),old=current.plans[key]||{};
      current.plans[key]={...old,sid,arm,video_uid:video,memo:event.target.value,ts:Date.now()};
    });
  });
});
document.querySelectorAll('details.gtbox').forEach(details=>{
  details.addEventListener('toggle',()=>{
    if(!details.open)return;
    const sid=details.dataset.sid;
    putRating(current=>{
      current.samples[sid]={
        ...(current.samples[sid]||{}),
        sid,
        video_uid:VIDEO_UID,
        gt_revealed:true,
        gt_revealed_at:Date.now()
      };
    });
  });
});
render();
"""
    page_js = page_js.replace("__VIDEO__", video_json).replace("__SAMPLES__", sid_list)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>{_escape(goal_summary)} — VPA T4 정성 평가</title>
<style>{VPA_CSS}</style></head><body>
<header>
  <a href="index.html">← 영상 목록</a>
  <h1>VPA T4 · {_escape(goal_summary)}</h1>
  <span class="pill">{len(samples)} samples</span>
  <span style="flex:1"></span>
  <div class="sticky-tools">
    <span class="t">평가자</span><input id="rater" size="10" autocomplete="off">
    <span class="prog"><i id="bar" style="width:0%"></i></span>
    <span class="t" id="count">0/{len(samples) * len(ARM_SPECS)}</span>
    <button type="button" onclick="exportJSON()">부분 결과 내보내기</button>
  </div>
</header>
<main class="wrap">
  <section class="card hero">
    <h2>{_escape(goal_summary)}</h2>
    <p>각 카드는 서로 독립적인 VPA 문항입니다. 목표, 완료 행동 이력, 관측 프레임만 보고
    네 모델의 <b>앞으로 4개 행동 계획</b>을 각각 판정하세요. 모든 문항을 채우지 않아도 저장·내보내기가 됩니다.</p>
    <p>GT는 네 계획을 먼저 판정한 뒤 여는 것을 권장합니다. 관측 스트립의 개별 프레임을 클릭하면
    크게 열리고, 화면 버튼 또는 키보드 ← → 키로 이웃 프레임을 볼 수 있습니다.</p>
  </section>
  {''.join(cards)}
</main>
{viewer_html}
<script>{JS_COMMON}
{page_js}
{D.FRAME_VIEWER_JS}
</script></body></html>"""


def _metric_card(arm: str, label: str, description: str, metric: dict) -> str:
    return f"""
<section class="metric-card">
  <div class="metric-name">{_escape(label)} <span class="arm-badge">{arm}</span></div>
  <div class="method-note">{_escape(description)}</div>
  <div class="metric-values">
    <div><b>{metric['SR']:.2f}</b><span>SR (%)</span></div>
    <div><b>{metric['mAcc']:.2f}</b><span>mAcc (%)</span></div>
    <div><b>{metric['mIoU']:.2f}</b><span>mIoU (%)</span></div>
  </div>
</section>"""


def index_page(rows: list[dict], summary: dict) -> str:
    cards = "".join(
        _metric_card(arm, label, description, summary["metrics"][arm])
        for arm, label, description in ARM_SPECS
    )
    table_rows = "".join(
        f'<tr data-video-row data-search="{_escape((row["goals"] + " " + row["video_uid"]).lower())}">'
        f'<td><a href="video_{_escape(row["video_uid"])}.html">{_escape(row["goals"])}</a>'
        f'<div class="t">{_escape(row["video_uid"])}</div></td>'
        f'<td>{row["n"]}</td><td>{row["n"] * len(ARM_SPECS)}</td>'
        f'<td><span class="prog"><i data-bar="{_escape(row["video_uid"])}" style="width:0%"></i></span> '
        f'<span class="t" data-count="{_escape(row["video_uid"])}">0/{row["n"] * len(ARM_SPECS)}</span></td>'
        f'<td class="t" data-revealed="{_escape(row["video_uid"])}">0/{row["n"]}</td></tr>'
        for row in rows
    )
    video_info = {
        row["video_uid"]: {
            "n": row["n"],
            "sids": row["sample_ids"],
        }
        for row in rows
    }
    index_js = r"""
const VIDEO_INFO=__VIDEO_INFO__;
const TOTAL=Object.values(VIDEO_INFO).reduce((n,v)=>n+v.n*ARMS.length,0);
function render(){
  const [,current]=bucket();
  const plans=current.plans||{},samples=current.samples||{};
  let allRated=0;
  for(const [video,info] of Object.entries(VIDEO_INFO)){
    let rated=0,revealed=0;
    for(const sid of info.sids){
      for(const arm of ARMS){
        if(plans[sid+'::'+arm]?.v)rated++;
      }
      if(samples[sid]?.gt_revealed)revealed++;
    }
    allRated+=rated;
    document.querySelector(`[data-bar="${video}"]`).style.width=(100*rated/(info.n*ARMS.length))+'%';
    document.querySelector(`[data-count="${video}"]`).textContent=rated+'/'+(info.n*ARMS.length);
    document.querySelector(`[data-revealed="${video}"]`).textContent=revealed+'/'+info.n;
  }
  document.getElementById('overallCount').textContent=allRated+'/'+TOTAL;
  document.getElementById('overallBar').style.width=(100*allRated/TOTAL)+'%';
}
bindRater(render);
const filter=document.getElementById('filter');
filter.addEventListener('input',()=>{
  const query=filter.value.trim().toLowerCase();
  let visible=0;
  document.querySelectorAll('[data-video-row]').forEach(row=>{
    const show=!query||row.dataset.search.includes(query);
    row.hidden=!show;
    if(show)visible++;
  });
  document.getElementById('noResults').style.display=visible?'none':'block';
});
render();
"""
    index_js = index_js.replace(
        "__VIDEO_INFO__",
        json.dumps(video_info, ensure_ascii=False, separators=(",", ":")),
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>VPA T4 Full-History · 정성 평가</title>
<style>{VPA_CSS}</style></head><body>
<header>
  <h1>VPA T4 Full-History · 정성 평가</h1>
  <span class="pill">{sum(row['n'] for row in rows)} samples</span>
  <span class="pill">{len(rows)} videos</span>
  <span style="flex:1"></span>
  <div class="sticky-tools">
    <span class="t">평가자</span><input id="rater" size="10" autocomplete="off">
    <div class="overall-progress">
      <span class="prog"><i id="overallBar" style="width:0%"></i></span>
      <span class="t" id="overallCount">0/{sum(row['n'] for row in rows) * len(ARM_SPECS)}</span>
    </div>
    <button type="button" onclick="exportJSON()">부분 결과 내보내기</button>
    <label class="t" style="cursor:pointer">결과 불러오기
      <input type="file" accept="application/json" onchange="importJSON(event)" style="display:none">
    </label>
  </div>
</header>
<main class="wrap">
  <section class="card hero">
    <h2>저장된 VPA 결과만으로 4-step 계획 비교</h2>
    <p><b>T=4 · full action history · 8 visual frames</b> 조건의 504개 문항입니다.
    EGO, EGO+WM 첫-step 제약, Qwen backbone, Frontier의 저장된 예측을 같은 입력 문맥에서 비교합니다.</p>
    <p>영상별 묶음은 탐색을 위한 것입니다. 각 샘플은 독립 문항이며, 평가는 계획마다
    타당 / 애매 / 부적절 중 하나와 선택 메모를 남깁니다. 몇 개만 평가해도 브라우저에 자동 저장되고
    JSON으로 내보낼 수 있습니다.</p>
  </section>
  <div class="metric-grid">{cards}</div>
  <section class="card">
    <div class="filter-row">
      <label for="filter" class="t">영상/goal 검색</label>
      <input id="filter" type="search" placeholder="goal 또는 video UID">
    </div>
    <table>
      <thead><tr><th>Goal / video</th><th>문항</th><th>계획 판정</th><th>진행률</th><th>GT 열람</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    <div id="noResults" class="no-results">검색 결과가 없습니다.</div>
  </section>
  <section class="card method-note">
    정량 지표는 기존 <code>T4_full/summary.json</code> 값을 그대로 표시합니다.
    SR은 4-step 전체 exact match, mAcc는 위치별 정확도, mIoU는 행동 집합 IoU입니다.
    이 사이트는 추론을 다시 실행하거나 서버로 평가 내용을 전송하지 않습니다.
  </section>
</main>
<script>{JS_COMMON}
{index_js}
</script></body></html>"""


def build_site(
    samples: list[dict],
    predictions: dict[str, dict[str, list[str]]],
    scored_records: dict[str, dict[str, dict]],
    summary: dict,
    cache_root: Path,
    site_dir: Path,
) -> dict:
    site_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    missing_strips: list[str] = []
    for number, sample in enumerate(samples, 1):
        video_uid, sid = sample["video_uid"], sample["sample_id"]
        grouped[video_uid].append(sample)
        thumb = site_dir / "strips" / video_uid / f"{sid}.jpg"
        zoom = site_dir / D.OBS_ZOOM_DIRNAME / video_uid / f"{sid}.jpg"
        if not D.build_strip(cache_root, video_uid, sid, thumb, zoom):
            missing_strips.append(sid)
        if number % 50 == 0 or number == len(samples):
            print(f"[strips] {number}/{len(samples)}")
    if missing_strips:
        raise RuntimeError(
            f"{len(missing_strips)} observation strips could not be built: {missing_strips[:5]}"
        )

    rows = []
    for video_uid in sorted(grouped):
        video_samples = sorted(grouped[video_uid], key=lambda row: float(row["target_start_sec"]))
        grouped[video_uid] = video_samples
        goals = list(dict.fromkeys(row["goal_text"] for row in video_samples))
        goal_text = " / ".join(goals)
        rows.append(
            {
                "video_uid": video_uid,
                "goals": goal_text,
                "n": len(video_samples),
                "sample_ids": [row["sample_id"] for row in video_samples],
            }
        )
        (site_dir / f"video_{video_uid}.html").write_text(
            video_page(video_uid, video_samples, predictions, scored_records),
            encoding="utf-8",
        )
    (site_dir / "index.html").write_text(index_page(rows, summary), encoding="utf-8")
    (site_dir / "robots.txt").write_text(
        "User-agent: *\nDisallow: /\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "task": "vpa_t4_full_history_review",
        "generated_at": datetime.now(UTC).isoformat(),
        "condition": {
            "horizon": HORIZON,
            "history": "full",
            "with_frames": True,
            "n_observation_frames": N_FRAMES,
        },
        "counts": {
            "samples": len(samples),
            "videos": len(grouped),
            "plans": len(samples) * len(ARM_SPECS),
            "observation_strips": len(samples),
        },
        "arms": {
            arm: {
                "label": label,
                "description": description,
                "metrics": summary["metrics"][arm],
            }
            for arm, label, description in ARM_SPECS
        },
    }
    (site_dir / "site_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gt", type=Path, default=Path("runs/vpa_v2/vpa_v2_T4.json"))
    parser.add_argument(
        "--subset",
        type=Path,
        default=Path("runs/vpa_v2/frames_subset_T4.json"),
    )
    parser.add_argument(
        "--phase-dir",
        type=Path,
        default=Path("runs/vpa_v2/action_history_ablation/T4_full"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("runs/vpa_v2/frame_cache_w4_g1_n8_s336"),
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path("runs/vpa_v2/review_t4_full/site"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, predictions, scored_records, summary = load_contract(
        args.gt,
        args.subset,
        args.phase_dir,
    )
    print(
        f"[contract] {len(samples)} T4 samples · "
        f"{len({row['video_uid'] for row in samples})} videos · "
        f"{len(ARM_SPECS)} stored prediction arms"
    )
    manifest = build_site(
        samples,
        predictions,
        scored_records,
        summary,
        args.cache_root,
        args.site_dir,
    )
    print(
        f"[done] {args.site_dir} · "
        f"{manifest['counts']['samples']} samples · "
        f"{manifest['counts']['plans']} plan judgments"
    )


if __name__ == "__main__":
    main()
