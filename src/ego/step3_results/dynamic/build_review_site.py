"""정성 평가 사이트 빌더 — 한 에피소드를 위에서 아래로 훑으며 사람이 판정한다.

만드는 것:
  site/strips/<video_uid>/<sample_id>.jpg   관측 8프레임을 가로로 이어붙인 스트립
  site/index.html                            에피소드 목록 · 평가자 이름 · 진행률 · export
  site/ep_<video_uid>.html                   스텝 카드 세로 나열 + 평가 위젯

평가 설계 (3인 독립):
  · 스텝별 3지선다 — 타당 / 애매 / 부적절 : "이 프레임과 goal, 지금까지의 진행을 볼 때
    모델이 고른 다음 행동이 그럴듯한가". **GT 는 기본으로 가려 둔다** — 정답을 먼저 보면
    판단이 정답에 정박되어 정성 평가의 독립성이 깨진다. 토글로 열 수 있고, 열었다는 사실을
    `gt_revealed` 로 기록해 사후에 분리 집계할 수 있게 한다.
  · 에피소드별 1~5점 — "이 궤적 전체가 goal 을 수행했다고 볼 수 있는가" + 자유 메모.
  · 저장은 localStorage 자동, 제출은 `ratings_<평가자>.json` 다운로드. merge_ratings.py 가 집계.

사용:
  PYTHONPATH=src python -m ego.step3_results.dynamic.build_review_site --arm ego_closed
  python -m http.server 8899 --directory runs/dynamic_v1/site
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

from ego.step3_results.dynamic import common as C
from ego.step3_results.vpa.v2 import frames as F

STRIP_HEIGHT = 150


def build_strip(cache_root: Path, video_uid: str, sample_id: str, out: Path) -> bool:
    from PIL import Image

    if out.is_file():
        return True
    paths = F.frame_paths(cache_root, {"video_uid": video_uid, "sample_id": sample_id})
    if not all(p.is_file() for p in paths):
        return False
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        ims.append(im.resize((max(1, round(w * STRIP_HEIGHT / h)), STRIP_HEIGHT)))
    total = sum(im.size[0] for im in ims) + 2 * (len(ims) - 1)
    canvas = Image.new("RGB", (total, STRIP_HEIGHT), (255, 255, 255))
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0))
        x += im.size[0] + 2
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=80)
    return True


CSS = """
:root{--bg:#0f1115;--card:#171a21;--line:#272b36;--fg:#e6e8ee;--dim:#9aa3b2;--ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif}
a{color:var(--accent)}
header{position:sticky;top:0;z-index:10;background:#0f1115ee;backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:12px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:650}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 16px}
.step-head{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.k{font-weight:700;font-size:17px}
.t{color:var(--dim);font-size:13px}
img.strip{width:100%;border-radius:6px;display:block;border:1px solid var(--line)}
.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;margin-top:12px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.lbl{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin:10px 0 4px}
.pick{font-size:19px;font-weight:700;color:#fff;background:#1f6feb22;border:1px solid #1f6feb66;border-radius:6px;padding:6px 10px;display:inline-block}
.belief{background:#0d1117;border-left:3px solid var(--warn);padding:8px 10px;border-radius:4px;color:#d8dee9}
.reason{color:var(--dim);font-size:13.5px;white-space:pre-wrap}
ol.cands{margin:4px 0;padding-left:22px;color:var(--dim);font-size:13.5px;columns:2}
ol.cands li.picked{color:#fff;font-weight:650}
ol.cands li.wm1::after{content:" ← WM top-1";color:var(--warn);font-size:11px}
details summary{cursor:pointer;color:var(--dim);font-size:13px;user-select:none}
.gt{margin-top:6px;font-size:14px}
.gt .ok{color:var(--ok);font-weight:650}.gt .bad{color:var(--bad);font-weight:650}
.rate{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap;border-top:1px dashed var(--line);padding-top:10px}
.rate button{background:#21262d;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:6px 12px;cursor:pointer;font-size:14px}
.rate button.sel[data-v="ok"]{background:#238636;border-color:#3fb950}
.rate button.sel[data-v="mid"]{background:#9e6a03;border-color:#d29922}
.rate button.sel[data-v="no"]{background:#8b2c2c;border-color:#f85149}
input,textarea,select{background:#0d1117;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:6px 8px;font:inherit}
textarea{width:100%;min-height:70px}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;font-size:14px}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase}
.pill{font-size:12px;color:var(--dim);border:1px solid var(--line);border-radius:99px;padding:2px 8px}
.prog{height:6px;background:#21262d;border-radius:99px;overflow:hidden;width:130px}
.prog>i{display:block;height:100%;background:var(--ok)}
"""

JS_COMMON = """
const RK='dynplanRatings';
function rater(){return localStorage.getItem('dynplanRater')||''}
function setRater(v){localStorage.setItem('dynplanRater',v)}
function all(){try{return JSON.parse(localStorage.getItem(RK)||'{}')}catch(e){return {}}}
function save(o){localStorage.setItem(RK,JSON.stringify(o))}
function bucket(){const a=all(),r=rater()||'(unnamed)';a[r]=a[r]||{steps:{},episodes:{}};return [a,a[r]]}
function put(fn){const [a,b]=bucket();fn(b);save(a)}
function exportJSON(){
  const [a,b]=bucket();
  const blob=new Blob([JSON.stringify({rater:rater(),exported_at:new Date().toISOString(),...b},null,1)],{type:'application/json'});
  const u=URL.createObjectURL(blob),el=document.createElement('a');
  el.href=u;el.download='ratings_'+(rater()||'unnamed').replace(/\\s+/g,'_')+'.json';el.click();URL.revokeObjectURL(u);
}
function importJSON(ev){
  const f=ev.target.files[0];if(!f)return;const fr=new FileReader();
  fr.onload=()=>{const d=JSON.parse(fr.result);if(d.rater)setRater(d.rater);
    const a=all();a[d.rater||rater()]={steps:d.steps||{},episodes:d.episodes||{}};save(a);location.reload()};
  fr.readAsText(f);
}
"""


def episode_page(ep: dict, recs: list[dict], arm: str, strip_rel) -> str:
    cards = []
    for r in recs:
        sid = r["sample_id"]
        cand_items = []
        for c in r["candidates"]:
            cls = " ".join(x for x in [("picked" if c == r["pred_action"] else ""),
                                       ("wm1" if c == r["wm_top1"] else "")] if x)
            cand_items.append(f'<li class="{cls}">{html.escape(c)}</li>')
        ok = r["correct"]
        gt_line = (f'<div class="gt">GT: <span class="{"ok" if ok else "bad"}">{html.escape(r["gt_action"])}</span> '
                   f'&nbsp;<span class="pill">{"일치" if ok else "불일치"}</span> '
                   f'<span class="pill">{"GT가 WM top-10 안" if r["gt_in_candidates"] else "GT가 top-10 밖 (구조적으로 선택 불가)"}</span></div>')
        t = r["target_start_sec"]
        cards.append(f"""
<div class="card" id="s{r['step_idx']}" data-step="{r['step_idx']}">
  <div class="step-head">
    <span class="k">#{r['step_idx'] + 1}</span>
    <span class="t">t = {int(t) // 60:02d}:{t % 60:04.1f} · 관측 [{t - 5:.1f}s → {t - 1:.1f}s] · 8 frames @2fps</span>
  </div>
  <img class="strip" loading="lazy" src="{strip_rel(ep['video_uid'], sid)}" alt="obs frames">
  <div class="grid">
    <div>
      <div class="lbl">프레임워크가 고른 다음 행동</div>
      <div class="pick">{html.escape(r['pred_action'])}</div>
      <div class="lbl">이 시점의 task belief</div>
      <div class="belief">{html.escape(r['belief']) or '<i>(없음)</i>'}</div>
      <details><summary>reasoning 보기</summary><div class="reason">{html.escape(r['reasoning'])}</div></details>
      <details class="gtbox"><summary>정답(GT) 보기 — 판정 후에 열 것</summary>{gt_line}</details>
    </div>
    <div>
      <div class="lbl">월드 모델이 제시한 후보 10</div>
      <ol class="cands">{''.join(cand_items)}</ol>
    </div>
  </div>
  <div class="rate" data-sid="{sid}">
    <span class="t">이 행동이 그럴듯한가?</span>
    <button data-v="ok">타당</button><button data-v="mid">애매</button><button data-v="no">부적절</button>
    <input class="memo" placeholder="메모(선택)" style="flex:1;min-width:180px">
  </div>
</div>""")

    steps_json = json.dumps([r["sample_id"] for r in recs])
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(ep['goal_text'])} — closed-loop 정성 평가</title><style>{CSS}</style></head><body>
<header>
  <a href="index.html">← 목록</a>
  <h1>{html.escape(ep['goal_text'] or ep['video_uid'])}</h1>
  <span class="pill">{html.escape(ep['goal_category'])}</span>
  <span class="pill">{len(recs)} steps</span>
  <span class="pill">arm={arm}</span>
  <span style="flex:1"></span>
  <span class="t">평가자</span><input id="rater" size="10">
  <span class="prog"><i id="bar" style="width:0%"></i></span><span class="t" id="cnt">0/{len(recs)}</span>
  <button onclick="exportJSON()">결과 내보내기</button>
</header>
<div class="wrap">
<div class="card"><b>보는 법</b> — 각 카드는 결정지점 하나다. 스트립은 그 행동이 시작하기
<b>1초 전까지의 4초</b>를 8프레임으로 보여주며, 프레임워크는 <b>정답을 보지 않고</b> 월드 모델이 준 후보 10개
중에서 하나를 골랐다. 히스토리는 <b>모델이 앞서 스스로 고른 행동들</b>이다(GT 아님).
목표를 향해 그럴듯하게 진행하는지 위에서 아래로 훑어보고 각 스텝을 판정한 뒤, 맨 아래에서 궤적 전체를 채점한다.
정답은 판정 후에 열어보길 권한다.</div>
{''.join(cards)}
<div class="card" id="epbox">
  <div class="lbl">에피소드 종합 — 이 궤적 전체가 goal("{html.escape(ep['goal_text'])}")을 수행했다고 볼 수 있는가?</div>
  <div class="rate" id="eprate">
    <select id="epscore">
      <option value="">— 선택 —</option>
      <option value="5">5 · 목표를 일관되게 수행</option>
      <option value="4">4 · 대체로 수행, 일부 이탈</option>
      <option value="3">3 · 절반 정도만 말이 됨</option>
      <option value="2">2 · 대부분 이탈</option>
      <option value="1">1 · 목표와 무관</option>
    </select>
  </div>
  <div class="lbl">자유 메모 (어디서 무너졌는지, 복구했는지 등)</div>
  <textarea id="epmemo"></textarea>
</div>
</div>
<script>{JS_COMMON}
const EP="{ep['video_uid']}", SIDS={steps_json};
const ri=document.getElementById('rater'); ri.value=rater();
ri.oninput=()=>{{setRater(ri.value); render();}};
function render(){{
  const [,b]=bucket(); let n=0;
  document.querySelectorAll('.rate[data-sid]').forEach(div=>{{
    const sid=div.dataset.sid, cur=(b.steps||{{}})[sid];
    div.querySelectorAll('button').forEach(x=>x.classList.toggle('sel', !!cur && cur.v===x.dataset.v));
    div.querySelector('.memo').value=(cur&&cur.memo)||'';
    if(cur&&cur.v) n++;
  }});
  document.getElementById('cnt').textContent=n+'/'+SIDS.length;
  document.getElementById('bar').style.width=(100*n/SIDS.length)+'%';
  const e=(b.episodes||{{}})[EP]||{{}};
  document.getElementById('epscore').value=e.score||'';
  document.getElementById('epmemo').value=e.memo||'';
}}
document.querySelectorAll('.rate[data-sid]').forEach(div=>{{
  const sid=div.dataset.sid;
  div.querySelectorAll('button').forEach(btn=>btn.onclick=()=>{{
    put(b=>{{b.steps[sid]=Object.assign({{}},b.steps[sid],{{v:btn.dataset.v,ep:EP,ts:Date.now()}})}}); render();
  }});
  div.querySelector('.memo').oninput=ev=>put(b=>{{b.steps[sid]=Object.assign({{}},b.steps[sid],{{memo:ev.target.value,ep:EP}})}});
}});
document.getElementById('epscore').onchange=ev=>{{put(b=>{{b.episodes[EP]=Object.assign({{}},b.episodes[EP],{{score:ev.target.value}})}});}};
document.getElementById('epmemo').oninput=ev=>put(b=>{{b.episodes[EP]=Object.assign({{}},b.episodes[EP],{{memo:ev.target.value}})}});
document.querySelectorAll('details.gtbox').forEach(d=>d.ontoggle=()=>{{
  if(d.open){{const sid=d.closest('.card').querySelector('.rate').dataset.sid;
    put(b=>{{b.steps[sid]=Object.assign({{}},b.steps[sid],{{gt_revealed:true,ep:EP}})}});}}
}});
render();
</script></body></html>"""


def index_page(rows: list[dict], arm: str, stats: dict) -> str:
    trs = "".join(
        f"<tr><td><a href='ep_{r['video_uid']}.html'>{html.escape(r['goal_text'] or r['video_uid'])}</a></td>"
        f"<td class='t'>{html.escape(r['goal_category'])}</td><td>{r['n']}</td>"
        f"<td><span class='prog'><i data-ep='{r['video_uid']}' style='width:0%'></i></span> "
        f"<span class='t' data-cnt='{r['video_uid']}'>0/{r['n']}</span></td>"
        f"<td class='t' data-score='{r['video_uid']}'>—</td></tr>" for r in rows)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Closed-Loop Dynamic Planning — 정성 평가</title><style>{CSS}</style></head><body>
<header><h1>Closed-Loop Dynamic Planning · 정성 평가</h1>
<span class="pill">arm={arm}</span><span class="pill">{len(rows)} 에피소드</span>
<span class="pill">{sum(r['n'] for r in rows)} 스텝</span>
<span style="flex:1"></span>
<span class="t">평가자</span><input id="rater" size="12">
<button onclick="exportJSON()">결과 내보내기</button>
<label class="t" style="cursor:pointer">불러오기<input type="file" accept="application/json" onchange="importJSON(event)" style="display:none"></label>
</header>
<div class="wrap">
<div class="card"><b>무엇을 보는가</b> — 각 에피소드는 goal 하나를 가진 영상 하나다. 프레임워크는 그 영상의 결정지점을
처음부터 끝까지 따라가며, 매 시점 <b>월드 모델이 준 후보 10개</b> 중 하나를 고른다. 이때 히스토리로 들어가는 것은
<b>정답이 아니라 모델이 앞서 스스로 고른 행동과 자신의 belief</b>다. 즉 이 궤적은 사람이 궤도를 잡아주지 않은
<b>닫힌 루프</b>의 산물이다. 3인이 각자 이름을 넣고 독립적으로 판정한 뒤 <b>결과 내보내기</b>로 JSON을 저장한다.
<br><br><b>평가 순서</b> ① 이름 입력 → ② 에피소드를 위에서 아래로 훑으며 스텝별 타당/애매/부적절 →
③ 맨 아래에서 궤적 전체 1~5점 + 메모 → ④ 내보내기. 브라우저에 자동 저장되므로 중간에 닫아도 이어서 할 수 있다.</div>
<table><thead><tr><th>goal (영상)</th><th>category</th><th>steps</th><th>진행</th><th>종합점수</th></tr></thead>
<tbody>{trs}</tbody></table>
<div class="card t">데이터 통계: 스텝 {stats.get('n_steps')} · 에피소드 {stats.get('n_episodes')} ·
WM top-10 커버리지 {100 * stats.get('coverage_rate', 0):.1f}% (GT가 후보 안에 있는 비율 — 이 밖의 스텝은 구조적으로 정답 선택이 불가능하다)</div>
</div>
<script>{JS_COMMON}
const ri=document.getElementById('rater'); ri.value=rater(); ri.oninput=()=>{{setRater(ri.value);render()}};
function render(){{
  const [,b]=bucket(); const byEp={{}};
  Object.values(b.steps||{{}}).forEach(s=>{{if(s.v&&s.ep)byEp[s.ep]=(byEp[s.ep]||0)+1}});
  document.querySelectorAll('[data-cnt]').forEach(el=>{{
    const ep=el.dataset.cnt, tot=+el.textContent.split('/')[1], n=byEp[ep]||0;
    el.textContent=n+'/'+tot; document.querySelector(`[data-ep="${{ep}}"]`).style.width=(100*n/tot)+'%';
  }});
  document.querySelectorAll('[data-score]').forEach(el=>{{
    const e=(b.episodes||{{}})[el.dataset.score]; el.textContent=(e&&e.score)?e.score+'점':'—';
  }});
}}
render();
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="ego_closed")
    p.add_argument("--episodes", default="runs/dynamic_v1/episodes.json")
    p.add_argument("--pred-dir", default="runs/dynamic_v1/preds")
    p.add_argument("--site-dir", default="runs/dynamic_v1/site")
    p.add_argument("--cache-root", default=f"runs/vpa_v2/{F.cache_dirname()}")
    args = p.parse_args()

    data = C.load_json(args.episodes)
    eps = {e["video_uid"]: e for e in data["episodes"]}
    recs = C.read_jsonl(Path(args.pred_dir) / f"{args.arm}.records.jsonl")
    latest: dict[tuple, dict] = {}
    for r in recs:
        latest[(r["video_uid"], r["step_idx"])] = r
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for (v, _), r in sorted(latest.items()):
        by_ep[v].append(r)
    if not by_ep:
        raise SystemExit(f"{args.pred_dir}/{args.arm}.records.jsonl 에 기록이 없다")

    site = Path(args.site_dir)
    site.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root)
    rel = lambda v, s: f"strips/{v}/{s}.jpg"  # noqa: E731

    n_strip = 0
    for v, rs in by_ep.items():
        for r in rs:
            if build_strip(cache_root, v, r["sample_id"], site / "strips" / v / f"{r['sample_id']}.jpg"):
                n_strip += 1

    rows = []
    for v, rs in sorted(by_ep.items(), key=lambda kv: -len(kv[1])):
        rs.sort(key=lambda r: r["step_idx"])
        ep = eps[v]
        (site / f"ep_{v}.html").write_text(episode_page(ep, rs, args.arm, rel), encoding="utf-8")
        rows.append({"video_uid": v, "goal_text": ep["goal_text"],
                     "goal_category": ep["goal_category"], "n": len(rs)})
    (site / "index.html").write_text(index_page(rows, args.arm, data["stats"]), encoding="utf-8")

    print(f"에피소드 페이지 {len(rows)}개 · 스트립 {n_strip}장 · {site}/index.html")
    print(f"서빙:  python -m http.server 8899 --directory {site}")


if __name__ == "__main__":
    main()
