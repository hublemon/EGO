#!/usr/bin/env python3
"""p3_smoke_trend.py — 스모크 스윕의 cons_loss 추세를 여러 통계로 나란히 본다.

체인의 선택 규칙은 '첫 점 − 마지막 점'인데, 관측점이 6개뿐이고 진폭이 ±1.5라
그 통계는 마지막 한 점에 좌우된다. 판단을 그 하나에 걸지 않기 위해
기울기(OLS)와 전후반 평균차를 함께 낸다. 셋이 어긋나면 신호가 없는 것이다.
"""
import json, sys
from pathlib import Path

RUN = Path("/mnt/nvme/migration/jihun/EGO/runs/p3_cons")

def ols_slope(y):
    n = len(y); xm = (n - 1) / 2; ym = sum(y) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(y))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0

print(f"{'cw':>6} {'n':>3} {'첫−끝':>8} {'기울기/점':>10} {'후반−전반':>10} {'진폭':>7} {'보상':>6}")
for d in sorted(x for x in RUN.glob("smoke_cw*") if x.is_dir()):
    rows = [json.loads(l) for l in open(d / "gr_log.jsonl") if l.strip()] if (d / "gr_log.jsonl").is_file() else []
    c = [r["cons_loss"] for r in rows if r.get("cons_loss") is not None]
    w = [r["reward_ma"] for r in rows if r.get("reward_ma") is not None]
    if len(c) < 3:
        print(f"{d.name.replace('smoke_cw',''):>6} {len(c):>3}  (점 부족)"); continue
    half = len(c) // 2
    print(f"{d.name.replace('smoke_cw',''):>6} {len(c):>3} {c[0]-c[-1]:>+8.2f} {ols_slope(c):>+10.3f} "
          f"{sum(c[half:])/len(c[half:]) - sum(c[:half])/len(c[:half]):>+10.2f} "
          f"{max(c)-min(c):>7.2f} {w[-1] if w else float('nan'):>6.2f}")
print("\n부호 규약: '첫−끝'은 + 가 하락(좋음), '기울기'와 '후반−전반'은 − 가 하락(좋음).\n세 통계의 방향이 일치해야 추세로 인정한다. 진폭이 추세보다 크면 노이즈다.")
