from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path


def _week(value) -> int:
    try:
        n = int(value)
    except Exception as exc:
        raise ValueError("Warmup week must be a positive integer.") from exc
    if n < 1 or n > 60:
        raise ValueError("Warmup week must be between 1 and 60.")
    return n


def config_path(config_root: Path, week_number: int) -> Path:
    n = _week(week_number)
    return config_root / f"week_{n:02d}.json"


def validate(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Warmup config must be an object.")
    out = deepcopy(config)
    n = _week(out.get("week_number"))
    out["schema_version"] = 1
    out["week_number"] = n
    out["title"] = str(out.get("title") or f"Week {n} Warmups").strip() or f"Week {n} Warmups"
    out["intro"] = str(out.get("intro") or "Warmups review only material taught before each day.").strip()
    settings = out.get("settings") if isinstance(out.get("settings"), dict) else {}
    try:
        workspace = float(settings.get("workspace", 0.55))
    except Exception:
        workspace = 0.55
    settings["workspace"] = max(0.0, min(3.0, workspace))
    out["settings"] = settings

    warmups = out.get("warmups") if isinstance(out.get("warmups"), list) else []
    clean_warmups = []
    for idx, raw_w in enumerate(warmups[:10], start=1):
        if not isinstance(raw_w, dict):
            continue
        w = deepcopy(raw_w)
        try:
            number = int(w.get("number") or idx)
        except Exception:
            number = idx
        w["number"] = number
        w["label"] = str(w.get("label") or "").strip()
        questions = w.get("questions") if isinstance(w.get("questions"), list) else []
        clean_q = []
        for qidx, raw_q in enumerate(questions[:6], start=1):
            if not isinstance(raw_q, dict):
                continue
            q = deepcopy(raw_q)
            q["id"] = str(q.get("id") or f"w{number}q{qidx}")
            q["visible"] = bool(q.get("visible", True))
            q["role"] = str(q.get("role") or ("Recent Review" if qidx == 1 else "Spiral Review")).strip()
            q["topic"] = str(q.get("topic") or "Previously taught skill").strip()
            calc = str(q.get("calculator") or "No Calculator").strip()
            q["calculator"] = calc if calc in {"No Calculator", "Calculator Permitted"} else "No Calculator"
            instances = q.get("instances") if isinstance(q.get("instances"), list) else []
            if not instances:
                instances = [{"label": "Original", "stem": "", "choices": ["", "", "", "", ""]}]
            clean_instances = []
            for iidx, inst in enumerate(instances[:12], start=1):
                if not isinstance(inst, dict):
                    continue
                choices = inst.get("choices") if isinstance(inst.get("choices"), list) else []
                choices = [str(x) for x in choices[:5]]
                while len(choices) < 5:
                    choices.append("")
                clean_instances.append({
                    "label": str(inst.get("label") or f"Instance {iidx}").strip() or f"Instance {iidx}",
                    "stem": str(inst.get("stem") or ""),
                    "choices": choices,
                })
            if not clean_instances:
                clean_instances = [{"label": "Original", "stem": "", "choices": ["", "", "", "", ""]}]
            q["instances"] = clean_instances
            try:
                active = int(q.get("active_instance", 0))
            except Exception:
                active = 0
            q["active_instance"] = max(0, min(active, len(clean_instances) - 1))
            clean_q.append(q)
        w["questions"] = clean_q
        clean_warmups.append(w)
    out["warmups"] = clean_warmups
    return out


def load(config_root: Path, week_number: int) -> dict:
    p = config_path(config_root, week_number)
    if not p.is_file():
        raise FileNotFoundError(f"No teacher-editable warmup config exists for Week {_week(week_number)} yet.")
    return validate(json.loads(p.read_text(encoding="utf-8")))


def student_paths(library_root: Path, apcalc_root: Path, public_base: str, week_number: int):
    n = _week(week_number)
    folder = f"week_{n:02d}_warmups"
    name = f"{folder}.html"
    local = library_root / "warmups" / folder / name
    public = apcalc_root / "warmups" / folder / name
    url = public_base.rstrip("/") + f"/warmups/{folder}/{name}"
    return local, public, url


def render(config: dict) -> str:
    config = validate(config)
    n = config["week_number"]
    workspace = float(config.get("settings", {}).get("workspace", 0.55))
    cards = []
    for warmup in config.get("warmups", []):
        qhtml = []
        for q in warmup.get("questions", []):
            if not q.get("visible", True):
                continue
            inst = q["instances"][q["active_instance"]]
            calc = q.get("calculator", "No Calculator")
            calc_cls = "calc" if calc == "Calculator Permitted" else "nocalc"
            choices = "".join(f"<li>{escape(choice)}</li>" for choice in inst.get("choices", []))
            qhtml.append(
                f'<div class="question"><div class="meta"><span class="badge {calc_cls}">{escape(calc)}</span>'
                f'<span class="role">{escape(q.get("role", ""))} · {escape(q.get("topic", ""))}</span></div>'
                f'<div class="stem">{escape(inst.get("stem", ""))}</div><ol class="choices" type="A">{choices}</ol><div class="work"></div></div>'
            )
        number = int(warmup.get("number", 0))
        cards.append(
            f'<section class="warmup" id="w{number}"><header class="warmup-head"><h2>{number}</h2>'
            f'<span class="date">{escape(warmup.get("label", ""))}</span></header>{"".join(qhtml)}</section>'
        )
    jumps = "".join(f'<a href="#w{int(w.get("number", 0))}">{int(w.get("number", 0))}</a>' for w in config.get("warmups", []))
    title = escape(config["title"])
    intro = escape(config["intro"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AP Calculus AB · {title}</title><script>window.MathJax={{tex:{{inlineMath:[["\\(","\\)"],["$","$"]],displayMath:[["\\[","\\]"],["$$","$$"]],processEscapes:true}}}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script><style>
:root{{--navy:#173f6d;--gold:#e0bd4f;--ink:#1f2937;--muted:#64748b;--line:#d6dde6;--bg:#eef2f6;--paper:#fff;--pale:#fff8dc;--calc:#e8f2fb;--nocalc:#fff3cc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);font-family:Arial,Helvetica,sans-serif;color:var(--ink)}}.screen-tools{{position:sticky;top:0;z-index:5;background:var(--navy);color:#fff;padding:10px 14px;display:flex;justify-content:center;align-items:center;gap:10px}}.screen-tools button{{border:1px solid rgba(255,255,255,.7);background:#fff;color:var(--navy);border-radius:9px;padding:9px 13px;font-weight:900;cursor:pointer}}.screen-tools span{{font-size:.78rem;opacity:.9}}main{{width:min(900px,calc(100% - 24px));margin:22px auto 48px}}.title{{background:var(--navy);color:#fff;padding:18px 20px;border-radius:16px 16px 0 0}}.title .kicker{{font-size:.78rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase;opacity:.85}}.title h1{{margin:5px 0 0;font-size:1.8rem}}.intro{{background:#fff;border:1px solid var(--line);border-top:0;padding:14px 18px;color:var(--muted);line-height:1.45}}.jump{{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}}.jump a{{color:var(--navy);font-weight:900;text-decoration:none;border:1px solid var(--gold);border-radius:999px;padding:5px 10px;background:#fff}}.warmup{{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin:16px 0;overflow:hidden;box-shadow:0 5px 16px rgba(23,63,109,.05)}}.warmup-head{{display:flex;justify-content:space-between;align-items:center;gap:12px;background:var(--pale);border-bottom:1px solid #ead690;padding:12px 16px}}.warmup-head h2{{margin:0;color:var(--navy);font-size:1.3rem}}.date{{font-size:.82rem;color:#725d18;font-weight:800}}.question{{padding:16px;border-bottom:1px solid #e8ebef}}.question:last-child{{border-bottom:0}}.meta{{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin-bottom:8px}}.badge{{font-size:.7rem;font-weight:900;text-transform:uppercase;letter-spacing:.04em;padding:5px 8px;border-radius:999px}}.badge.calc{{background:var(--calc);color:#245f91}}.badge.nocalc{{background:var(--nocalc);color:#725d18}}.role{{font-size:.76rem;color:var(--muted);font-weight:800}}.stem{{font-size:1rem;line-height:1.45}}.choices{{margin:10px 0 0 24px;padding-left:18px}}.choices li{{padding:4px 0;line-height:1.35}}.work{{height:{workspace:g}in}}.foot{{font-size:.78rem;color:var(--muted);text-align:center;margin-top:16px}}@media print{{body{{background:#fff}}.no-print,.intro,.foot{{display:none!important}}main{{width:100%;margin:0}}.title{{border-radius:0;padding:10px 14px}}.warmup{{box-shadow:none;break-inside:avoid;margin:8px 0;border-radius:0}}.warmup-head{{padding:7px 11px}}.question{{padding:9px 11px}}.stem,.choices{{font-size:10pt}}.work{{height:.35in}}}}
</style></head><body><div class="screen-tools no-print"><button type="button" onclick="window.print()">Print / Share to Notability</button><span>On iPad: open print preview, then Share → Notability.</span></div><main><header class="title"><div class="kicker">AP Calculus AB · Retrieval + Skill Strengthening</div><h1>{title}</h1></header><section class="intro">{intro}<div class="jump">{jumps}</div></section>{"".join(cards)}<div class="foot">Week {n} · Warmups review only material taught before each day.</div></main></body></html>'''


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save(config: dict, *, config_root: Path, library_root: Path, apcalc_root: Path, public_base: str, publish: bool, agenda_path: Path | None = None, agenda_html: str | None = None, history_path: Path | None = None) -> dict:
    config = validate(config)
    n = config["week_number"]
    config_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(config_path(config_root, n), json.dumps(config, indent=2) + "\n")
    local, public, url = student_paths(library_root, apcalc_root, public_base, n)
    _atomic_write(local, render(config))
    if publish:
        public.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, public)
        if agenda_path is not None and agenda_html is not None:
            _atomic_write(agenda_path, agenda_html)
        if history_path is not None:
            history = {"schema_version": 1, "weeks": {}}
            if history_path.is_file():
                try:
                    loaded = json.loads(history_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        history = loaded
                except Exception:
                    pass
            history.setdefault("weeks", {})[str(n)] = {"status": "published", "published_at": datetime.now().isoformat(timespec="seconds"), "public_url": url}
            _atomic_write(history_path, json.dumps(history, indent=2) + "\n")
    return {"ok": True, "config": config, "week_number": n, "local_path": str(local), "public_url": url, "published": publish, "message": (f"Published Week {n} Warmups to the student URL and refreshed the agenda. Use GitHub Sync when ready." if publish else f"Saved Week {n} Warmups locally. The public student page was not changed.")}


def weeks_payload(config_root: Path, library_root: Path, apcalc_root: Path, public_base: str) -> dict:
    config_root.mkdir(parents=True, exist_ok=True)
    weeks = []
    for path in sorted(config_root.glob("week_*.json"), key=lambda p: p.name):
        m = re.fullmatch(r"week_(\d+)\.json", path.name)
        if not m:
            continue
        n = int(m.group(1))
        _, public, url = student_paths(library_root, apcalc_root, public_base, n)
        weeks.append({"week_number": n, "title": f"Week {n} Warmups", "published": public.is_file(), "public_url": url})
    return {"ok": True, "weeks": weeks}
