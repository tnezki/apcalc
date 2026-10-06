#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import webbrowser
import zipfile
import warmup_editor
from copy import deepcopy
from datetime import date, datetime
from html import escape
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

HOST = "127.0.0.1"
PORT = 8769
PLANNER_VERSION = "1.0-authoritative"
HERE = Path(__file__).resolve().parent
TEACHER_TOOLS_ROOT = HERE.parent
GITHUB_ROOT = HERE.parents[1]
APCALC_ROOT = GITHUB_ROOT / "apcalc"
AGENDA_PATH = APCALC_ROOT / "agenda" / "index.html"
SHARED_ROOT = GITHUB_ROOT / "teacher_shared"
SHARED_CALC_ROOT = SHARED_ROOT / "calc"
SHARED_INDEX = SHARED_CALC_ROOT / "index.html"
PRIVATE_ARTIFACTS = SHARED_CALC_ROOT / "data" / "private_artifacts.json"
STATE_PATH = HERE / "planner_state.json"
LIBRARY_ROOT = TEACHER_TOOLS_ROOT / "library"
LIBRARY_CATEGORIES = ("notes", "exercises", "sets", "explore", "lab", "warmups")
QUICK_CHECK_LIBRARY = APCALC_ROOT / "quick_check____htq5855" / "library.json"
PUBLIC_BASE = "https://tnezki.github.io/apcalc/"
WARMUP_BUILDER_ROOT = TEACHER_TOOLS_ROOT / "warmup_builder"
WARMUP_HISTORY_PATH = WARMUP_BUILDER_ROOT / "warmup_history.json"
WARMUP_PENDING_DIR = WARMUP_BUILDER_ROOT / "pending"
WARMUP_BACKUP_DIR = WARMUP_BUILDER_ROOT / "backups"
WARMUP_REQUESTS_DIR = TEACHER_TOOLS_ROOT / "requests" / "outgoing"
WARMUP_RESULT_MAX_BYTES = 25_000_000
EXERCISE_BUILDER_ROOT = TEACHER_TOOLS_ROOT / "exercise_builder"
EXERCISE_CONFIG_ROOT = EXERCISE_BUILDER_ROOT / "sections"
WARMUP_CONFIG_ROOT = WARMUP_BUILDER_ROOT / "weeks"

STUDENT_RESOURCE_OPTIONS = [
    "", "Warmups", "Notes", "Exercises", "Sets", "Explore", "Lab",
    "Quick Check", "Review", "Custom Item",
    # Legacy labels remain valid so earlier agenda history keeps working.
    "Warm Up", "Investigation", "Activity", "Demo", "Performance Task", "Practice Set", "Extra Practice",
]
EDITOR_RESOURCE_OPTIONS = [
    "", "Warmups", "Notes", "Exercises", "Sets", "Explore", "Lab",
    "Quick Check", "Review", "Custom Item",
]
DAY_CONTROLS = [
    "Regular Day", "Insert Blank Day", "Bump Back", "Snow Day", "P.D.",
    "1/2 Day", "Break", "No School", "Assembly Schedule", "Testing", "Sub Assignment",
]
SHIFTING_CONTROLS = {
    "Insert Blank Day", "Snow Day", "P.D.", "Break", "No School"
}
BLOCKED_CONTROLS = {"Snow Day", "P.D.", "Break", "No School"}
EDITOR_DAY_CONTROLS = list(DAY_CONTROLS)




def _git_run(repo: Path, args: list[str], timeout: int = 120) -> tuple[int, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "").strip()


def _git_repositories() -> list[Path]:
    repos = []
    if not GITHUB_ROOT.is_dir():
        return repos
    for child in sorted(GITHUB_ROOT.iterdir(), key=lambda x: x.name.casefold()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if (child / ".git").exists():
            repos.append(child)
    return repos


def git_pull_all() -> dict:
    results = []
    for repo in _git_repositories():
        code, branch = _git_run(repo, ["branch", "--show-current"], timeout=20)
        branch = branch.strip() if code == 0 else ""
        if not branch:
            results.append({"repo": repo.name, "ok": False, "detail": "Skipped: no current branch."})
            continue
        code, remote = _git_run(repo, ["remote", "get-url", "origin"], timeout=20)
        if code != 0:
            results.append({"repo": repo.name, "ok": False, "detail": "Skipped: no origin remote."})
            continue
        code, out = _git_run(repo, ["pull", "--rebase", "--autostash"], timeout=180)
        results.append({"repo": repo.name, "ok": code == 0, "detail": out or ("Up to date." if code == 0 else "Pull failed.")})
    failed = [r for r in results if not r["ok"]]
    return {
        "ok": not failed,
        "action": "pull",
        "results": results,
        "message": (
            f"Pulled {len(results)} GitHub repos." if not failed
            else f"Pull finished with {len(failed)} repo(s) needing attention: " + ", ".join(r["repo"] for r in failed)
        ),
    }


def git_push_all() -> dict:
    results = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for repo in _git_repositories():
        code, branch = _git_run(repo, ["branch", "--show-current"], timeout=20)
        branch = branch.strip() if code == 0 else ""
        if not branch:
            results.append({"repo": repo.name, "ok": False, "detail": "Skipped: no current branch."})
            continue
        code, remote = _git_run(repo, ["remote", "get-url", "origin"], timeout=20)
        if code != 0:
            results.append({"repo": repo.name, "ok": False, "detail": "Skipped: no origin remote."})
            continue
        code, out = _git_run(repo, ["add", "-A"], timeout=60)
        if code != 0:
            results.append({"repo": repo.name, "ok": False, "detail": out or "git add failed."})
            continue
        code, staged = _git_run(repo, ["diff", "--cached", "--quiet"], timeout=30)
        if code == 1:
            code, out = _git_run(repo, ["commit", "-m", f"Teacher Tools sync {stamp}"], timeout=120)
            if code != 0:
                results.append({"repo": repo.name, "ok": False, "detail": out or "Commit failed."})
                continue
        elif code not in (0, 1):
            results.append({"repo": repo.name, "ok": False, "detail": staged or "Could not inspect staged changes."})
            continue
        code, out = _git_run(repo, ["pull", "--rebase"], timeout=180)
        if code != 0:
            results.append({"repo": repo.name, "ok": False, "detail": out or "Pull before push failed."})
            continue
        code, out = _git_run(repo, ["push"], timeout=180)
        results.append({"repo": repo.name, "ok": code == 0, "detail": out or ("Pushed." if code == 0 else "Push failed.")})
    failed = [r for r in results if not r["ok"]]
    return {
        "ok": not failed,
        "action": "push",
        "results": results,
        "message": (
            f"Committed and pushed {len(results)} GitHub repos." if not failed
            else f"Push finished with {len(failed)} repo(s) needing attention: " + ", ".join(r["repo"] for r in failed)
        ),
    }

def ensure_calc_tools_layout() -> None:
    LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    for category in LIBRARY_CATEGORIES:
        (LIBRARY_ROOT / category).mkdir(parents=True, exist_ok=True)


def _published_url_for_library_item(category: str, path: Path) -> str | None:
    public_root = APCALC_ROOT / category
    if path.is_file():
        target = public_root / path.name
        if target.is_file():
            return PUBLIC_BASE + quote(f"{category}/{path.name}", safe="/._-")
        return None

    published_folder = public_root / path.name
    if not published_folder.is_dir():
        return None
    preferred = published_folder / f"{path.name}.html"
    if preferred.is_file():
        rel = preferred.relative_to(APCALC_ROOT).as_posix()
        return PUBLIC_BASE + quote(rel, safe="/._-")
    index = published_folder / "index.html"
    if index.is_file():
        rel = index.relative_to(APCALC_ROOT).as_posix()
        return PUBLIC_BASE + quote(rel, safe="/._-")
    html_files = sorted(published_folder.glob("*.html"), key=lambda x: x.name.casefold())
    if html_files:
        rel = html_files[0].relative_to(APCALC_ROOT).as_posix()
        return PUBLIC_BASE + quote(rel, safe="/._-")
    return None


def library_payload() -> dict:
    ensure_calc_tools_layout()
    categories = {}
    for category in LIBRARY_CATEGORIES:
        folder = LIBRARY_ROOT / category
        items = []
        for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
            if path.name.startswith('.') or path.name == "index.html" or path.name in {"_assets", "assets"}:
                continue
            if category == "warmups" and re.fullmatch(r"u\d+_\d+_warmups", path.name):
                continue
            stat = path.stat()
            editor_url = None
            if category == "exercises":
                m = re.fullmatch(r"u(\d+)_(\d+)_exercises", path.name)
                if m:
                    editor_url = f"/exercise_builder/?section={int(m.group(1))}.{int(m.group(2))}"
            elif category == "warmups":
                m = re.fullmatch(r"week_(\d+)_warmups", path.name)
                if m:
                    editor_url = f"/warmup_builder/editor.html?week={int(m.group(1))}"
            items.append({
                "name": path.name,
                "kind": "folder" if path.is_dir() else "file",
                "size": 0 if path.is_dir() else stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "published_url": _published_url_for_library_item(category, path),
                "editor_url": editor_url,
            })
        categories[category] = {"count": len(items), "items": items}
    return {"ok": True, "root": str(LIBRARY_ROOT), "categories": categories}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)

def ensure_warmup_layout() -> None:
    WARMUP_BUILDER_ROOT.mkdir(parents=True, exist_ok=True)
    WARMUP_PENDING_DIR.mkdir(parents=True, exist_ok=True)
    WARMUP_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    WARMUP_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    WARMUP_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)


def _load_warmup_history() -> dict:
    ensure_warmup_layout()
    if WARMUP_HISTORY_PATH.is_file():
        try:
            data = json.loads(WARMUP_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("schema_version", 1)
                data.setdefault("weeks", {})
                return data
        except Exception:
            pass
    return {"schema_version": 1, "weeks": {}}


def _save_warmup_history(data: dict) -> None:
    data["schema_version"] = 1
    data.setdefault("weeks", {})
    atomic_write_text(WARMUP_HISTORY_PATH, json.dumps(data, indent=2) + "\n")


def _has_warmup_slot(day: dict) -> bool:
    labels = {str(x or "").strip().lower() for x in (day.get("student_slots") or [])}
    return bool(labels & {"warmups", "warm up"})


def _title_for_code(state: dict, code: str) -> str:
    title = code
    for day in state.get("days", []):
        if section_code(str(day.get("section") or "")) == code:
            title = str(day.get("section") or code).strip()
    return title


def _eligible_taught_codes(state: dict, before_date: str | None) -> list[str]:
    seen = []
    for day in sorted(state.get("days", []), key=lambda d: str(d.get("date") or "")):
        ddate = str(day.get("date") or "")
        if before_date and ddate >= before_date:
            break
        code = section_code(str(day.get("section") or ""))
        if not code:
            continue
        try:
            if int(code.split(".")[0]) <= 0:
                continue
        except Exception:
            continue
        if code in seen:
            seen.remove(code)
        seen.append(code)
    return seen


def _usage_from_history(history: dict) -> dict[str, dict]:
    usage: dict[str, dict] = {}
    for week_key, entry in (history.get("weeks") or {}).items():
        try:
            week_num = int(week_key)
        except Exception:
            continue
        if not isinstance(entry, dict) or entry.get("status") != "published":
            continue
        plan = entry.get("plan") or {}
        for slot in plan.get("slots", []):
            code = str((slot.get("retain") or {}).get("section_code") or "")
            if not code:
                continue
            item = usage.setdefault(code, {"retain_count": 0, "last_retain_week": 0})
            item["retain_count"] += 1
            item["last_retain_week"] = max(item["last_retain_week"], week_num)
    return usage


def _select_retain_code(
    eligible: list[str],
    strengthen_code: str | None,
    slot_number: int,
    week_number: int,
    history: dict,
    used_this_week: set[str],
) -> str | None:
    options = [c for c in eligible if c and c != strengthen_code]
    if not options:
        return None
    usage = _usage_from_history(history)
    fresh = [c for c in options if c not in used_this_week] or options
    current_unit = strengthen_code.split(".")[0] if strengthen_code and "." in strengthen_code else ""
    same_unit = [c for c in fresh if c.split(".")[0] == current_unit]
    prior_unit = [c for c in fresh if c.split(".")[0] != current_unit]

    def recent(pool: list[str]) -> str | None:
        return pool[-1] if pool else None

    def neglected(pool: list[str]) -> str | None:
        if not pool:
            return None
        order = {code: i for i, code in enumerate(options)}
        return min(
            pool,
            key=lambda c: (
                int((usage.get(c) or {}).get("retain_count", 0)),
                int((usage.get(c) or {}).get("last_retain_week", 0)),
                -order.get(c, 0),
            ),
        )

    if slot_number == 1:
        choice = recent(fresh)
    elif slot_number == 2:
        choice = recent(same_unit) or recent(fresh)
    elif slot_number == 3:
        choice = recent(prior_unit) or recent(fresh)
    elif slot_number == 4:
        choice = neglected(fresh)
    else:
        choice = neglected(prior_unit) or neglected(fresh)
    if choice:
        used_this_week.add(choice)
    return choice


def _question_representation(slot_number: int, calculator_policy: str, role: str) -> str:
    if calculator_policy == "Calculator Permitted":
        choices = ["table/data", "numerical model", "graph", "table/data", "context/model"]
    else:
        choices = ["analytic", "conceptual", "graph", "analytic", "mixed reasoning"]
    idx = max(0, min(4, slot_number - 1))
    if role == "retain":
        idx = (idx + 2) % len(choices)
    return choices[idx]


def _backfill_warmup_history_from_library(history: dict) -> dict:
    """Register manually seeded weekly warmups so future recurrence can avoid over-repeating them."""
    changed = False
    weeks = history.setdefault("weeks", {})
    root = LIBRARY_ROOT / "warmups"
    if root.is_dir():
        for folder in root.glob("week_*_warmups"):
            manifest_path = folder / "WARMUP_RESULT.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                week_number = int(manifest.get("week_number") or 0)
            except Exception:
                continue
            if week_number < 1:
                continue
            plan = manifest.get("plan")
            if not isinstance(plan, dict):
                continue
            current = weeks.get(str(week_number)) if isinstance(weeks.get(str(week_number)), dict) else None
            if current and current.get("request_id") == manifest.get("request_id") and current.get("plan") == plan:
                continue
            weeks[str(week_number)] = {
                "status": "published",
                "published_at": datetime.fromtimestamp(manifest_path.stat().st_mtime).isoformat(timespec="seconds"),
                "request_id": manifest.get("request_id"),
                "plan": plan,
                "public_url": PUBLIC_BASE + f"warmups/week_{week_number:02d}_warmups/week_{week_number:02d}_warmups.html",
            }
            changed = True
    if changed:
        _save_warmup_history(history)
    return history


def build_warmup_plan(state: dict, week_number: int) -> dict:
    if week_number < 1:
        raise ValueError("Week number must be at least 1.")
    week_index = week_number - 1
    week_days = [d for d in state.get("days", []) if int(d.get("week_index", -1)) == week_index]
    if not week_days:
        raise ValueError(f"Week {week_number} is not present in the Planner.")
    week_days.sort(key=lambda d: str(d.get("date") or ""))
    history = _backfill_warmup_history_from_library(_load_warmup_history())
    scheduled = [
        d for d in week_days
        if str(d.get("control") or "Regular Day") not in BLOCKED_CONTROLS and _has_warmup_slot(d)
    ][:5]
    if not scheduled:
        scheduled = [
            d for d in week_days
            if str(d.get("control") or "Regular Day") not in BLOCKED_CONTROLS
            and section_code(str(d.get("section") or ""))
        ][:5]

    slots = []
    used_retain: set[str] = set()
    last_recent_code = None
    last_recent_title = None
    week_end = str(week_days[-1].get("date") or "")

    for number in range(1, 6):
        day = scheduled[number - 1] if number <= len(scheduled) else None
        date_value = str(day.get("date") or "") if day else None
        display_date = str(day.get("display_date") or "") if day else "Flex"
        # HARD RULE: the warmup never previews today's lesson. Only sections taught
        # strictly before the warmup date are eligible for Recent Review or Spiral Review.
        before_date = date_value or (week_end + "Z" if week_end else None)
        eligible = _eligible_taught_codes(state, before_date)
        recent_code = eligible[-1] if eligible else last_recent_code
        recent_title = _title_for_code(state, recent_code) if recent_code else (last_recent_title or "Previously taught course skill")
        if recent_code:
            last_recent_code = recent_code
            last_recent_title = recent_title
        retain_code = _select_retain_code(
            eligible,
            recent_code,
            number,
            week_number,
            history,
            used_retain,
        )
        retain_title = _title_for_code(state, retain_code) if retain_code else "Earlier prerequisite skill"
        recent_calc = "No Calculator" if number % 2 == 1 else "Calculator Permitted"
        retain_calc = "Calculator Permitted" if recent_calc == "No Calculator" else "No Calculator"
        slots.append({
            "number": number,
            "date": date_value,
            "display_date": display_date,
            "source": "planner" if day else "flex",
            "scheduled_lesson": str(day.get("section") or "").strip() if day else None,
            "strengthen": {
                "section_code": recent_code,
                "section_title": recent_title,
                "calculator_policy": recent_calc,
                "representation": _question_representation(number, recent_calc, "strengthen"),
            },
            "retain": {
                "section_code": retain_code,
                "section_title": retain_title,
                "calculator_policy": retain_calc,
                "representation": _question_representation(number, retain_calc, "retain"),
            },
        })

    return {
        "week_number": week_number,
        "week_index": week_index,
        "school_year": state.get("school_year"),
        "week_start": str(week_days[0].get("date") or ""),
        "week_end": str(week_days[-1].get("date") or ""),
        "scheduled_instructional_warmups": len(scheduled),
        "slots": slots,
        "rule": "Warmups look backward only. Question 1 reviews the most recently taught skill before that day. Question 2 spirals an older taught skill. Never use the lesson scheduled for that day until a later warmup. One question is No Calculator and one is Calculator Permitted.",
    }


def _week_status_payload(state: dict) -> dict:
    history = _backfill_warmup_history_from_library(_load_warmup_history())
    weeks = []
    week_indexes = sorted({int(d.get("week_index", 0)) for d in state.get("days", [])})
    for wi in week_indexes:
        days = sorted([d for d in state.get("days", []) if int(d.get("week_index", -1)) == wi], key=lambda d: str(d.get("date") or ""))
        if not days:
            continue
        num = wi + 1
        local_html = LIBRARY_ROOT / "warmups" / f"week_{num:02d}_warmups" / f"week_{num:02d}_warmups.html"
        public_html = APCALC_ROOT / "warmups" / f"week_{num:02d}_warmups" / f"week_{num:02d}_warmups.html"
        hist = (history.get("weeks") or {}).get(str(num), {})
        weeks.append({
            "week_number": num,
            "week_index": wi,
            "start": str(days[0].get("display_date") or days[0].get("date") or ""),
            "end": str(days[-1].get("display_date") or days[-1].get("date") or ""),
            "planner_warmup_days": sum(1 for d in days if _has_warmup_slot(d) and str(d.get("control") or "Regular Day") not in BLOCKED_CONTROLS),
            "local": local_html.is_file(),
            "published": public_html.is_file(),
            "history_status": hist.get("status") if isinstance(hist, dict) else None,
            "public_url": PUBLIC_BASE + f"warmups/week_{num:02d}_warmups/week_{num:02d}_warmups.html" if public_html.is_file() else None,
        })
    return {"ok": True, "current_week_number": int(state.get("current_week_index", 0)) + 1, "weeks": weeks}


def _source_files_for_section(code: str | None) -> list[tuple[str, Path]]:
    if not code or "." not in code:
        return []
    unit_s, sec_s = code.split(".", 1)
    try:
        tag = f"{int(unit_s)}_{int(sec_s)}"
    except Exception:
        return []
    candidates = [
        ("notes.html", LIBRARY_ROOT / "notes" / f"u{tag}_notes" / f"u{tag}_notes.html"),
        ("notes.html", APCALC_ROOT / "notes" / f"u{tag}_notes" / f"u{tag}_notes.html"),
        ("exercises.html", LIBRARY_ROOT / "exercises" / f"u{tag}_exercises" / f"u{tag}_exercises.html"),
        ("exercises.html", APCALC_ROOT / "exercises" / f"u{tag}_exercises" / f"u{tag}_exercises.html"),
        ("practice.html", APCALC_ROOT / "practice_sets" / f"practice_sets_{tag}" / f"practice_set_{tag}.html"),
        ("activity.html", APCALC_ROOT / "activities" / f"u{tag}_act1" / f"u{tag}_act1.html"),
    ]
    found = []
    used_names = set()
    for name, path in candidates:
        if name in used_names or not path.is_file():
            continue
        used_names.add(name)
        found.append((name, path))
    return found


def _warmup_request_readme(request: dict) -> str:
    week = request["week_number"]
    return f"""AP Calculus Warmup Request - Week {week}
========================================

This request was generated from the local AP Calculus Planner.

GOAL
----
Create five numbered warmups (1-5). Each warmup has exactly two original AP-style multiple-choice questions with choices A-E.

PURPOSE
-------
Question 1 = RECENT REVIEW of the most recently taught skill before that warmup day.
Question 2 = SPIRAL REVIEW of an older skill that has already been taught.
HARD RULE: never preview or teach the lesson scheduled for that day. The scheduled lesson is a boundary, not a source.
Each warmup has one No Calculator question and one Calculator Permitted question exactly as specified in the plan.

QUALITY RULES
-------------
- Use authentic AP Calculus AB wording and mathematical rigor, but write original questions.
- Do not copy copyrighted source problems verbatim.
- Use the bundled source files only to understand scope, terminology, representations, and prerequisite level.
- Use only skills/I Can statements from sections taught BEFORE that warmup day.
- Rotate representations as requested: analytic, graph, table/data, context/model, conceptual.
- Calculator-permitted questions should naturally fit calculator use.
- No Calculator questions must be solvable exactly without numerical approximation.
- Do NOT include answers, answer keys, explanations, or hidden answer data anywhere.

RETURN CONTRACT
---------------
Return ONE ZIP named approximately apcalc_week_{week:02d}_warmups_RESULT.zip containing at its root:
  WARMUP_RESULT.json
  week_{week:02d}_warmups.json
  week_{week:02d}_warmups.html

The JSON file is the teacher-editable source. It must contain week_number, title, intro, settings, and warmups. Each question stores role, topic, calculator, active_instance, and one or more instances with a stem plus exactly five choices. Do NOT include answers in this JSON.

WARMUP_RESULT.json must be:
{{
  "schema_version": 1,
  "package_type": "apcalc_warmup_result",
  "week_number": {week},
  "request_id": "{request['request_id']}",
  "teacher_config": "week_{week:02d}_warmups.json",
  "student_html": "week_{week:02d}_warmups.html"
}}

The importer regenerates the student HTML from the teacher config so the public page never contains teacher controls or answers.
Do not include student names or private student data.
"""


def create_warmup_request_zip(state: dict, week_number: int) -> tuple[Path, dict]:
    ensure_warmup_layout()
    plan = build_warmup_plan(state, week_number)
    request_id = f"apcalc-week-{week_number:02d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    request = {
        "schema_version": 1,
        "package_type": "apcalc_warmup_request",
        "request_id": request_id,
        "course": "AP Calculus AB",
        "week_number": week_number,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "plan": plan,
        "requirements": {
            "warmup_count": 5,
            "questions_per_warmup": 2,
            "question_1_role": "Recent Review",
            "question_2_role": "Spiral Review",
            "answer_choices": "A-E",
            "format": "AP-style multiple choice",
            "one_no_calculator_and_one_calculator_permitted_per_warmup": True,
            "student_answers": False,
            "print_share_notability": True,
        },
        "result_contract": {
            "schema_version": 1,
            "package_type": "apcalc_warmup_result",
            "teacher_config": f"week_{week_number:02d}_warmups.json",
            "student_html": f"week_{week_number:02d}_warmups.html",
        },
    }
    pending = WARMUP_PENDING_DIR / f"week_{week_number:02d}.json"
    atomic_write_text(pending, json.dumps(request, indent=2) + "\n")
    target = WARMUP_REQUESTS_DIR / f"apcalc_week_{week_number:02d}_warmups_REQUEST.zip"
    unique_codes = []
    for slot in plan["slots"]:
        for role in ("strengthen", "retain"):
            code = str((slot.get(role) or {}).get("section_code") or "")
            if code and code not in unique_codes:
                unique_codes.append(code)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("WARMUP_REQUEST.json", json.dumps(request, indent=2) + "\n")
        zf.writestr("README.txt", _warmup_request_readme(request))
        raw_days = [d for d in state.get("days", []) if int(d.get("week_index", -1)) == week_number - 1]
        zf.writestr("planner_week.json", json.dumps(raw_days, indent=2) + "\n")
        source_manifest = []
        for code in unique_codes:
            for label, path in _source_files_for_section(code):
                arc = f"sources/{code.replace('.', '_')}/{label}"
                zf.write(path, arc)
                source_manifest.append({"section_code": code, "file": arc, "source_path": str(path)})
        zf.writestr("sources/SOURCE_MANIFEST.json", json.dumps(source_manifest, indent=2) + "\n")
    return target, request


def _safe_zip_member(name: str) -> Path:
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe ZIP path: {name}")
    return rel


def import_warmup_result(raw: bytes, expected_week: int | None = None) -> dict:
    if not raw or len(raw) > WARMUP_RESULT_MAX_BYTES:
        raise ValueError("Warmup result ZIP is empty or too large.")
    ensure_warmup_layout()
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded result is not a valid ZIP file.") from exc
    names = zf.namelist()
    if "WARMUP_RESULT.json" not in names:
        raise ValueError("WARMUP_RESULT.json is missing from the result ZIP.")
    total = sum(info.file_size for info in zf.infolist())
    if total > WARMUP_RESULT_MAX_BYTES:
        raise ValueError("Warmup result expands beyond the allowed size.")
    manifest = json.loads(zf.read("WARMUP_RESULT.json").decode("utf-8"))
    if manifest.get("package_type") != "apcalc_warmup_result" or int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported warmup result contract.")
    week_number = int(manifest.get("week_number") or 0)
    if week_number < 1 or (expected_week and week_number != expected_week):
        raise ValueError("Warmup result week does not match the selected week.")
    config_name = str(manifest.get("teacher_config") or "").strip()
    if not config_name:
        raise ValueError("teacher_config is missing from WARMUP_RESULT.json.")
    config_rel = _safe_zip_member(config_name)
    if config_rel.as_posix() not in names:
        raise ValueError(f"Teacher warmup config is missing: {config_name}")
    teacher_config = warmup_editor.validate(json.loads(zf.read(config_rel.as_posix()).decode("utf-8")))
    if int(teacher_config.get("week_number") or 0) != week_number:
        raise ValueError("Teacher warmup config week does not match the result manifest.")
    # Student HTML is regenerated from the teacher config. A supplied HTML file is ignored.
    html_name = f"week_{week_number:02d}_warmups.html"
    html_text = warmup_editor.render(teacher_config)
    pending_path = WARMUP_PENDING_DIR / f"week_{week_number:02d}.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8")) if pending_path.is_file() else None
    request_id = str(manifest.get("request_id") or "")
    if pending and request_id and request_id != str(pending.get("request_id") or ""):
        raise ValueError("Result request_id does not match the latest request for this week.")

    target = LIBRARY_ROOT / "warmups" / f"week_{week_number:02d}_warmups"
    if target.exists():
        backup = WARMUP_BACKUP_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_week_{week_number:02d}_warmups"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(target, backup)
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    # Preserve optional student assets, but never publish the teacher-editable config.
    for info in zf.infolist():
        if info.is_dir() or info.filename in {"WARMUP_RESULT.json", config_rel.as_posix(), str(manifest.get("student_html") or "")} :
            continue
        rel = _safe_zip_member(info.filename)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(info.filename))
    atomic_write_text(target / html_name, html_text)
    atomic_write_text(WARMUP_CONFIG_ROOT / f"week_{week_number:02d}.json", json.dumps(teacher_config, indent=2) + "\n")
    atomic_write_text(target / "WARMUP_RESULT.json", json.dumps(manifest, indent=2) + "\n")

    published = APCALC_ROOT / "warmups" / target.name
    if published.exists():
        shutil.rmtree(published)
    shutil.copytree(target, published)

    history = _load_warmup_history()
    plan = (pending or {}).get("plan") if isinstance(pending, dict) else None
    if not plan:
        plan = build_warmup_plan(get_state(), week_number)
    history.setdefault("weeks", {})[str(week_number)] = {
        "status": "published",
        "published_at": datetime.now().isoformat(timespec="seconds"),
        "request_id": request_id or (pending or {}).get("request_id"),
        "plan": plan,
        "public_url": PUBLIC_BASE + f"warmups/week_{week_number:02d}_warmups/week_{week_number:02d}_warmups.html",
    }
    _save_warmup_history(history)
    state = get_state()
    atomic_write_text(AGENDA_PATH, build_student_agenda(state))
    return {
        "ok": True,
        "week_number": week_number,
        "local_path": str(target),
        "published_path": str(published),
        "public_url": history["weeks"][str(week_number)]["public_url"],
        "message": f"Week {week_number} Warmups installed locally, published into the apcalc repository, and the Student Agenda was refreshed. Use GitHub Sync when ready.",
    }


def weekly_warmup_url(day: dict) -> str | None:
    """Public student URL used by generated agendas."""
    try:
        week_number = int(day.get("week_index", -1)) + 1
    except Exception:
        return None
    if week_number < 1:
        return None
    relative = f"warmups/week_{week_number:02d}_warmups/week_{week_number:02d}_warmups.html"
    return rel_if_exists(relative)


def planner_weekly_warmup_url(day: dict) -> str | None:
    """Local Planner preview link. Prefer the local Library so the link works before GitHub sync."""
    try:
        week_number = int(day.get("week_index", -1)) + 1
    except Exception:
        return None
    if week_number < 1:
        return None
    folder = f"week_{week_number:02d}_warmups"
    name = f"{folder}.html"
    local = LIBRARY_ROOT / "warmups" / folder / name
    if local.is_file():
        return f"/library/warmups/{folder}/{name}"
    return weekly_warmup_url(day)


def ensure_exercise_builder_layout() -> None:
    EXERCISE_BUILDER_ROOT.mkdir(parents=True, exist_ok=True)
    EXERCISE_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)


def _normalize_exercise_section(value: str) -> str:
    value = str(value or "").strip()
    if not re.fullmatch(r"\d+\.\d+", value):
        raise ValueError("Exercise section must look like 3.1.")
    return value


def _exercise_config_path(section: str) -> Path:
    section = _normalize_exercise_section(section)
    return EXERCISE_CONFIG_ROOT / f"u{section.replace('.', '_')}.json"


def _exercise_paths(section: str) -> tuple[Path, Path, str]:
    section = _normalize_exercise_section(section)
    tag = section.replace('.', '_')
    rel = f"exercises/u{tag}_exercises/u{tag}_exercises.html"
    return LIBRARY_ROOT / rel, APCALC_ROOT / rel, PUBLIC_BASE + rel


def _load_exercise_config(section: str) -> dict:
    path = _exercise_config_path(section)
    if not path.is_file():
        raise FileNotFoundError(f"No local exercise builder config for {section}.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Exercise config is invalid.")
    data.setdefault("settings", {})
    data.setdefault("printables", [])
    data.setdefault("questions", [])
    return data


def _validate_exercise_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Exercise config is required.")
    section = _normalize_exercise_section(config.get("section"))
    config = deepcopy(config)
    config["section"] = section
    config["schema_version"] = 1
    config["title"] = str(config.get("title") or f"Section {section}").strip()
    settings = config.setdefault("settings", {})
    if settings.get("spacing") not in {"compact", "normal", "roomy"}:
        settings["spacing"] = "normal"
    if settings.get("margins") not in {"narrow", "normal", "wide"}:
        settings["margins"] = "normal"
    try:
        settings["workspace"] = max(0.0, min(5.0, float(settings.get("workspace", 1.0))))
    except Exception:
        settings["workspace"] = 1.0
    settings["show_title"] = bool(settings.get("show_title", True))
    clean_printables = []
    for raw in config.get("printables") or []:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "Printable").strip() or "Printable"
        url = str(raw.get("url") or "").strip()
        if url:
            clean_printables.append({"label": label, "url": url})
    config["printables"] = clean_printables
    clean_questions = []
    for idx, raw in enumerate(config.get("questions") or [], 1):
        if not isinstance(raw, dict):
            continue
        instances = []
        for j, inst in enumerate(raw.get("instances") or []):
            if not isinstance(inst, dict):
                continue
            instances.append({"label": str(inst.get("label") or f"Instance {j+1}"), "html": str(inst.get("html") or "")})
        if not instances:
            instances = [{"label": "Original", "html": ""}]
        try:
            active = int(raw.get("active_instance", 0))
        except Exception:
            active = 0
        active = max(0, min(active, len(instances) - 1))
        try:
            workspace = max(0.0, min(5.0, float(raw.get("workspace", settings["workspace"]))))
        except Exception:
            workspace = settings["workspace"]
        clean_questions.append({
            "id": str(raw.get("id") or f"q{idx}"),
            "stage": str(raw.get("stage") or "").strip(),
            "visible": bool(raw.get("visible", True)),
            "workspace": workspace,
            "page_break": bool(raw.get("page_break", False)),
            "active_instance": active,
            "instances": instances,
        })
    config["questions"] = clean_questions
    return config


def render_exercise_student_page(config: dict) -> str:
    config = _validate_exercise_config(config)
    section = config["section"]
    settings = config["settings"]
    title = f"{section} · {config['title']} — Exercises"
    body_cls = f"spacing-{settings['spacing']} margin-{settings['margins']}"
    printable_links = []
    for item in config.get("printables", []):
        printable_links.append(f'<a href="{escape(item["url"], quote=True)}" target="_blank" rel="noopener">{escape(item["label"])}</a>')
    printables = f'<div class="printables"><strong>Printables:</strong> {" ".join(printable_links)}</div>' if printable_links else ""
    blocks = []
    number = 0
    for q in config.get("questions", []):
        if not q.get("visible", True):
            continue
        instances = q.get("instances") or []
        active = max(0, min(int(q.get("active_instance", 0)), len(instances) - 1)) if instances else 0
        prompt = str(instances[active].get("html") or "") if instances else ""
        number += 1
        stage = f'<div class="stage">{escape(str(q.get("stage") or ""))}</div>' if q.get("stage") else ""
        page_break = " page-break" if q.get("page_break") else ""
        blocks.append(f'<section class="question{page_break}">{stage}<div class="qtext"><span class="qnum">{number}.</span>{prompt}</div><div class="workspace" style="--local-ws:{float(q.get("workspace", settings["workspace"])):g}in"></div></section>')
    title_display = "block" if settings.get("show_title", True) else "none"
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AP Calculus AB · {escape(section)} · Exercises</title><script>window.MathJax={{tex:{{inlineMath:[["\\\\(","\\\\)"],["$","$"]],displayMath:[["\\\\[","\\\\]"],["$$","$$"]],processEscapes:true}}}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script><link rel="stylesheet" href="../assets/student_exercises.css"></head><body class="{body_cls}" style="--ws:{settings['workspace']:g}in"><div class="screen-tools no-print"><button type="button" onclick="window.print()">Print / Share PDF</button><span>On iPad: use the print preview Share button to send to Notability.</span></div><main class="preview"><article class="sheet"><header class="doc-title" style="display:{title_display}"><div class="kicker">AP Calculus AB · Chapter {escape(section.split('.')[0])}</div><h1>{escape(title)}</h1></header><div class="student-line"><div>Name <span></span></div><div>Date <span></span></div><div>Period <span></span></div></div><p class="directions">Show work and use correct notation. Follow any calculator directions printed with a question.</p>{printables}{''.join(blocks)}</article></main></body></html>'''


def save_exercise_config(config: dict, publish: bool = False) -> dict:
    ensure_exercise_builder_layout()
    config = _validate_exercise_config(config)
    section = config["section"]
    cfg_path = _exercise_config_path(section)
    atomic_write_text(cfg_path, json.dumps(config, indent=2) + "\n")
    local_html, public_html, public_url = _exercise_paths(section)
    atomic_write_text(local_html, render_exercise_student_page(config))
    if publish:
        public_html.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_html, public_html)
        src_assets = LIBRARY_ROOT / "exercises" / "assets"
        if src_assets.is_dir():
            dst_assets = APCALC_ROOT / "exercises" / "assets"
            dst_assets.mkdir(parents=True, exist_ok=True)
            for item in src_assets.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst_assets / item.name)
        atomic_write_text(AGENDA_PATH, build_student_agenda(get_state()))
    return {
        "ok": True,
        "config": config,
        "local_path": str(local_html),
        "public_url": public_url,
        "published": publish,
        "message": (f"Published {section} to the student URL and refreshed the agenda. Use GitHub Sync when ready." if publish else f"Saved {section} locally. The public student page was not changed."),
    }


def exercise_sections_payload() -> dict:
    ensure_exercise_builder_layout()
    sections = []
    for path in sorted(EXERCISE_CONFIG_ROOT.glob("u*.json"), key=lambda p: p.name):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            section = _normalize_exercise_section(cfg.get("section"))
        except Exception:
            continue
        sections.append({"section": section, "title": str(cfg.get("title") or section), "public_url": _exercise_paths(section)[2]})
    return {"ok": True, "sections": sections}


CHAPTER3_MIGRATION_ID = "chapter3-rebuild-20261005-v1"

CHAPTER3_SCHEDULE = {
    "2026-10-05": {"section": "", "day_number": 0, "slots": ["", "", "", "", ""]},
    "2026-10-06": {"section": "3.1: Derivative of a Function", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-07": {"section": "3.1: Derivative of a Function", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-08": {"section": "3.2: Differentiability", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-09": {"section": "3.2: Differentiability", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-12": {"section": "3.3: Rules for Differentiation", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-13": {"section": "3.3: Rules for Differentiation", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-14": {"section": "3.3: Rules for Differentiation", "day_number": 3, "slots": ["Warmups", "Lab", "Exercises", "Sets", "Quick Check"]},
    "2026-10-15": {"section": "3.4: Velocity & Rates", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-16": {"section": "3.4: Velocity & Rates", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-19": {"section": "3.5: Trig Derivatives", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-20": {"section": "3.5: Trig Derivatives", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-21": {"section": "3.6: Chain Rule", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-22": {"section": "3.6: Chain Rule", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-23": {"section": "3.6: Chain Rule", "day_number": 3, "slots": ["Warmups", "Lab", "Exercises", "Sets", "Quick Check"]},
    "2026-10-26": {"section": "3.7: Implicit Differentiation", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-27": {"section": "3.7: Implicit Differentiation", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-28": {"section": "3.8: Inverse Trig Derivatives", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-10-29": {"section": "3.8: Inverse Trig Derivatives", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-10-30": {"section": "3.9: Exponential & Log Derivatives", "day_number": 1, "slots": ["Warmups", "Notes", "Exercises", "", ""]},
    "2026-11-02": {"section": "3.9: Exponential & Log Derivatives", "day_number": 2, "slots": ["Warmups", "Explore", "Exercises", "Sets", ""]},
    "2026-11-03": {"section": "3.9: Exponential & Log Derivatives", "day_number": 3, "slots": ["Warmups", "Lab", "Exercises", "Sets", "Quick Check"]},
    "2026-11-04": {"section": "", "day_number": 0, "slots": ["Custom Item", "", "", "", ""], "custom": "Unit 3 Review"},
    "2026-11-05": {"section": "", "day_number": 0, "slots": ["Custom Item", "", "", "", ""], "custom": "AP Classroom / Unit 3 Review"},
    "2026-11-06": {"section": "Unit 3 Test", "day_number": 0, "slots": ["", "", "", "", ""], "assessment_unit": 3},
}

def apply_chapter3_rebuild(state: dict) -> bool:
    migrations = state.get("migrations")
    if not isinstance(migrations, list):
        migrations = []
        state["migrations"] = migrations
    if CHAPTER3_MIGRATION_ID in migrations:
        return False
    by_date = {str(d.get("date")): d for d in state.get("days", [])}
    changed = False
    for iso, spec in CHAPTER3_SCHEDULE.items():
        day = by_date.get(iso)
        if not day:
            continue
        day["control"] = "Regular Day"
        day["section"] = spec.get("section", "")
        day["day_number"] = int(spec.get("day_number", 0) or 0)
        day["student_slots"] = list(spec.get("slots", ["", "", "", "", ""]))
        day["custom_slots"] = [{"label": "", "url": ""} for _ in range(5)]
        if spec.get("custom"):
            day["custom_slots"][0] = {"label": spec["custom"], "url": ""}
        day["assessment_unit"] = int(spec.get("assessment_unit", 0) or 0)
        day["locked_past"] = False
        day["shift_applied"] = False
        changed = True
    # Week beginning 10/5 is the manually selected current week.
    for day in state.get("days", []):
        if str(day.get("date")) == "2026-10-05":
            state["current_week_index"] = int(day.get("week_index", state.get("current_week_index", 0)))
            break
    migrations.append(CHAPTER3_MIGRATION_ID)
    return changed

def publish_local_curriculum_to_repo() -> list[str]:
    ensure_calc_tools_layout()
    written = []
    for category in LIBRARY_CATEGORIES:
        src_root = LIBRARY_ROOT / category
        dst_root = APCALC_ROOT / category
        if not src_root.is_dir():
            continue
        for src in src_root.rglob("*"):
            if not src.is_file() or src.name.startswith("."):
                continue
            rel = src.relative_to(src_root)
            if "_assets" in rel.parts:
                continue
            if category == "warmups" and rel.parts and re.fullmatch(r"u\d+_\d+_warmups", rel.parts[0]):
                continue
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.read_bytes() == src.read_bytes():
                continue
            shutil.copy2(src, dst)
            written.append(str(dst.relative_to(GITHUB_ROOT)))
    return written

def _merge_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in list(src.iterdir()):
        _merge_tree(child, dst / child.name)
    try:
        src.rmdir()
    except OSError:
        pass


def _rewrite_secure_assessment_assets(path: Path) -> None:
    """Keep secure assessment HTML private while loading non-secure shared CSS/images publicly."""
    if not path.is_file() or path.suffix.lower() != ".html":
        return
    text = path.read_text(encoding="utf-8")
    rewritten = re.sub(
        r'(?P<attr>\b(?:src|href))=(?P<quote>["\'])\.\./\.\./',
        lambda m: f'{m.group("attr")}={m.group("quote")}{PUBLIC_BASE}',
        text,
    )
    if rewritten != text:
        atomic_write_text(path, rewritten)


def _move_tree_secure(src: Path, dst: Path) -> None:
    """Move one public secure-artifact tree into teacher_shared without overwriting existing private files."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in list(src.iterdir()):
        target = dst / child.name
        if child.is_dir():
            _move_tree_secure(child, target)
            try:
                child.rmdir()
            except OSError:
                pass
            continue
        if target.exists():
            if target.is_file() and target.read_bytes() == child.read_bytes():
                child.unlink()
                continue
            stem, suffix = target.stem, target.suffix
            index = 2
            while True:
                alternate = target.with_name(f"{stem}_migrated_{index}{suffix}")
                if not alternate.exists():
                    target = alternate
                    break
                index += 1
        child.rename(target)
    try:
        src.rmdir()
    except OSError:
        pass


def migrate_secure_summatives() -> None:
    """Move any AP Calculus AB unit summatives out of the public repo into teacher_shared/calc/assessments."""
    secure_root = SHARED_CALC_ROOT / "assessments"
    secure_root.mkdir(parents=True, exist_ok=True)
    # assessment_plans is the current public folder name. The legacy name is retained only for one-time cleanup.
    for public_root in (APCALC_ROOT / "assessment_plans", APCALC_ROOT / "assessments_zxrtjp"):
        if not public_root.is_dir():
            continue
        for unit in range(1, 9):
            src = public_root / f"unit{unit}_summative"
            if not src.exists():
                continue
            dst = secure_root / f"unit{unit}_summative"
            _move_tree_secure(src, dst)
            for html_path in dst.rglob("*.html"):
                _rewrite_secure_assessment_assets(html_path)


def _secure_assessments(unit: int) -> str:
    folder = SHARED_CALC_ROOT / "assessments" / f"unit{unit}_summative"
    if not folder.is_dir():
        return f'<p class="placeholder">No approved Unit {unit} secure assessments are listed yet.</p>'
    files = sorted(folder.glob("*.html"), key=lambda p: p.name.casefold())
    if not files:
        return f'<p class="placeholder">No approved Unit {unit} secure assessments are listed yet.</p>'
    rows = []
    for path in files:
        label = path.stem.replace("_", " ").title()
        href = f'../../assessments/unit{unit}_summative/{quote(path.name)}'
        rows.append(f'<li><a href="{href}" target="_blank" rel="noopener">{escape(label)}</a></li>')
    return '<ul>' + ''.join(rows) + '</ul>'


def build_teacher_shared_home() -> str:
    return """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Teacher Shared</title><style>body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f4f6f8;color:#173f6d}.wrap{max-width:900px;margin:48px auto;padding:0 20px}h1{background:#173f6d;color:#fff;padding:20px;border-radius:14px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}.card{display:block;background:#fff;border:2px solid #d9c06d;border-radius:14px;padding:24px;text-decoration:none;color:#173f6d;font-size:1.2rem;font-weight:800}.card:hover{background:#fff8df}.muted{color:#667085;font-size:.9rem}</style></head><body><div class=\"wrap\"><h1>Teacher Shared</h1><div class=\"cards\"><a class=\"card\" href=\"algebra/index.html\">Algebra 1</a><a class=\"card\" href=\"physics/index.html\">Physics</a><a class=\"card\" href=\"calc/index.html\">Calculus</a></div><p class=\"muted\">Private shared teacher resources. Course folders are populated by the local teacher tools.</p></div></body></html>"""


def build_course_placeholder(title: str) -> str:
    return f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{escape(title)} Teacher Shared</title><style>body{{font-family:Arial,Helvetica,sans-serif;margin:40px;color:#173f6d}}a{{color:#173f6d}}</style></head><body><h1>{escape(title)} Teacher Shared</h1><p>This course folder is ready. Its planner will publish here when connected.</p><p><a href=\"../index.html\">Teacher Shared home</a></p></body></html>"""


def build_calc_library_index() -> str:
    units = "".join(
        f'<a class="unit" href="unit{u}/index.html">Unit {u}</a>' for u in range(1, 9)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AP Calculus AB Teacher Library</title><style>body{{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f4f6f8;color:#173f6d}}.wrap{{max-width:1000px;margin:40px auto;padding:0 18px}}h1{{background:#173f6d;color:white;padding:18px 20px;border-radius:14px}}.units{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}}.unit{{display:block;background:white;border:2px solid #d9c06d;border-radius:12px;padding:18px;text-align:center;text-decoration:none;color:#173f6d;font-weight:900}}.unit:hover{{background:#fff8df}}.back{{display:inline-block;margin-top:18px;color:#173f6d;font-weight:800;text-decoration:none;border:1px solid #d4b34e;border-radius:999px;padding:8px 13px;background:#fff}}</style></head><body><div class="wrap"><h1>AP Calculus AB Teacher Library</h1><p>Choose a unit. Each unit page contains approved assessment collections plus direct links to the unit assessment plan and bank.</p><div class="units">{units}</div><a class="back" href="../index.html">Teacher Agenda</a></div></body></html>"""


def _approved_quick_checks(unit: int) -> str:
    entries = []
    if QUICK_CHECK_LIBRARY.is_file():
        try:
            data = json.loads(QUICK_CHECK_LIBRARY.read_text(encoding="utf-8"))
            entries = [e for e in data.get("entries", []) if int(e.get("unit", 0) or 0) == unit]
        except Exception:
            entries = []
    if not entries:
        return f'<p class="placeholder">No approved Unit {unit} Quick Checks are registered yet.</p>'
    entries.sort(key=lambda e: (str(e.get("section", "")), str(e.get("run_id", ""))))
    rows = []
    for e in entries:
        section = escape(str(e.get("section") or f"Unit {unit}"))
        html_rel = str(e.get("html") or "").lstrip("/")
        pdf_rel = str(e.get("pdf") or "").lstrip("/")
        links = []
        if html_rel:
            links.append(f'<a href="{PUBLIC_BASE}quick_check____htq5855/{escape(html_rel, quote=True)}" target="_blank" rel="noopener">HTML</a>')
        if pdf_rel:
            links.append(f'<a href="{PUBLIC_BASE}quick_check____htq5855/{escape(pdf_rel, quote=True)}" target="_blank" rel="noopener">PDF</a>')
        rows.append(f'<li><strong>{section} Quick Check</strong> — {" · ".join(links) if links else "registered"}</li>')
    return '<ul>' + ''.join(rows) + '</ul>'


def _performance_tasks(unit: int) -> str:
    rows = []
    for section in range(1, 6):
        tag = f"{unit}_{section}"
        student = rel_if_exists(f"performance_task___5134zrt/u{tag}_performance_task/u{tag}_performance_task.html")
        teacher = rel_if_exists(f"performance_task___5134zrt/u{tag}_performance_task/u{tag}_performance_task_teacher_guide.html")
        if not student and not teacher:
            continue
        links = []
        if student:
            links.append(f'<a href="{escape(student, quote=True)}" target="_blank" rel="noopener">Performance Task</a>')
        if teacher:
            links.append(f'<a href="{escape(teacher, quote=True)}" target="_blank" rel="noopener">Teacher Guide</a>')
        rows.append(f'<li><strong>{unit}.{section}</strong> — {" · ".join(links)}</li>')
    if not rows:
        return f'<p class="placeholder">No Unit {unit} performance tasks are published yet.</p>'
    return '<ul>' + ''.join(rows) + '</ul>'


def build_calc_unit_library(unit: int) -> str:
    unit_nav = "".join(
        f'<a class="unitnav{" active" if u == unit else ""}" href="../unit{u}/index.html">Unit {u}</a>'
        for u in range(1, 9)
    )
    assessment_plan = f'https://tnezki.github.io/apcalc/assessment_plans/unit{unit}_assessment_plan/unit{unit}_assessment_plan.html'
    bank_rel = f"banks/unit{unit}/unit{unit}.html"
    bank_public = rel_if_exists(bank_rel)
    bank_target = bank_public or "bank.html"
    direct_nav = ''.join([
        f'<a class="directnav" href="{assessment_plan}" target="_blank" rel="noopener">Unit {unit} Assessment Plan</a>',
        f'<a class="directnav" href="{escape(bank_target, quote=True)}" target="_blank" rel="noopener">Unit {unit} Bank</a>',
    ])
    quick_checks = _approved_quick_checks(unit)
    assessments = _secure_assessments(unit)
    perf = _performance_tasks(unit)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AP Calculus AB Unit {unit} Teacher Library</title><style>body{{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f4f6f8;color:#173f6d}}.wrap{{max-width:1080px;margin:30px auto;padding:0 18px}}h1{{background:#173f6d;color:#fff;padding:18px 20px;border-radius:14px;margin-bottom:12px}}.unitbar,.directbar{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}}.unitnav,.directnav{{text-decoration:none;color:#173f6d;background:#fff;border:1px solid #d4b34e;border-radius:999px;padding:8px 12px;font-weight:800}}.unitnav.active{{background:#e4bf3f}}.directnav:hover,.unitnav:hover{{background:#fff8df}}.section{{background:#fff;border:1px solid #d9dee5;border-radius:14px;padding:18px;margin:14px 0;scroll-margin-top:16px}}.section h2{{margin:0 0 8px}}.section a{{color:#173f6d;font-weight:800}}.placeholder{{color:#667085}}ul{{margin:8px 0 0;padding-left:24px}}li{{margin:7px 0}}</style></head><body><div class="wrap"><h1>AP Calculus AB Unit {unit} Teacher Library</h1><div class="unitbar">{unit_nav}</div><div class="directbar">{direct_nav}</div><section id="quick-checks" class="section"><h2>Approved Quick Checks</h2>{quick_checks}</section><section id="assessments" class="section"><h2>Approved Assessments</h2>{assessments}</section><section id="performance-tasks" class="section"><h2>Performance Tasks</h2>{perf}</section></div></body></html>"""

def ensure_teacher_shared_layout() -> None:
    migrate_secure_summatives()
    SHARED_ROOT.mkdir(parents=True, exist_ok=True)
    SHARED_CALC_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_text(SHARED_ROOT / "index.html", build_teacher_shared_home())
    for folder, title in (("algebra", "Algebra 1"), ("physics", "Physics")):
        target = SHARED_ROOT / folder / "index.html"
        if not target.is_file():
            atomic_write_text(target, build_course_placeholder(title))

    # Reuse the same shared teacher agenda styling as Algebra when available.
    calc_assets = SHARED_CALC_ROOT / "assets"
    calc_assets.mkdir(parents=True, exist_ok=True)
    source_css = SHARED_ROOT / "algebra" / "assets" / "teacher_portal.css"
    target_css = calc_assets / "teacher_portal.css"
    if source_css.is_file():
        target_css.write_bytes(source_css.read_bytes())
    elif not target_css.is_file():
        atomic_write_text(target_css, "body{font-family:Arial,Helvetica,sans-serif;margin:0;color:#1f2937}.wrapper{width:min(1500px,calc(100% - 28px));margin:14px auto}.titlebar,.section-title{background:#173f6d;color:#fff;padding:12px}.resource-block{background:#fff9e9;border:1px solid #e0bd4f;padding:12px}.resource-row{display:flex;flex-wrap:wrap;gap:8px}.resource-row a{border:1px solid #e0bd4f;border-radius:999px;padding:7px 12px;color:#173f6d;text-decoration:none}.calendar-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;table-layout:fixed;min-width:1050px}th,td{border:1px solid #cfd4da;padding:7px}.week-head th{background:#173f6d;color:#fff}.teacher-row td{background:#e8eff7}.teacher-link{color:#25486f;font-weight:800;text-decoration:none}")

    data_dir = SHARED_CALC_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not PRIVATE_ARTIFACTS.is_file():
        atomic_write_text(PRIVATE_ARTIFACTS, json.dumps({"schema": "teacher-private-artifacts/1.0", "entries": []}, indent=2) + "\n")

    library = SHARED_CALC_ROOT / "library"
    atomic_write_text(library / "index.html", build_calc_library_index())
    for unit in range(1, 9):
        unit_dir = library / f"unit{unit}"
        atomic_write_text(unit_dir / "index.html", build_calc_unit_library(unit))
        if not (APCALC_ROOT / f"banks/unit{unit}/unit{unit}.html").is_file():
            atomic_write_text(unit_dir / "bank.html", f'<!doctype html><meta charset="utf-8"><title>AP Calculus AB Unit {unit} Bank</title><style>body{{font-family:Arial,Helvetica,sans-serif;max-width:800px;margin:48px auto;padding:0 20px;color:#173f6d}}h1{{background:#173f6d;color:white;padding:18px;border-radius:12px}}</style><h1>AP Calculus AB Unit {unit} Bank</h1><p>This unit bank has not been built yet.</p><p><a href="index.html">Back to Unit {unit} Teacher Library</a></p>')

    assess = SHARED_CALC_ROOT / "assessments"
    assess.mkdir(parents=True, exist_ok=True)
    atomic_write_text(assess / "index.html", '<!doctype html><meta http-equiv="refresh" content="0; url=../library/index.html"><p><a href="../library/index.html">Open AP Calculus AB Teacher Library</a></p>')
    quick = assess / "quick_checks" / "index.html"
    quick.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(quick, '<!doctype html><meta http-equiv="refresh" content="0; url=../../library/index.html"><p><a href="../../library/index.html">Open AP Calculus AB Teacher Library</a></p>')

def compact_text(parts) -> str:
    return re.sub(r"\s+", " ", "".join(parts)).strip()


class AgendaParser(HTMLParser):
    """Parse the existing static agenda so Planner can adopt it once."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.capture = False
        self.tbody_classes = []
        self.rows = []
        self.current_row = None
        self.current_cell = None
        self.current_anchor = None
        self.weeks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tbody":
            classes = attrs.get("class", "").split()
            # Ignore the featured duplicate current week; capture ALL WEEKS blocks only.
            if "week-block" in classes and (
                "previous-week" in classes or "all-current-week" in classes
            ):
                self.capture = True
                self.tbody_classes = classes
                self.rows = []
        elif self.capture and tag == "tr":
            self.current_row = []
        elif self.capture and tag in {"th", "td"} and self.current_row is not None:
            self.current_cell = {
                "tag": tag,
                "text_parts": [],
                "href": None,
                "classes": attrs.get("class", "").split(),
                "colspan": int(attrs.get("colspan", "1") or 1),
            }
        elif self.capture and tag == "a" and self.current_cell is not None:
            self.current_anchor = attrs.get("href")
            if self.current_cell["href"] is None:
                self.current_cell["href"] = self.current_anchor

    def handle_data(self, data):
        if self.capture and self.current_cell is not None:
            self.current_cell["text_parts"].append(data)

    def handle_endtag(self, tag):
        if not self.capture:
            return
        if tag == "a":
            self.current_anchor = None
        elif tag in {"th", "td"} and self.current_cell is not None:
            self.current_cell["text"] = compact_text(self.current_cell.pop("text_parts"))
            if self.current_anchor and not self.current_cell["href"]:
                self.current_cell["href"] = self.current_anchor
            self.current_row.append(self.current_cell)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "tbody":
            self.weeks.append({"classes": list(self.tbody_classes), "rows": self.rows})
            self.capture = False
            self.tbody_classes = []
            self.rows = []


def academic_iso(mmdd: str) -> str | None:
    m = re.search(r"(\d{1,2})/(\d{1,2})", mmdd or "")
    if not m:
        return None
    month, day = map(int, m.groups())
    year = 2026 if month >= 7 else 2027
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def section_code(title: str) -> str | None:
    m = re.match(r"\s*(\d+)\.(\d+)", title or "")
    if not m:
        return None
    return f"{int(m.group(1))}.{int(m.group(2))}"


def section_sort_key(title: str):
    code = section_code(title)
    if not code:
        return (999, 999, title)
    u, s = code.split(".")
    return (int(u), int(s), title)


def parse_welcome_day(url: str | None) -> int | None:
    if not url:
        return None
    m = re.search(r"/(\d+)_(\d+)_(\d+)_welcome\.html(?:\?|$)", url)
    if not m:
        return None
    return int(m.group(3))


def normalize_control(label: str) -> str:
    low = (label or "").strip().casefold()
    if not low:
        return "Regular Day"
    if "summative" in low:
        return "Regular Day"
    if "sub" in low:
        return "Sub Assignment"
    if "nwea" in low or "testing" in low or low == "test":
        return "Testing"
    if "ok2say" in low or "assembly" in low:
        return "Assembly Schedule"
    if low in {"p.d.", "pd", "p.d"} or "professional development" in low:
        return "P.D."
    if "half" in low or "1/2" in low:
        return "1/2 Day"
    if "snow" in low:
        return "Snow Day"
    if "no school" in low:
        return "No School"
    if "break" in low or "labor day" in low or "holiday" in low:
        return "Break"
    return label if label in DAY_CONTROLS else "Regular Day"


def default_slots(day_number: int | None) -> list[str]:
    if day_number == 1:
        return ["Warmups", "Notes", "Exercises", "", ""]
    if day_number == 2:
        return ["Warmups", "Explore", "Exercises", "Sets", ""]
    if day_number == 3:
        return ["Warmups", "Lab", "Exercises", "Sets", "Quick Check"]
    return ["", "", "", "", ""]


def expand_row(row: list[dict]) -> list[dict]:
    cells = []
    for cell in row:
        span = max(1, int(cell.get("colspan", 1) or 1))
        for _ in range(span):
            cells.append(cell)
    while len(cells) < 5:
        cells.append({"text": "", "href": None, "classes": [], "tag": "td", "colspan": 1})
    return cells[:5]


def item_kind(label: str, first: bool) -> str:
    if not first:
        return ""
    low = (label or "").casefold()
    if any(t in low for t in ("labor day", "p.d", "no school", "holiday", "break", "snow")):
        return "holiday"
    return "lesson"


def parse_existing_agenda(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Student agenda not found: {path}")
    parser = AgendaParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if not parser.weeks:
        raise RuntimeError("Could not find ALL WEEKS blocks in the current student agenda.")

    days = []
    current_week_index = 0
    section_occurrence = {}

    for week_index, week in enumerate(parser.weeks):
        if "all-current-week" in week["classes"]:
            current_week_index = week_index
        rows = week["rows"]
        if not rows:
            continue
        header = expand_row(rows[0])
        data_rows = [expand_row(r) for r in rows[1:] if r]
        for col in range(5):
            date_label = header[col].get("text", "")
            iso = academic_iso(date_label)
            if not iso:
                iso = f"2026-01-{col+1:02d}"
            d = datetime.strptime(iso, "%Y-%m-%d").date()
            items = []
            for row in data_rows:
                cell = row[col]
                label = cell.get("text", "").strip()
                if not label or label == "No published agenda items.":
                    continue
                items.append({
                    "label": label,
                    "url": cell.get("href"),
                    "kind": item_kind(label, not items),
                })

            first = items[0] if items else None
            section = first["label"] if first and section_code(first["label"]) else ""
            control = "Regular Day" if section else normalize_control(first["label"] if first else "")
            day_number = parse_welcome_day(first.get("url") if first else None)
            if section:
                code = section_code(section)
                if day_number is None:
                    count = section_occurrence.get(code, 0) + 1
                    section_occurrence[code] = count
                    day_number = ((count - 1) % 3) + 1
                else:
                    section_occurrence[code] = max(section_occurrence.get(code, 0), day_number)

            past = week_index < current_week_index
            if week_index <= current_week_index:
                slots = [i["label"] for i in items[1:6]]
                slots += [""] * (5 - len(slots))
            elif section and control == "Regular Day":
                slots = default_slots(day_number)
            else:
                slots = ["", "", "", "", ""]

            days.append({
                "week_index": week_index,
                "date": iso,
                "dow": d.strftime("%A"),
                "display_date": f"{d.month}/{d.day}",
                "control": control,
                "section": section,
                "day_number": day_number or 0,
                "student_slots": slots,
                "custom_slots": [{"label": "", "url": ""} for _ in range(5)],
                "legacy_items": items if past else [],
                "locked_past": False,
                # Existing non-regular dates were already reflected in the old sheet plan.
                "shift_applied": control in SHIFTING_CONTROLS,
                "assessment_unit": 0,
            })

    return {
        "schema": "apcalc-planner-state/0.4",
        "school_year": "2026-2027",
        "current_week_index": current_week_index,
        "days": days,
        "created_from": "agenda/index.html",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_quick_check_map() -> dict[str, str]:
    if not QUICK_CHECK_LIBRARY.is_file():
        return {}
    try:
        data = json.loads(QUICK_CHECK_LIBRARY.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for entry in data.get("entries", []):
        section = str(entry.get("section") or "").strip()
        rel = str(entry.get("html") or "").strip()
        if section and rel:
            out[section] = f"{PUBLIC_BASE}quick_check____htq5855/{rel}"
    return out


def rel_if_exists(relative: str) -> str | None:
    path = APCALC_ROOT / relative
    if path.is_file():
        return PUBLIC_BASE + quote(relative, safe="/._-")
    return None


def normalize_custom_url(raw: str) -> str | None:
    value = str(raw or "").strip()
    if not value:
        return None
    if re.match(r"^https?://", value, flags=re.I):
        return value
    if value.startswith(("/", "./", "../")):
        return value
    if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/|$)", value):
        return "https://" + value
    return None


def resource_url(resource: str, section_title: str, day_number: int = 0) -> str | None:
    code = section_code(section_title)
    if not code:
        return None
    unit_s, sec_s = code.split(".")
    unit, sec = int(unit_s), int(sec_s)
    tag = f"{unit}_{sec}"
    # Canonical simplified Calculus curriculum.
    if resource == "Warmups":
        # Weekly warmups are pace-aware and are resolved by weekly_warmup_url().
        # Never fall back to the retired section-placeholder warmup pages.
        return None
    if resource == "Notes":
        return rel_if_exists(f"notes/u{tag}_notes/u{tag}_notes.html")
    if resource == "Exercises":
        return rel_if_exists(f"exercises/u{tag}_exercises/u{tag}_exercises.html")
    if resource == "Sets":
        return rel_if_exists(f"sets/u{tag}_sets/u{tag}_sets.html")
    if resource == "Explore":
        return rel_if_exists(f"explore/u{tag}_explore/u{tag}_explore.html")
    if resource == "Lab":
        return rel_if_exists(f"lab/u{tag}_lab/u{tag}_lab.html")
    if resource == "Quick Check":
        return None
    # Legacy history support.
    if resource == "Warm Up":
        return rel_if_exists(f"warmups/unit_{unit}_warmups/unit_{unit}_warmups.html")
    if resource == "Investigation":
        return rel_if_exists(f"investigations/u{tag}_investigation/u{tag}_investigation.html")
    if resource == "Activity":
        return rel_if_exists(f"activities/u{tag}_act1/u{tag}_act1.html")
    if resource == "Demo":
        return rel_if_exists(f"demo/u{tag}_demo/u{tag}_demo.html")
    if resource == "Performance Task":
        return rel_if_exists(f"performance_task___5134zrt/u{tag}_performance_task/u{tag}_performance_task.html")
    if resource == "Practice Set":
        return rel_if_exists(f"practice_sets/practice_sets_{tag}/practice_set_{tag}.html")
    if resource == "Extra Practice":
        return rel_if_exists(f"practice_sets/practice_sets_{tag}/extra_practice_{tag}.html")
    return None

def welcome_url(section_title: str, day_number: int) -> str | None:
    code = section_code(section_title)
    if not code or day_number not in {1, 2, 3}:
        return None
    unit_s, sec_s = code.split(".")
    relative = f"misc/welcomes/slides/{int(unit_s)}_{int(sec_s)}_{day_number}_welcome.html"
    return rel_if_exists(relative)


def teacher_links(section_title: str, student_slots: list[str]) -> list[dict]:
    code = section_code(section_title)
    if not code:
        return []
    unit_s, sec_s = code.split(".")
    tag = f"{int(unit_s)}_{int(sec_s)}"
    links = []

    def add(label: str, relative: str):
        url = rel_if_exists(relative)
        if url and not any(x["url"] == url for x in links):
            links.append({"label": label, "url": url})

    if "Notes" in student_slots:
        add("Notes Teacher Guide", f"notes/u{tag}_notes/u{tag}_notes_teacher.html")
    if "Investigation" in student_slots:
        add(
            "Investigation Teacher Guide",
            f"investigations/u{tag}_investigation/u{tag}_investigation_teacher_guide.html",
        )
    if "Demo" in student_slots:
        add(
            "Demo Teacher Guide",
            f"demo/u{tag}_demo/u{tag}_demo_teacher_guide.html",
        )
    if "Performance Task" in student_slots:
        add(
            "Performance Task Teacher Guide",
            f"performance_task___5134zrt/u{tag}_performance_task/u{tag}_performance_task_teacher_guide.html",
        )
    return links


def load_private_artifacts() -> list[dict]:
    if not PRIVATE_ARTIFACTS.is_file():
        return []
    try:
        data = json.loads(PRIVATE_ARTIFACTS.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = []
    for raw in data.get("entries", []):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        rel = str(item.get("path") or "").strip().lstrip("/")
        external = str(item.get("url") or "").strip()
        if rel:
            local = (SHARED_CALC_ROOT / rel).resolve()
            try:
                local.relative_to(SHARED_CALC_ROOT.resolve())
            except ValueError:
                continue
            if not local.is_file():
                continue
            encoded = quote(rel, safe="/._-")
            item["planner_url"] = f"/shared/calc/{encoded}"
            item["shared_url"] = encoded
        elif external:
            item["planner_url"] = external
            item["shared_url"] = external
        else:
            continue
        entries.append(item)
    return entries


def private_links_for_section(code: str) -> list[dict]:
    out = []
    for item in load_private_artifacts():
        if str(item.get("section") or "").strip() != code:
            continue
        label = str(item.get("label") or "Teacher Resource").strip()
        out.append({
            "label": label,
            "planner_url": item["planner_url"],
            "shared_url": item["shared_url"],
            "role": str(item.get("role") or "Teacher").strip() or "Teacher",
        })
    return out


def private_summative_links(unit: int) -> list[dict]:
    out = []
    valid_kinds = {
        "summative_assessment",
        "summative_answer_key",
        "summative_teacher_guide",
    }
    for item in load_private_artifacts():
        kind = str(item.get("kind") or "").strip().casefold()
        try:
            item_unit = int(item.get("unit"))
        except Exception:
            continue
        if item_unit != unit or kind not in valid_kinds:
            continue
        label = str(item.get("label") or "Summative Assessment").strip()
        default_role = "Assessment" if kind == "summative_assessment" else "Teacher"
        out.append({
            "label": label,
            "planner_url": item["planner_url"],
            "shared_url": item["shared_url"],
            "role": str(item.get("role") or default_role).strip() or default_role,
        })
    return out


def build_registry(state: dict) -> dict:
    sections = sorted(
        {d.get("section", "") for d in state.get("days", []) if section_code(d.get("section", ""))},
        key=section_sort_key,
    )
    units = sorted({int(section_code(s).split(".")[0]) for s in sections if section_code(s)})
    registry = {
        "sections": sections,
        "resources": {},
        "teacher": {},
        "welcome": {},
        "weekly_warmups": {},
        "shared": {"section": {}, "summative": {}},
    }
    for day in state.get("days", []):
        key = str(int(day.get("week_index", 0)))
        if key not in registry["weekly_warmups"]:
            registry["weekly_warmups"][key] = planner_weekly_warmup_url(day)

    for section in sections:
        registry["resources"][section] = {
            r: resource_url(r, section) for r in STUDENT_RESOURCE_OPTIONS if r
        }
        registry["teacher"][section] = {}
        for resource in ["Notes", "Investigation", "Demo", "Performance Task", "Quick Check"]:
            entries = teacher_links(section, [resource])
            registry["teacher"][section][resource] = entries[0] if entries else None
        registry["welcome"][section] = {
            str(n): welcome_url(section, n) for n in (1, 2, 3)
        }
        code = section_code(section)
        registry["shared"]["section"][code] = [
            {"label": e["label"], "url": e["planner_url"], "role": e["role"]}
            for e in private_links_for_section(code)
        ]
    for unit in units:
        registry["shared"]["summative"][str(unit)] = [
            {"label": e["label"], "url": e["planner_url"], "role": e["role"]}
            for e in private_summative_links(unit)
        ]
    return registry


def apply_manual_current_state(state: dict) -> None:
    days = state.get("days", [])
    if not days:
        state["current_week_index"] = 0
        return
    valid_weeks = sorted({int(d.get("week_index", 0)) for d in days})
    try:
        chosen = int(state.get("current_week_index", valid_weeks[0]))
    except Exception:
        chosen = valid_weeks[0]
    if chosen not in valid_weeks:
        chosen = valid_weeks[0]
    state["current_week_index"] = chosen
    state["schema"] = "apcalc-planner-state/0.4"
    for idx, d in enumerate(days):
        # Every week remains editable. Current week controls highlighting/publishing, not locks.
        d["locked_past"] = False
        d.setdefault("shift_applied", False)
        d.setdefault("assessment_unit", 0)
        # Migrate the short-lived Summative/Remove-Blank experiment into the simpler model.
        if d.get("control") == "Summative Assessment":
            unit = 0
            try:
                unit = int(d.get("assessment_unit") or 0)
            except Exception:
                unit = 0
            if not unit:
                for prior in reversed(days[:idx]):
                    code = section_code(str(prior.get("section") or ""))
                    if code:
                        unit = int(code.split(".")[0])
                        break
            d["control"] = "Regular Day"
            d["section"] = str(d.get("section") or "").strip() or (f"Unit {unit} Test" if unit else "Unit Test")
            d["day_number"] = 0
            d["student_slots"] = ["", "", "", "", ""]
            d["shift_applied"] = False
        elif d.get("control") == "Insert Blank Day" and d.get("shift_applied"):
            # The shift was already applied by the previous Planner version. Keep the blank date,
            # but reset the dropdown so selecting Insert again is always a fresh action.
            d["control"] = "Regular Day"
            d["shift_applied"] = False
        custom = d.get("custom_slots")
        if not isinstance(custom, list):
            custom = []
        normalized = []
        for i in range(5):
            item = custom[i] if i < len(custom) and isinstance(custom[i], dict) else {}
            normalized.append({
                "label": str(item.get("label") or ""),
                "url": str(item.get("url") or ""),
            })
        d["custom_slots"] = normalized


LAST_WEEK_REPAIR_ID = "2026-09-21-sheet-repair-v1"
UNIT_TEST_LABELS_ID = "unit-transition-test-labels-v1"
IMPORTED_RESOURCE_NORMALIZE_ID = "normalize-imported-custom-resources-v1"


def normalize_imported_resources(state: dict) -> bool:
    """Preserve legacy agenda labels that are not Planner resource types as Custom Items.

    The old spreadsheet agenda contains legitimate one-off rows such as Overview, CYU,
    Progress, PHet, and similar links. They should remain editable, but they are not
    fixed Planner resource types. Convert only unsupported values, preserving the old
    label and URL whenever the imported legacy item is available.
    """
    migrations = state.get("migrations")
    if not isinstance(migrations, list):
        migrations = []
        state["migrations"] = migrations

    changed = False
    for day in state.get("days", []):
        if not isinstance(day, dict):
            continue
        slots = day.get("student_slots")
        if not isinstance(slots, list):
            continue
        custom = day.get("custom_slots")
        if not isinstance(custom, list):
            custom = []
        while len(custom) < 5:
            custom.append({"label": "", "url": ""})
        legacy = day.get("legacy_items")
        if not isinstance(legacy, list):
            legacy = []

        for i in range(min(5, len(slots))):
            value = str(slots[i] or "")
            if value in STUDENT_RESOURCE_OPTIONS:
                continue
            legacy_item = legacy[i + 1] if i + 1 < len(legacy) and isinstance(legacy[i + 1], dict) else {}
            existing = custom[i] if isinstance(custom[i], dict) else {}
            custom[i] = {
                "label": value,
                "url": str(existing.get("url") or legacy_item.get("url") or ""),
            }
            slots[i] = "Custom Item"
            changed = True
        day["student_slots"] = slots[:5] + [""] * max(0, 5 - len(slots))
        day["custom_slots"] = custom[:5]

    if IMPORTED_RESOURCE_NORMALIZE_ID not in migrations:
        migrations.append(IMPORTED_RESOURCE_NORMALIZE_ID)
        changed = True
    return changed


def apply_known_state_repairs(state: dict) -> bool:
    """One-time repair for the sparse 9/21 week imported from the old static agenda.

    The old Google Sheet had the selected items shown below, but the first Planner
    import only captured a few of them. Repair only the exact sparse pattern so
    later teacher edits are never overwritten.
    """
    migrations = state.get("migrations")
    if not isinstance(migrations, list):
        migrations = []
        state["migrations"] = migrations
    if LAST_WEEK_REPAIR_ID in migrations:
        return False

    by_date = {
        str(d.get("date") or ""): d
        for d in state.get("days", [])
        if isinstance(d, dict)
    }
    required = ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"]
    if not all(x in by_date for x in required):
        return False

    d21, d22, d23, d24, d25 = (by_date[x] for x in required)
    sparse_import = (
        str(d21.get("control") or "") == "P.D."
        and str(d22.get("control") or "") == "Sub Assignment"
        and not str(d22.get("section") or "").strip()
        and not str(d23.get("section") or "").strip()
        and section_code(str(d24.get("section") or "")) == "1.2"
        and not str(d25.get("section") or "").strip()
    )
    if not sparse_import:
        return False

    def set_day(day: dict, *, control: str, section: str, day_number: int, slots: list[str]):
        day["control"] = control
        day["section"] = section
        day["day_number"] = day_number
        day["student_slots"] = list(slots)
        day["custom_slots"] = [{"label": "", "url": ""} for _ in range(5)]
        day["locked_past"] = False
        day["shift_applied"] = control in SHIFTING_CONTROLS
        day["assessment_unit"] = 0

    # These are the checked/selected items from the 9/21-9/25 legacy sheet.
    set_day(d22, control="Sub Assignment", section="1.2: I/O & Functions", day_number=1,
            slots=["", "", "", "", ""])
    set_day(d23, control="Regular Day", section="1.2: I/O & Functions", day_number=2,
            slots=["Warm Up", "Activity", "Practice Set", "", ""])
    set_day(d24, control="Regular Day", section="1.2: I/O & Functions", day_number=3,
            slots=["Warm Up", "Activity", "Practice Set", "Extra Practice", ""])
    set_day(d25, control="Regular Day", section="1.3: Fn Families", day_number=1,
            slots=["Warm Up", "Notes", "Practice Set", "", ""])

    migrations.append(LAST_WEEK_REPAIR_ID)
    return True


def apply_unit_test_labels(state: dict) -> bool:
    """Fill the planned blank day between units with `Unit N Test`.

    This is a one-time planner migration. It does not create a new day and it
    does not shift instruction. It only labels an already-empty Regular Day
    that sits between the last section of one unit and the first section of the
    next unit. Existing teacher-entered content is never overwritten.
    """
    migrations = state.get("migrations")
    if not isinstance(migrations, list):
        migrations = []
        state["migrations"] = migrations
    if UNIT_TEST_LABELS_ID in migrations:
        return False

    days = state.get("days", [])
    changed = False

    def unit_of(day: dict) -> int:
        code = section_code(str(day.get("section") or ""))
        if not code:
            return 0
        return int(code.split(".", 1)[0])

    instructional = [(i, unit_of(day)) for i, day in enumerate(days) if unit_of(day)]
    for pos in range(len(instructional) - 1):
        left_i, left_unit = instructional[pos]
        right_i, right_unit = instructional[pos + 1]
        if right_unit <= left_unit:
            continue
        # Only a true unit transition gets a test label. Choose the first empty
        # Regular Day between the units, exactly matching the calendar pattern
        # used for Unit 1 in the Planner.
        for j in range(left_i + 1, right_i):
            day = days[j]
            if str(day.get("control") or "Regular Day") != "Regular Day":
                continue
            if str(day.get("section") or "").strip():
                continue
            if any(str(x or "").strip() for x in day.get("student_slots", [])):
                continue
            day["section"] = f"Unit {left_unit} Test"
            day["day_number"] = 0
            day["student_slots"] = ["", "", "", "", ""]
            day["custom_slots"] = [{"label": "", "url": ""} for _ in range(5)]
            day["assessment_unit"] = left_unit
            day["locked_past"] = False
            day["shift_applied"] = False
            changed = True
            break

    migrations.append(UNIT_TEST_LABELS_ID)
    return changed


def validate_state(state: dict) -> None:
    if not isinstance(state, dict) or not isinstance(state.get("days"), list):
        raise ValueError("Planner state must contain a days list.")
    for i, d in enumerate(state["days"]):
        if not isinstance(d, dict):
            raise ValueError(f"days[{i}] must be an object")
        try:
            datetime.strptime(str(d.get("date")), "%Y-%m-%d")
        except Exception as exc:
            raise ValueError(f"days[{i}] has invalid date") from exc
        slots = d.get("student_slots")
        if not isinstance(slots, list) or len(slots) != 5:
            raise ValueError(f"days[{i}] must have exactly 5 student_slots")
        for value in slots:
            if value not in STUDENT_RESOURCE_OPTIONS:
                raise ValueError(f"days[{i}] has unsupported resource: {value}")
        custom = d.get("custom_slots", [])
        if not isinstance(custom, list) or len(custom) != 5:
            raise ValueError(f"days[{i}] must have exactly 5 custom_slots")
        for ci, item in enumerate(custom):
            if not isinstance(item, dict):
                raise ValueError(f"days[{i}].custom_slots[{ci}] must be an object")
            if not isinstance(item.get("label", ""), str) or not isinstance(item.get("url", ""), str):
                raise ValueError(f"days[{i}].custom_slots[{ci}] must contain string label/url")
        control = d.get("control", "Regular Day")
        if control not in DAY_CONTROLS:
            raise ValueError(f"days[{i}] has unsupported control: {control}")
        if d.get("assessment_unit") not in (None, "", 0):
            try:
                int(d.get("assessment_unit"))
            except Exception as exc:
                raise ValueError(f"days[{i}] has invalid assessment_unit") from exc


def get_state() -> dict:
    if STATE_PATH.is_file():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            apply_manual_current_state(state)
            repaired = apply_unit_test_labels(state)
            normalized = normalize_imported_resources(state)
            chapter3 = apply_chapter3_rebuild(state)
            apply_manual_current_state(state)
            validate_state(state)
            if repaired or normalized or chapter3:
                state["updated_at"] = datetime.now().isoformat(timespec="seconds")
                atomic_write_text(STATE_PATH, json.dumps(state, indent=2) + "\n")
            return state
        except Exception as exc:
            print(f"Planner state could not be loaded; rebuilding from agenda: {exc}")
    state = parse_existing_agenda(AGENDA_PATH)
    apply_manual_current_state(state)
    apply_unit_test_labels(state)
    normalize_imported_resources(state)
    apply_chapter3_rebuild(state)
    apply_manual_current_state(state)
    atomic_write_text(STATE_PATH, json.dumps(state, indent=2) + "\n")
    return state


def save_state(state: dict) -> dict:
    apply_manual_current_state(state)
    normalize_imported_resources(state)
    validate_state(state)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    atomic_write_text(STATE_PATH, json.dumps(state, indent=2) + "\n")
    return state


def day_items(day: dict, past: bool = False) -> list[dict]:
    # Published output always comes from the editable Planner state, including past weeks.
    items = []
    control = day.get("control", "Regular Day")
    section = day.get("section", "")
    if control in BLOCKED_CONTROLS:
        kind = "holiday" if control in {"Snow Day", "P.D.", "Break", "No School"} else "lesson"
        items.append({"label": control, "url": None, "kind": kind})
        return items

    if section:
        # Calculus student agendas use the six canonical resource types below.
        # The lesson title is a label, not a legacy Welcome-slide link.
        items.append({
            "label": section,
            "url": None,
            "kind": "lesson",
        })
    if control != "Regular Day":
        items.append({"label": control, "url": None, "kind": ""})

    custom_slots = day.get("custom_slots") or []
    for idx, resource in enumerate(day.get("student_slots", [])):
        if not resource:
            continue
        if resource == "Custom Item":
            custom = custom_slots[idx] if idx < len(custom_slots) and isinstance(custom_slots[idx], dict) else {}
            label = str(custom.get("label") or "Custom Item").strip() or "Custom Item"
            url = normalize_custom_url(custom.get("url"))
            items.append({"label": label, "url": url, "kind": ""})
            continue
        url = None
        if resource in {"Warmups", "Warm Up"}:
            # Current weekly warmups appear only when the real weekly page exists.
            # Do not link retired section placeholders when the week has not been built yet.
            url = weekly_warmup_url(day)
        else:
            url = resource_url(resource, section, int(day.get("day_number", 0) or 0))
        items.append({
            "label": resource,
            "url": url,
            "kind": "",
        })
    return items


def render_item(item: dict, class_name: str = "cal-link") -> str:
    label = escape(str(item.get("label") or ""))
    kind = item.get("kind") or ""
    classes = class_name + (f" {kind}" if kind else "")
    url = item.get("url")
    if url:
        return (
            f'<a class="{classes}" href="{escape(url, quote=True)}" '
            f'target="_blank" rel="noopener">{label}</a>'
        )
    return f'<span class="{classes} no-link">{label}</span>'


def render_week(state: dict, week_index: int, display_class: str, featured: bool = False) -> str:
    days = [d for d in state["days"] if int(d.get("week_index", 0)) == week_index]
    if len(days) < 5:
        return ""
    if featured:
        header_cells = "".join(
            f"<th><span class='dow'>{escape(d['dow'])}</span><span class='date'>{escape(d['display_date'])}</span></th>"
            for d in days[:5]
        )
    else:
        header_cells = "".join(
            f"<th><div class='date'>{escape(d['display_date'])}</div></th>" for d in days[:5]
        )

    past = week_index < int(state.get("current_week_index", 0))
    per_day = [day_items(d, past) for d in days[:5]]
    row_count = max((len(x) for x in per_day), default=0)
    rows = []
    for ri in range(row_count):
        cells = []
        for items in per_day:
            cells.append(f"<td>{render_item(items[ri])}</td>" if ri < len(items) else "<td></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    if not rows:
        rows.append('<tr><td colspan="5" class="empty-week">No published agenda items.</td></tr>')
    return (
        f'<tbody class="week-block {display_class}"><tr class="week-head">'
        f'{header_cells}</tr>{"".join(rows)}</tbody>'
    )


def build_student_agenda(state: dict) -> str:
    apply_manual_current_state(state)
    current = int(state.get("current_week_index", 0))
    week_indices = sorted({int(d.get("week_index", 0)) for d in state.get("days", [])})
    featured = render_week(state, current, "current-week", featured=True)
    all_weeks = []
    for wi in week_indices:
        cls = "all-current-week" if wi == current else "previous-week"
        all_weeks.append(render_week(state, wi, cls, featured=False))

    student_links = [
        ("Notes", "https://tnezki.github.io/apcalc/notes/"),
        ("Exercises", "https://tnezki.github.io/apcalc/exercises/"),
        ("Sets", "https://tnezki.github.io/apcalc/sets/"),
        ("Explore", "https://tnezki.github.io/apcalc/explore/"),
        ("Lab", "https://tnezki.github.io/apcalc/lab/"),
        ("Warmups", "https://tnezki.github.io/apcalc/warmups/"),
        ("Textbook", "https://tnezki.github.io/textbooks/apcalc/index.html"),
        ("AP Classroom", "https://myap.collegeboard.org/login"),
        ("Desmos", "https://www.desmos.com/calculator"),
        ("GeoGebra", "https://www.geogebra.org/graphing"),
    ]
    resources = "\n".join(
        f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener">{escape(label)}</a>'
        for label, url in student_links
    )
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>AP Calculus AB Agenda 2026-2027</title>
<style>
:root{{--navy:#173f6d;--gold:#e0bd4f;--gold-light:#fff0b8;--current-row-a:#fffaf0;--current-row-b:#fff1bd;--ink:#1f2937;--muted:#64748b;--lesson:#fff0b8;--link:#173f6d}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Arial,Helvetica,sans-serif;background:#fff;color:var(--ink)}}
.wrapper{{width:min(980px,calc(100% - 24px));margin:18px auto 40px}} .titlebar{{background:var(--navy);color:#fff;padding:7px 11px;border-radius:10px 10px 0 0;text-align:center}}
.titlebar h1{{margin:0;font-size:.88rem;font-weight:800}} .titlebar .small{{font-size:1em;font-weight:600}}
.resources{{border:1px solid var(--gold);border-top:0;padding:12px 14px 13px;display:flex;flex-wrap:wrap;justify-content:center;align-items:center;text-align:center;gap:8px;background:#fff9e9}}
.resources a{{text-decoration:none;color:var(--navy);background:#fff;border:1px solid var(--gold);border-radius:999px;padding:7px 11px;font-size:.88rem;font-weight:700}}
.resources a:hover,.resources a:focus-visible{{background:var(--gold-light);border-color:var(--navy)}} .calendar-wrap{{overflow-x:auto;border:1px solid #c8b675;border-top:0}}
table{{border-collapse:collapse;width:100%;max-width:100%;min-width:0;table-layout:fixed}} th,td{{border-right:1px solid #cfd4da;border-bottom:1px solid #cfd4da;text-align:center;vertical-align:middle}} tr>*:last-child{{border-right:0}}
.week-head th{{width:20%;padding:7px 5px}} .dow{{font-size:.82rem;font-weight:800}} .date{{margin-top:2px;font-size:.76rem;font-weight:700}} td{{padding:4px 5px;background:#fff;height:32px}}
.cal-link{{display:block;width:100%;text-decoration:none;color:var(--link);font-size:.82rem;font-weight:650;padding:4px;border-radius:5px;overflow-wrap:anywhere}} a.cal-link:hover,a.cal-link:focus-visible{{background:#fff4c7}}
.cal-link.lesson{{background:var(--lesson);border:1px solid #e1c86d;font-weight:800;color:var(--navy)}} .cal-link.holiday{{background:#f6edcf;color:#475569;font-weight:800}} .no-link{{cursor:default}} .empty-week{{color:var(--muted);font-size:.8rem;padding:10px}}
.current-week .week-head th{{background:var(--navy);color:#fff;text-align:left;padding:7px 6px}} .current-week .dow{{display:inline;font-size:.86rem;font-weight:850}} .current-week .date{{display:inline;margin:0 0 0 6px;font-size:.86rem;font-weight:900;color:#fff}}
.current-week td{{padding:0;height:40px;background:var(--current-row-a)}} .current-week tr:nth-child(odd):not(.week-head) td{{background:var(--current-row-b)}} .current-week .cal-link{{display:flex;align-items:center;justify-content:center;min-height:40px;padding:7px 4px;font-size:.90rem;font-weight:750;line-height:1.15}}
.current-week .cal-link.lesson{{background:rgba(255,255,255,.45);border:1px solid #dbc36d;color:var(--navy);font-size:.96rem;font-weight:900}}
.previous-weeks-divider td{{background:var(--navy)!important;color:#fff;font-size:.96rem;font-weight:800;padding:9px 7px}}
.all-current-week .week-head th{{background:var(--gold);color:var(--navy);border-top:5px solid var(--navy);text-align:left;padding-left:9px}} .all-current-week .week-head th .date{{color:var(--navy);font-size:.9rem;font-weight:900}}
.all-current-week tr td{{background:var(--current-row-a)!important;border-color:#d8c77f}} .all-current-week tr:nth-child(even) td{{background:var(--current-row-b)!important}} .all-current-week .cal-link.lesson{{background:#fff6d5;border:1px solid #dbc36d;color:var(--navy);font-weight:900}}
.previous-week .week-head th{{background:#3f4650;color:#fff;border-top:5px solid #20242a;text-align:left;padding-left:9px}} .previous-week .week-head th .date{{color:#e5e7eb}} .previous-week tr td{{background:#fff!important;border-color:#cfd4da}} .previous-week tr:nth-child(even) td{{background:#f3f4f6!important}} .previous-week .cal-link.lesson{{background:#e5e7eb;border:1px solid #c7ccd1;color:#2f3740}} .previous-week .cal-link.holiday{{background:#eceff2;color:#4b5563}}
.updated{{text-align:center;color:var(--muted);font-size:.76rem;padding-top:8px}} @media(max-width:700px){{.wrapper{{width:100%;margin:0}}.titlebar{{border-radius:0}}.resources{{justify-content:center;padding:10px}}.resources a{{font-size:.8rem;padding:6px 9px}}}}
</style>
</head>
<body>
<div class="wrapper">
<header class="titlebar"><h1>AP Calculus AB <span class="small">– Agenda 2026-2027</span></h1></header>
<nav class="resources" aria-label="Student resources">{resources}</nav>
<div class="calendar-wrap"><table aria-label="AP Calculus AB student agenda">{featured}<tbody class="previous-weeks-divider"><tr><td colspan="5">ALL WEEKS</td></tr></tbody>{''.join(all_weeks)}</table></div>
<div class="updated">Planner updated {escape(stamp)}</div>
</div>
</body>
</html>'''


def visible_resource_labels(day: dict, past: bool = False) -> list[str]:
    return [str(x) for x in day.get("student_slots", []) if x and x != "Custom Item"]


def infer_unit_for_day(state: dict, day: dict) -> int:
    try:
        unit = int(day.get("assessment_unit") or 0)
        if unit:
            return unit
    except Exception:
        pass
    code = section_code(str(day.get("section") or ""))
    if code:
        return int(code.split(".")[0])
    days = state.get("days", [])
    try:
        idx = days.index(day)
    except ValueError:
        idx = -1
    for i in range(idx - 1, -1, -1):
        code = section_code(str(days[i].get("section") or ""))
        if code:
            return int(code.split(".")[0])
    return 0


def teacher_links_for_day(state: dict, day: dict, shared_mode: bool = False) -> list[dict]:
    past = int(day.get("week_index", 0)) < int(state.get("current_week_index", 0))
    links = []

    def add(label: str, url: str | None, role: str = "Teacher"):
        if not url or any(x["url"] == url for x in links):
            return
        links.append({"label": label, "url": url, "role": role})

    section = str(day.get("section") or "")
    labels = visible_resource_labels(day, past)
    if re.search(r"\b(?:unit\s*\d+\s*)?(?:test|exam|assessment)\b", section, re.I):
        unit = infer_unit_for_day(state, day) or 1
        target = f"library/unit{unit}/index.html#assessments"
        add(
            "Approved Assessments",
            target if shared_mode else f"/shared/calc/{target}",
            "Assessment",
        )
    if section_code(section):
        for entry in teacher_links(section, labels):
            add(entry["label"], entry["url"], "Teacher")
        if "Quick Check" in labels:
            unit = infer_unit_for_day(state, day) or 1
            target = f"library/unit{unit}/index.html#quick-checks"
            add(
                "Approved Quick Checks",
                target if shared_mode else f"/shared/calc/{target}",
                "Assessment",
            )
        code = section_code(section)
        for entry in private_links_for_section(code):
            add(
                entry["label"],
                entry["shared_url"] if shared_mode else entry["planner_url"],
                entry["role"],
            )
    return links[:3]


def render_teacher_link(item: dict) -> str:
    role = escape(str(item.get("role") or "Teacher"))
    label = escape(str(item.get("label") or "Teacher Resource"))
    url = escape(str(item.get("url") or ""), quote=True)
    return (
        f'<span class="teacher-label">{role}</span>'
        f'<a class="teacher-link" href="{url}" target="_blank" rel="noopener">{label} ↗</a>'
    )


def render_teacher_week(state: dict, week_index: int, display_class: str, featured: bool = False) -> str:
    days = [d for d in state["days"] if int(d.get("week_index", 0)) == week_index][:5]
    if len(days) < 5:
        return ""
    if featured:
        header = "".join(
            f"<th>{escape(d['dow'])}<span class='date'>{escape(d['display_date'])}</span></th>"
            for d in days
        )
    else:
        header = "".join(
            f"<th><span class='date'>{escape(d['display_date'])}</span></th>" for d in days
        )
    past = week_index < int(state.get("current_week_index", 0))
    per_day = [day_items(d, past) for d in days]
    row_count = max((len(items) for items in per_day), default=0)
    rows = []
    for ri in range(row_count):
        cells = []
        for items in per_day:
            cells.append(
                f"<td>{render_item(items[ri], 'item')}</td>" if ri < len(items) else "<td></td>"
            )
        row_class = "lesson-row" if ri == 0 else ("student-row alt" if ri % 2 == 1 else "student-row")
        rows.append(f'<tr class="{row_class}">' + "".join(cells) + "</tr>")
    for row in range(3):
        cells = []
        for d in days:
            links = teacher_links_for_day(state, d, shared_mode=True)
            cells.append(
                f"<td>{render_teacher_link(links[row]) if row < len(links) else ''}</td>"
            )
        rows.append('<tr class="teacher-row">' + "".join(cells) + "</tr>")
    return (
        f'<tbody class="week-block {display_class}"><tr class="week-head">'
        f'{header}</tr>{"".join(rows)}</tbody>'
    )


def current_unit_for_state(state: dict) -> int:
    current = int(state.get("current_week_index", 0))
    units = []
    for day in state.get("days", []):
        if int(day.get("week_index", 0)) != current:
            continue
        unit = infer_unit_for_day(state, day)
        if unit:
            units.append(unit)
    return max(units) if units else 1


def build_teacher_agenda(state: dict) -> str:
    apply_manual_current_state(state)
    current = int(state.get("current_week_index", 0))
    week_indices = sorted({int(d.get("week_index", 0)) for d in state.get("days", [])})
    featured = render_teacher_week(state, current, "current-week", featured=True)
    previous_candidates = [wi for wi in week_indices if wi < current]
    previous = previous_candidates[-1] if previous_candidates else None
    previous_featured = (
        render_teacher_week(state, previous, "previous-week", featured=True)
        if previous is not None else ""
    )
    all_weeks = [
        render_teacher_week(
            state,
            wi,
            "all-current-week" if wi == current else "previous-week",
            featured=False,
        )
        for wi in week_indices
    ]
    student_links = [
        ("Overview", "https://docs.google.com/document/d/1rrToxZ84-VGe-75JeIofcH6FXqdCFiE-MNlwDZNvMs4/edit?usp=sharing"),
        ("AP Classroom", "https://myap.collegeboard.org/login"),
        ("AP Central", "https://apcentral.collegeboard.org/courses/ap-calculus-ab"),
        ("Textbook", "https://tnezki.github.io/textbooks/apcalc/index.html"),
        ("Printables", "https://tnezki.github.io/algebra/misc/printables/aaagallery_index.html"),
        ("Upload Spot", "https://drive.google.com/drive/folders/10CsdXTol83A22hUiwB6c45CyY5eVCAxn?usp=drive_link"),
        ("Desmos", "https://www.desmos.com/calculator"),
        ("GeoGebra", "https://www.geogebra.org/graphing"),
        ("When You See", "https://drive.google.com/open?id=0B6fznOHK0RHAWHAwVjhIQmN2UEE"),
        ("Cheat Sheet", "https://drive.google.com/open?id=0B6fznOHK0RHAczRoVkw3QkEyc1U"),
        ("Note Cards", "https://drive.google.com/open?id=0B6fznOHK0RHAZGtQZWZmZHFJM00"),
        ("Parent Functions", "https://tnezki.github.io/pc/resources/parent_fuctions.pdf"),
        ("Slope Fields", "https://www.geogebra.org/m/Pd4Hn4BR"),
        ("Cross Sections", "https://www.geogebra.org/m/XFgMaKTy"),
        ("Calc in 20 Minutes", "https://www.youtube.com/watch?v=SOkMGWCLqoc"),
        ("Khan Academy", "https://www.khanacademy.org/math/calculus-1"),
        ("FRQ Videos", "https://www.youtube.com/playlist?list=PL3Gnjw2fQSnH_Okp7aPcH71yC8SqHKc-o"),
    ]
    current_unit = current_unit_for_state(state)
    unit_library = f"library/unit{current_unit}/index.html"
    teacher_links_top = [
        ("Year at a Glance", "https://docs.google.com/document/d/1QvXUsWFheXsYBYnz3dk3_YMlkLDRNXAj06kYKm--pyU/edit?usp=sharing"),
        ("Teacher Library", unit_library),
        ("Activity Structures", "https://docs.google.com/document/d/18imPwnYQjblasMS8x97DjKeNDjWNKTnOa3iBoTwCYVA/edit?usp=sharing"),
        ("Reflections", "https://docs.google.com/document/d/1vFhhtZUWGwT5vvVOrHxnkk1lRY_nZe7zCe4bhSPRwBw/edit?usp=drive_link"),
        ("Welcomes", "https://tnezki.github.io/apcalc/misc/welcomes/welcome.html"),
        ("Student Agenda", "https://tnezki.github.io/apcalc/agenda/index.html"),
    ]

    def nav(items):
        return "".join(
            f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener">{escape(label)}</a>'
            for label, url in items
        )

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AP Calculus AB Shared Teacher Agenda</title>
<link rel="stylesheet" href="assets/teacher_portal.css">
</head>
<body>
<div class="wrapper">
<header class="titlebar"><h1>AP Calculus AB - Shared Teacher Agenda</h1></header>
<section class="resource-block">
<p class="resource-label">Student Resources</p><div class="resource-row">{nav(student_links)}</div>
<p class="resource-label">Shared Teacher Resources</p><div class="resource-row">{nav(teacher_links_top)}</div>
</section>
<div class="notice">Read-only co-teacher view generated from the local Planner. Current and previous week are pinned at the top; ALL WEEKS remains below for reference.</div>
<div class="section-title">CURRENT WEEK - TEACHER VIEW</div>
<div class="calendar-wrap"><table aria-label="Current teacher agenda">{featured}</table></div>
{('<div class="section-title">PREVIOUS WEEK - TEACHER VIEW</div><div class="calendar-wrap"><table aria-label="Previous teacher agenda">' + previous_featured + '</table></div>') if previous_featured else ''}
<div class="section-title">ALL WEEKS</div>
<div class="calendar-wrap"><table aria-label="All teacher agenda weeks">{''.join(all_weeks)}</table></div>
<div class="footer">Planner published {escape(stamp)}</div>
</div>
</body>
</html>'''


def bootstrap_payload() -> dict:
    state = get_state()
    return {
        "state": state,
        "registry": build_registry(state),
        "options": {"resources": EDITOR_RESOURCE_OPTIONS, "controls": EDITOR_DAY_CONTROLS},
    }


class PlannerHandler(SimpleHTTPRequestHandler):
    server_version = "APCalcPlanner/0.4"

    def log_message(self, fmt, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")

    def _json(self, status: int, payload: dict):
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 2_000_000:
            raise ValueError("Invalid request body length")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _serve_shared_file(self, url_path: str):
        rel = unquote(url_path[len("/shared/"):]).lstrip("/")
        target = (SHARED_ROOT / rel).resolve()
        target.relative_to(SHARED_ROOT.resolve())
        if not target.is_file():
            self.send_error(404, "Shared teacher file not found")
            return
        raw = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_apcalc_file(self, url_path: str):
        rel = unquote(url_path[len("/apcalc/"):]).lstrip("/")
        target = (APCALC_ROOT / rel).resolve()
        target.relative_to(APCALC_ROOT.resolve())
        if not target.is_file():
            self.send_error(404, "AP Calculus AB file not found")
            return
        raw = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/version":
            self._json(200, {"ok": True, "version": PLANNER_VERSION})
            return
        if path == "/api/library":
            try:
                self._json(200, library_payload())
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
            return
        if path == "/api/exercises/sections":
            try:
                self._json(200, exercise_sections_payload())
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
            return
        if path == "/api/exercises/section":
            try:
                query = parse_qs(urlparse(self.path).query)
                section = str((query.get("section") or [""])[0])
                self._json(200, {"ok": True, "config": _load_exercise_config(section)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        if path == "/api/warmups/weeks":
            try:
                self._json(200, _week_status_payload(get_state()))
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
            return
        if path == "/api/warmups/editor/weeks":
            try:
                self._json(200, warmup_editor.weeks_payload(WARMUP_CONFIG_ROOT, LIBRARY_ROOT, APCALC_ROOT, PUBLIC_BASE))
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
            return
        if path == "/api/warmups/editor/week":
            try:
                query = parse_qs(urlparse(self.path).query)
                week = int((query.get("week") or ["0"])[0])
                self._json(200, {"ok": True, "config": warmup_editor.load(WARMUP_CONFIG_ROOT, week)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        if path == "/api/warmups/plan":
            try:
                query = parse_qs(urlparse(self.path).query)
                week = int((query.get("week") or ["0"])[0])
                self._json(200, {"ok": True, "plan": build_warmup_plan(get_state(), week)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        if path.startswith("/downloads/"):
            try:
                name = Path(unquote(path[len("/downloads/"):])).name
                target = (WARMUP_REQUESTS_DIR / name).resolve()
                target.relative_to(WARMUP_REQUESTS_DIR.resolve())
                if not target.is_file():
                    self.send_error(404, "Download not found")
                    return
                raw = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)
            except Exception:
                self.send_error(404, "Download not found")
            return
        if path == "/api/bootstrap":
            try:
                self._json(200, bootstrap_payload())
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
            return
        if path == "/api/state":
            try:
                self._json(200, {"ok": True, "state": get_state()})
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
            return
        if path in {"/shared/calc", "/shared/calc/", "/shared/calc/index.html"}:
            try:
                raw = build_teacher_agenda(get_state()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                self.send_error(500, f"Could not build co-teacher preview: {exc}")
            return
        if path in {"/shared", "/shared/", "/shared/index.html"}:
            try:
                self._serve_shared_file("/shared/index.html")
            except Exception:
                self.send_error(404, "Teacher Shared home not found")
            return
        if path.startswith("/shared/"):
            try:
                self._serve_shared_file(path)
            except Exception:
                self.send_error(404, "Shared teacher file not found")
            return
        if path.startswith("/apcalc/"):
            try:
                self._serve_apcalc_file(path)
            except Exception:
                self.send_error(404, "AP Calculus AB file not found")
            return
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/index.html")
            self.end_headers()
            return
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/warmups/import":
                query = parse_qs(urlparse(self.path).query)
                week = int((query.get("week") or ["0"])[0] or 0)
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length <= 0 or length > WARMUP_RESULT_MAX_BYTES:
                    raise ValueError("Invalid warmup result upload size.")
                raw = self.rfile.read(length)
                result = import_warmup_result(raw, week or None)
                self._json(200, result)
                return
            payload = self._read_json()
            state = payload.get("state") if isinstance(payload, dict) else None
            if path == "/api/warmups/editor/save":
                result = warmup_editor.save((payload or {}).get("config"), config_root=WARMUP_CONFIG_ROOT, library_root=LIBRARY_ROOT, apcalc_root=APCALC_ROOT, public_base=PUBLIC_BASE, publish=False, history_path=WARMUP_HISTORY_PATH)
                self._json(200, result)
                return
            if path == "/api/warmups/editor/publish":
                current_state = get_state()
                result = warmup_editor.save((payload or {}).get("config"), config_root=WARMUP_CONFIG_ROOT, library_root=LIBRARY_ROOT, apcalc_root=APCALC_ROOT, public_base=PUBLIC_BASE, publish=True, agenda_path=AGENDA_PATH, agenda_html=build_student_agenda(current_state), history_path=WARMUP_HISTORY_PATH)
                self._json(200, result)
                return
            if path == "/api/exercises/save":
                result = save_exercise_config((payload or {}).get("config"), publish=False)
                self._json(200, result)
                return
            if path == "/api/exercises/publish":
                result = save_exercise_config((payload or {}).get("config"), publish=True)
                self._json(200, result)
                return
            if path == "/api/git-pull-all":
                result = git_pull_all()
                self._json(200 if result.get("ok") else 409, result)
                return
            if path == "/api/git-push-all":
                result = git_push_all()
                self._json(200 if result.get("ok") else 409, result)
                return
            if path == "/api/warmups/request":
                week = int((payload or {}).get("week_number") or 0)
                target, request = create_warmup_request_zip(get_state(), week)
                self._json(200, {
                    "ok": True,
                    "week_number": week,
                    "plan": request["plan"],
                    "request_id": request["request_id"],
                    "download_url": "/downloads/" + quote(target.name),
                    "filename": target.name,
                    "message": f"Week {week} warmup request is ready. Download it and upload that ZIP to ChatGPT.",
                })
                return
            if path == "/api/publish-curriculum":
                written = publish_local_curriculum_to_repo()
                state = get_state()
                atomic_write_text(AGENDA_PATH, build_student_agenda(state))
                self._json(200, {
                    "ok": True,
                    "state": state,
                    "written": written,
                    "message": f"Published local Calculus curriculum and updated the Student Agenda. {len(written)} file(s) changed in the apcalc repository. Review in GitHub Desktop, then sync/push when ready.",
                })
                return
            if path == "/api/save":
                state = save_state(state)
                self._json(
                    200,
                    {"ok": True, "state": state, "message": "Planner state saved."},
                )
                return
            if path in {"/api/update-agenda", "/api/publish"}:
                state = save_state(state)
                atomic_write_text(AGENDA_PATH, build_student_agenda(state))
                self._json(
                    200,
                    {
                        "ok": True,
                        "state": state,
                        "message": "Planner saved and Student Agenda updated locally. Use GitHub Sync when ready.",
                        "student_path": str(AGENDA_PATH),
                    },
                )
                return
            self._json(404, {"ok": False, "error": "Unknown API endpoint"})
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})


def main():
    ensure_teacher_shared_layout()
    ensure_calc_tools_layout()
    ensure_warmup_layout()
    ensure_exercise_builder_layout()
    # Keep the shared runtime launcher aligned with the code actually running.
    atomic_write_text(TEACHER_TOOLS_ROOT / "runtime" / "planner_version.txt", PLANNER_VERSION + "\n")
    if not APCALC_ROOT.is_dir():
        raise SystemExit(f"AP Calculus AB repository not found at {APCALC_ROOT}")
    if not AGENDA_PATH.is_file():
        raise SystemExit(f"Student agenda not found at {AGENDA_PATH}")

    handler = lambda *args, **kwargs: PlannerHandler(
        *args, directory=str(TEACHER_TOOLS_ROOT), **kwargs
    )
    server = ThreadingHTTPServer((HOST, PORT), handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"AP Calculus Tools v{PLANNER_VERSION}")
    print("===============")
    print(f"Teacher Tools root: {TEACHER_TOOLS_ROOT}")
    print(f"AP Calculus AB repo:        {APCALC_ROOT}")
    print(f"Teacher shared repo: {SHARED_ROOT}")
    print(f"AP Calculus AB shared path: {SHARED_CALC_ROOT}")
    print(f"Planner:             {url}")
    print("Press Control-C to stop.")
    if os.environ.get("PLANNER_NO_BROWSER") != "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
