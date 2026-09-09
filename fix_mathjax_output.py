#!/usr/bin/env python3
"""Mechanical MathJax repair + audit for curriculum outputs.

Stable canonical replacement for the old _fix_mathjax_v11.py.

Key rule: preserve the file's existing math delimiter contract. Current Notes use
\\(...\\) and \\[...\\]. Legacy files that already use $...$ / $$...$$ are also
supported when explicitly configured in rendered HTML. Non-HTML files and
embedded source code are excluded. This tool never converts delimiter families.

Usage:
    python3 Tools/fix_mathjax_output.py <file-or-folder>

Exit codes:
    0 = repair/audit PASS
    2 = one or more audit warnings remain
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.cwd()
# Only rendered HTML is supported. Source files and serialized records need
# their own schema-aware validators, never heuristic TeX rewriting.
SUPPORTED = {".html"}

BYTE_FIXES = [
    (b"\x0crac{", b"\\frac{"),
    (b"\x0corall", b"\\forall"),
    (b"\x08egin{", b"\\begin{"),
    (b"\x09heta", b"\\theta"),
    (b"\x09ext{", b"\\text{"),
    (b"\x09imes", b"\\times"),
    (b"\x0dight", b"\\right"),
    (b"\x0dangle", b"\\rangle"),
    (b"\x07lpha", b"\\alpha"),
    (b"\x07ngle", b"\\angle"),
    (b"\x0bec{", b"\\vec{"),
    (b"\x0bee", b"\\vee"),
    (b"\x0bdots", b"\\vdots"),
    (b"\\rac{", b"\\frac{"),
    (b"\\ight", b"\\right"),
]

KNOWN_TEX_COMMANDS = (
    "frac", "dfrac", "tfrac", "sqrt", "left", "right", "text", "theta",
    "alpha", "beta", "gamma", "delta", "lambda", "mu", "pi", "sigma",
    "times", "cdot", "div", "to", "rightarrow", "leftarrow",
    "leftrightarrow", "infty", "pm", "mp", "le", "ge", "lt", "gt", "ne",
    "neq", "approx", "sim", "equiv", "forall", "exists", "angle", "rangle",
    "langle", "vec", "overline", "underline", "hat", "bar", "sin", "cos",
    "tan", "log", "ln", "exp", "min", "max", "lim", "sum", "prod",
    "begin", "end", "qquad", "quad"
)
DOUBLED_KNOWN_TEX_RE = re.compile(
    r"\\\\(?=(?:" + "|".join(map(re.escape, KNOWN_TEX_COMMANDS)) + r")\b)"
)
MERGED_RELATION_RE = re.compile(
    r"\\(?P<rel>lt|gt|le|ge|ne)(?P<var>[a-pr-zA-PR-Z])"
    r"(?=(?:\^|_|[+\-*/=<>),.;:\]}]|\\|$|\s))"
)
BARE_RAC_RE = re.compile(r"(?<![A-Za-z\\])rac\{")
LEFT_GLYPH_TAIL_RE = re.compile(r"(?:≤|&le;|&#8804;)\s*ft(?=\s*[({\[])")
RIGHT_GLYPH_TAIL_RE = re.compile(r"(?:≥|&ge;|&#8805;)\s*ight(?=\s*[)}\]])")
MULTIROW_ENV_RE = re.compile(
    r"\\begin\{(?:cases|aligned|alignedat|array|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|gathered)\}"
)
CORRUPTED_SYSTEM_ROW_RE = re.compile(
    r"(?<!\\)\\(?P<var>[A-Za-z])(?=\s*(?:[-+*/=]|\\(?:le|ge|lt|gt|ne|neq)\b))"
)

# Four delimiter families. Order matters: display before inline, current before legacy.
MATH_BLOCK_RE = re.compile(
    r"(?P<bracket>\\\[(?P<bracket_body>.*?)\\\])"
    r"|(?P<paren>\\\((?P<paren_body>.*?)\\\))"
    r"|(?P<dollar_display>(?<!\\)\$\$(?P<dollar_display_body>.*?)(?<!\\)\$\$)"
    r"|(?P<dollar_inline>(?<!\\)\$(?!\$)(?P<dollar_inline_body>.*?)(?<!\\)\$(?!\$))",
    re.DOTALL,
)

PROTECTED_HTML_RE = re.compile(
    r"(<!--.*?-->|<script\b[^>]*>.*?</script\s*>|<style\b[^>]*>.*?</style\s*>"
    r"|<pre\b[^>]*>.*?</pre\s*>|<code\b[^>]*>.*?</code\s*>|<textarea\b[^>]*>.*?</textarea\s*>"
    r"|<![^>]*>|</?[A-Za-z][A-Za-z0-9:-]*(?:\s+(?:[^>\"']|\"[^\"]*\"|'[^']*')*)?\s*/?>)", re.I | re.S
)


def split_math(match):
    if match.group("bracket") is not None:
        return r"\[", match.group("bracket_body"), r"\]"
    if match.group("paren") is not None:
        return r"\(", match.group("paren_body"), r"\)"
    if match.group("dollar_display") is not None:
        return "$$", match.group("dollar_display_body"), "$$"
    return "$", match.group("dollar_inline_body"), "$"


def repair_math_content(content: str):
    hits = 0
    # Repairs are confined to a real math span, never scripts/attributes/prose.
    for bad, good in BYTE_FIXES:
        bad_text, good_text = bad.decode("utf-8"), good.decode("utf-8")
        hits += content.count(bad_text)
        content = content.replace(bad_text, good_text)

    if MULTIROW_ENV_RE.search(content):
        def row_repl(m):
            nonlocal hits
            hits += 1
            return r"\\ " + m.group("var")
        content = CORRUPTED_SYSTEM_ROW_RE.sub(row_repl, content)

    for pat, repl in [
        (LEFT_GLYPH_TAIL_RE, r"\\left"),
        (RIGHT_GLYPH_TAIL_RE, r"\\right"),
    ]:
        n = len(pat.findall(content))
        if n:
            content = pat.sub(repl, content)
            hits += n

    # High-confidence control-character command tails that survived decoding.
    tail_fixes = [
        (re.compile("\theta"), r"\\theta"),
        (re.compile("\text\\{"), r"\\text{"),
        (re.compile("\times"), r"\\times"),
        (re.compile(r"\to(?=[0-9A-Za-z({\[+\-])"), r"\\to "),
        (re.compile(r"\r?\nightarrow"), r"\\rightarrow"),
        (re.compile(r"\r?\nangle"), r"\\rangle"),
    ]
    for pat, repl in tail_fixes:
        n = len(pat.findall(content))
        if n:
            content = pat.sub(repl, content)
            hits += n

    n = len(DOUBLED_KNOWN_TEX_RE.findall(content))
    if n:
        content = DOUBLED_KNOWN_TEX_RE.sub(lambda _: "\\", content)
        hits += n

    n = len(BARE_RAC_RE.findall(content))
    if n:
        content = BARE_RAC_RE.sub(r"\\frac{", content)
        hits += n

    def rel_repl(m):
        nonlocal hits
        hits += 1
        return "\\" + m.group("rel") + " " + m.group("var")
    content = MERGED_RELATION_RE.sub(rel_repl, content)

    # HTML-safe relations inside math.
    for src, dst in [
        ("&lt;", r"\lt "), ("&#60;", r"\lt "),
        ("&gt;", r"\gt "), ("&#62;", r"\gt "),
    ]:
        n = content.count(src)
        if n:
            content = content.replace(src, dst)
            hits += n
    n = content.count("<")
    if n:
        content = content.replace("<", r"\lt ")
        hits += n
    n = content.count(">")
    if n:
        content = content.replace(">", r"\gt ")
        hits += n

    return content, hits


def repair_math_blocks(text: str, allow_dollar: bool = False, allow_display: bool = False):
    total = 0
    def repl(match):
        nonlocal total
        open_delim, body, close_delim = split_math(match)
        if (open_delim == "$" and not allow_dollar) or (open_delim == "$$" and not allow_display):
            return match.group(0)
        body, n = repair_math_content(body)
        total += n
        return open_delim + body + close_delim
    # Disabled dollar syntax must not consume a span containing real current
    # delimiters (for example currency surrounding a fraction).
    pattern = MATH_BLOCK_RE.pattern
    if not allow_dollar:
        pattern = pattern.replace("|(?P<dollar_inline>", "|(?!)" + "(?P<dollar_inline>")
    if not allow_display:
        pattern = pattern.replace("|(?P<dollar_display>", "|(?!)" + "(?P<dollar_display>")
    return re.sub(pattern, repl, text, flags=re.DOTALL), total


def content_without_script_style(text: str) -> str:
    return PROTECTED_HTML_RE.sub("\n", text)


def delimiter_usage(text: str):
    visible = content_without_script_style(text)
    return {
        "paren": bool(re.search(r"\\\(.*?\\\)", visible, re.S)),
        "bracket": bool(re.search(r"\\\[.*?\\\]", visible, re.S)),
        "dollar_inline": bool(re.search(r"(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)", visible, re.S)),
        "dollar_display": bool(re.search(r"(?<!\\)\$\$.*?(?<!\\)\$\$", visible, re.S)),
    }


def configuration_spans(text: str):
    for script in re.finditer(r"<script\b[^>]*>(.*?)</script\s*>", text, re.I | re.S):
        assignment = re.search(r"(?:^|[;\n])\s*window\.MathJax\s*=\s*\{", script.group(1))
        if assignment:
            yield script.start(1) + assignment.start() + assignment.group().index("window.MathJax"), script.end(1)


def config_support(text: str):
    # Source examples and attributes cannot opt a page into dollar math.
    text = "\n".join(text[start:end] for start, end in configuration_spans(text))
    return {
        "paren": bool(re.search(r"inlineMath\s*:\s*\[[^\]]*\\\\\([^\]]*\\\\\)", text, re.S)),
        "bracket": bool(re.search(r"displayMath\s*:\s*\[[^\]]*\\\\\[[^\]]*\\\\\]", text, re.S)),
        "dollar_inline": bool(re.search(r"inlineMath\s*:\s*\[[^\]]*['\"]\$['\"]", text, re.S)),
        "dollar_display": bool(re.search(r"displayMath\s*:\s*\[[^\]]*['\"]\$\$['\"]", text, re.S)),
    }


def repair_compact_current_config(text: str):
    """Repair an actual delimiter config only, never a quoted source example."""
    usage = delimiter_usage(text)
    support = config_support(text)
    if (usage["paren"] and not support["paren"]) or (usage["bracket"] and not support["bracket"]):
        pat = re.compile(r"\s*window\.MathJax\s*=\s*\{\s*tex\s*:\s*\{.*?\}\s*\}\s*;?", re.S)
        for start, end in configuration_spans(text):
            match = pat.match(text, start, end)
            if match:
                repl = r'window.MathJax={tex:{inlineMath:[["\\(","\\)"]],displayMath:[["\\[","\\]"]],processEscapes:true}};'
                return text[:start] + repl + text[match.end():], 1
    return text, 0


def audit(path: Path, text: str):
    issues = []
    visible = content_without_script_style(text)

    # Current delimiter balance.
    if visible.count(r"\(") != visible.count(r"\)"):
        issues.append("unbalanced \\( ... \\) delimiters")
    if visible.count(r"\[") != visible.count(r"\]"):
        issues.append("unbalanced \\[ ... \\] delimiters")

    usage = delimiter_usage(text)
    support = config_support(text) if path.suffix.lower() == ".html" else {"dollar_inline": True, "dollar_display": True}

    # Only treat dollar signs as math delimiters when the HTML config actually
    # enables dollar math. Current Notes intentionally do not, so $5 is prose.
    if support.get("dollar_inline"):
        dollars = len(re.findall(r"(?<!\\)\$(?!\$)", visible.replace("$$", "")))
        if dollars % 2:
            issues.append("odd unescaped $ delimiter count")
    if support.get("dollar_display"):
        displays = len(re.findall(r"(?<!\\)\$\$", visible))
        if displays % 2:
            issues.append("odd unescaped $$ delimiter count")

    if list(configuration_spans(text)):
        for key, label in [
            ("paren", r"\\(...\\)"),
            ("bracket", r"\\[...\\]"),
            ("dollar_inline", "$...$"),
            ("dollar_display", "$$...$$"),
        ]:
            if key.startswith("dollar_") and not support[key]:
                continue
            if usage[key] and not support[key]:
                issues.append(f"MathJax config does not enable used delimiter {label}")

    # Broken command tails that should never leave final output.
    for desc, pat in [
        ("dropped frac command", re.compile(r"(?<![A-Za-z\\])rac\{")),
        ("corrupted left command", LEFT_GLYPH_TAIL_RE),
        ("corrupted right command", RIGHT_GLYPH_TAIL_RE),
    ]:
        if pat.search(visible):
            issues.append(desc)

    # Control chars other than newline/tab/carriage return.
    bad_controls = [ord(ch) for ch in visible if ord(ch) < 32 and ch not in "\n\r\t"]
    if bad_controls:
        issues.append("unexpected control character remains")

    if "mathjax" in text.lower() and ("cdnjs" in text or "unpkg" in text):
        issues.append("noncanonical MathJax CDN")
    config_pos = text.find("window.MathJax")
    cdn_pos = text.find("cdn.jsdelivr.net")
    if config_pos != -1 and cdn_pos != -1 and config_pos > cdn_pos:
        issues.append("MathJax config appears after CDN script")

    return issues


def process(path: Path):
    if path.suffix.lower() not in SUPPORTED:
        return False, 0, []
    original = path.read_bytes().decode("utf-8")
    support = config_support(original)
    # Split, transform only rendered text, then rejoin source spans verbatim.
    chunks = PROTECTED_HTML_RE.split(original)
    repair_count = 0
    for i in range(0, len(chunks), 2):
        chunks[i], count = repair_math_blocks(
            chunks[i], allow_dollar=support["dollar_inline"],
            allow_display=support["dollar_display"],
        )
        repair_count += count
    text = "".join(chunks)
    text, count = repair_compact_current_config(text)
    repair_count += count
    changed = text != original
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed, repair_count, audit(path, text)


def main():
    if not TARGET.exists():
        print(f"FAIL missing finalization target: {TARGET}")
        return 2
    if TARGET.is_file():
        files = [TARGET] if TARGET.suffix.lower() in SUPPORTED else []
    else:
        files = sorted(
            p for p in TARGET.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED
            and "__MACOSX" not in p.parts and not p.name.startswith("._")
        )

    if not files:
        print(f"No supported files found: {TARGET}")
        return 0

    changed_files = 0
    warning_files = 0
    total_repairs = 0
    for path in files:
        changed, repairs, issues = process(path)
        total_repairs += repairs
        if changed:
            changed_files += 1
            print(f"FIXED {path}")
        if issues:
            warning_files += 1
            print(f"FAIL  {path}")
            for issue in issues:
                print(f"  - {issue}")

    print(f"MathJax finalizer: {len(files)} file(s), {changed_files} changed, {total_repairs} repair(s), {warning_files} warning file(s).")
    return 2 if warning_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
