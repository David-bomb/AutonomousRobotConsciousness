"""Check UTF-8 and local inline Markdown links without third-party packages.

Run from any directory: python /path/to/project/scripts/check_docs.py
External URLs, Markdown reference links and fragment anchors are not validated.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\s*\)")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def documents() -> list[Path]:
    roots = [ROOT / name for name in ("README.md", "AGENTS.md", "DOCS.md")]
    nested = []
    for directory in ("docs", "Articles"):
        nested.extend(
            p for p in (ROOT / directory).rglob("*.md")
            if not any(part in {".git", "node_modules", ".venv"} for part in p.relative_to(ROOT).parts)
        )
    return sorted(set(roots + nested))


def prose_lines(text: str):
    """Ignore fenced examples so placeholder paths are not treated as links."""
    fence_char = None
    fence_size = 0
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        fence = re.match(r"(`{3,}|~{3,})", stripped)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char = None
            continue
        if fence_char is None:
            yield number, line


def main() -> int:
    errors = []
    checked_links = 0
    files = documents()
    for path in files:
        display = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{display}: cannot read UTF-8: {exc}")
            continue
        if not text.strip():
            errors.append(f"{display}: empty document")
        if "\ufffd" in text:
            errors.append(f"{display}: Unicode replacement character")
        for number, line in prose_lines(text):
            for match in LINK.finditer(line):
                target = match.group(1).strip("<>")
                if SCHEME.match(target) or target.startswith(("//", "#")):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not target:
                    continue
                checked_links += 1
                if not (path.parent / target).exists():
                    errors.append(f"{display}:{number}: missing target {target}")
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    print(f"Checked {len(files)} Markdown files and {checked_links} local links; {len(errors)} errors.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
