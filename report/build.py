"""Fill report/report_src.md with numbers from results/summary.json -> report/report.md.

Placeholders look like {{mspacman_s0.best.mean:.1f}}: a dotted path into summary.json and an
optional Python format spec. A missing key stops the build, so the PDF can never contain a
number that is not in a result file.
"""
import json
import re
import sys

S = json.load(open("results/summary.json"))


def lookup(path):
    cur = S
    for part in path.split("|"):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def sub(m):
    path, fmt = m.group(1).strip(), m.group(2)
    try:
        v = lookup(path)
    except (KeyError, IndexError, TypeError):
        sys.exit(f"report placeholder not found in summary.json: {path}")
    return format(v, fmt) if fmt else str(v)


src = open("report/report_src.md").read()
out = re.sub(r"\{\{([^}:]+?)(?::([^}]+))?\}\}", sub, src)
open("report/report.md", "w").write(out)
print("wrote report/report.md")
