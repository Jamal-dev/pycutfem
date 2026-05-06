"""Write a compact pytest JUnit summary to the GitHub Actions summary."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def _as_int(value: Optional[str]) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _as_float(value: Optional[str]) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: junit_summary.py JUNIT_XML TITLE", file=sys.stderr)
        return 2

    junit_path = Path(sys.argv[1])
    title = sys.argv[2]

    if not junit_path.exists():
        message = f"### {title}\n\nNo JUnit XML file was produced at `{junit_path}`.\n"
    else:
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        if not suites:
            suites = root.findall(".//testsuite")

        tests = sum(_as_int(suite.get("tests")) for suite in suites)
        failures = sum(_as_int(suite.get("failures")) for suite in suites)
        errors = sum(_as_int(suite.get("errors")) for suite in suites)
        skipped = sum(_as_int(suite.get("skipped")) for suite in suites)
        elapsed = sum(_as_float(suite.get("time")) for suite in suites)
        passed = max(tests - failures - errors - skipped, 0)

        message = "\n".join(
            [
                f"### {title}",
                "",
                "| Tests | Passed | Failed | Errors | Skipped | Time (s) |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| {tests} | {passed} | {failures} | {errors} | {skipped} | {elapsed:.2f} |",
                "",
            ]
        )

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(message)
    else:
        print(message)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
