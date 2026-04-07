from __future__ import annotations

from examples.NIRB.example2_workflow import run_example2


if __name__ == "__main__":
    result = run_example2()
    print(result["artifacts"]["results"])
