from __future__ import annotations

from examples.NIRB.example1_workflow import run_example1


if __name__ == "__main__":
    result = run_example1()
    print(result["artifacts"]["results"])
