"""Run the whole pipeline: clean -> weather -> train/evaluate -> water-level risk.

Run:  python scripts/run_all.py            (full)
      python scripts/run_all.py --quick    (fast smoke run)
      python scripts/run_all.py --no-weather   (offline: weather features become missing)
"""

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    print(f"\n=== {script} {' '.join(args)}")
    sys.argv = [str(HERE / script), *args]
    runpy.run_path(str(HERE / script), run_name="__main__")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-weather", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    run("01_clean_data.py")
    if not a.no_weather:
        run("02_fetch_weather.py")
    run("03_train_evaluate.py", *(["--quick"] if a.quick else []))
    run("05_water_level_risk.py")
