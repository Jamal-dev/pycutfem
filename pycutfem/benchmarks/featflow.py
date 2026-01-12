from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import urllib.request
import zipfile

import numpy as np


FEATFLOW_BASE = "https://wwwold.mathematik.tu-dortmund.de/~featflow/media/dfg_bench3_2d"
DRAGLIFT_ZIP = f"{FEATFLOW_BASE}/draglift_q2_cn_lv1-6_dt4.zip"
PRESSURE_ZIP = f"{FEATFLOW_BASE}/pressure_q2_cn_lv1-6_dt4.zip"


def default_cache_dir() -> Path:
    root = os.getenv("PYCUTFEM_REF_DIR", None)
    if root:
        return Path(root).expanduser()
    return Path("~/.cache/pycutfem_reference/featflow").expanduser()


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    urllib.request.urlretrieve(url, tmp)  # nosec - trusted public URL
    tmp.replace(dest)
    return dest


def ensure_featflow_zips(*, cache_dir: Path | None = None) -> tuple[Path, Path]:
    cache_dir = cache_dir or default_cache_dir()
    drag_zip = _download(DRAGLIFT_ZIP, Path(cache_dir) / Path("draglift_q2_cn_lv1-6_dt4.zip"))
    pres_zip = _download(PRESSURE_ZIP, Path(cache_dir) / Path("pressure_q2_cn_lv1-6_dt4.zip"))
    return drag_zip, pres_zip


@dataclass(frozen=True)
class DragLiftReference:
    time: np.ndarray
    cd: np.ndarray
    cl: np.ndarray


@dataclass(frozen=True)
class PressureReference:
    time: np.ndarray
    p_a: np.ndarray
    p_b: np.ndarray
    dp: np.ndarray


def load_draglift(zip_path: Path, *, level: int) -> DragLiftReference:
    if int(level) not in range(1, 7):
        raise ValueError("FeatFlow level must be 1..6.")
    member = f"bdforces_lv{int(level)}"
    time: list[float] = []
    cd: list[float] = []
    cl: list[float] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                time.append(float(parts[1]))
                cd.append(float(parts[3]))
                cl.append(float(parts[4]))
    return DragLiftReference(
        time=np.asarray(time, dtype=float),
        cd=np.asarray(cd, dtype=float),
        cl=np.asarray(cl, dtype=float),
    )


def load_pressure_points(zip_path: Path, *, level: int) -> PressureReference:
    if int(level) not in range(1, 7):
        raise ValueError("FeatFlow level must be 1..6.")
    member = f"pointvalues_lv{int(level)}"
    time: list[float] = []
    p_a: list[float] = []
    p_b: list[float] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # Format: step time  x1 y1 type1 deriv1 val1  x2 y2 type2 deriv2 val2
                if len(parts) < 12:
                    continue
                time.append(float(parts[1]))
                # Two probe points per line:
                # [x1 y1 type1 deriv1 val1 x2 y2 type2 deriv2 val2]
                p_a.append(float(parts[6]))
                p_b.append(float(parts[11]))
    t = np.asarray(time, dtype=float)
    pa = np.asarray(p_a, dtype=float)
    pb = np.asarray(p_b, dtype=float)
    return PressureReference(time=t, p_a=pa, p_b=pb, dp=(pa - pb))


def compare_timeseries(
    *,
    ref_time: np.ndarray,
    ref_val: np.ndarray,
    sim_time: np.ndarray,
    sim_val: np.ndarray,
) -> dict[str, float]:
    """
    Compare two time series on the overlapping time window using linear interpolation of the
    simulated values onto the reference time grid.
    """
    ref_time = np.asarray(ref_time, dtype=float)
    ref_val = np.asarray(ref_val, dtype=float)
    sim_time = np.asarray(sim_time, dtype=float)
    sim_val = np.asarray(sim_val, dtype=float)
    if ref_time.size == 0 or sim_time.size == 0:
        raise ValueError("Empty time series.")

    t0 = max(float(ref_time.min()), float(sim_time.min()))
    t1 = min(float(ref_time.max()), float(sim_time.max()))
    if not (t1 > t0):
        raise ValueError("No overlapping time interval.")

    mask = (ref_time >= t0) & (ref_time <= t1)
    t = ref_time[mask]
    r = ref_val[mask]
    s = np.interp(t, sim_time, sim_val)
    diff = s - r
    return {
        "t0": float(t0),
        "t1": float(t1),
        "n": int(t.size),
        "max_abs": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }
