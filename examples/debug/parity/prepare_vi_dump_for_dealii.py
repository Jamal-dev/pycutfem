from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp  # type: ignore


def _write_vector(path: Path, vec: np.ndarray) -> None:
    arr = np.asarray(vec, dtype=float).ravel()
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{int(arr.size)}\n")
        for val in arr.tolist():
            fh.write(f"{float(val):.17e}\n")


def _write_triplets(path: Path, mat: sp.spmatrix) -> None:
    coo = mat.tocoo()
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{int(coo.shape[0])} {int(coo.shape[1])} {int(coo.nnz)}\n")
        for r, c, v in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
            fh.write(f"{int(r)} {int(c)} {float(v):.17e}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a pycutfem VI dump into simple text files for the deal.II parity harness.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--stem", type=str, default="step0001_it01")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    stem = str(args.stem)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = sp.load_npz(dump_dir / f"{stem}_A_red.npz").tocsr()
    eq_B = np.asarray(np.load(dump_dir / f"{stem}_eq_B_red.npy"), dtype=float)
    eq_B_sparse = sp.csr_matrix(eq_B)

    _write_triplets(out_dir / "A_red.triplets", A)
    _write_triplets(out_dir / "B_red.triplets", eq_B_sparse)
    _write_vector(out_dir / "x_red.txt", np.load(dump_dir / f"{stem}_x_red.npy"))
    _write_vector(out_dir / "lo_red.txt", np.load(dump_dir / f"{stem}_lo_red.npy"))
    _write_vector(out_dir / "hi_red.txt", np.load(dump_dir / f"{stem}_hi_red.npy"))
    _write_vector(out_dir / "c_red.txt", np.load(dump_dir / f"{stem}_c_red.npy"))
    _write_vector(out_dir / "stat_red.txt", np.load(dump_dir / f"{stem}_stat_red.npy"))
    _write_vector(out_dir / "eq_b_eff.txt", np.load(dump_dir / f"{stem}_eq_b_eff.npy"))
    _write_vector(out_dir / "eq_lambda.txt", np.load(dump_dir / f"{stem}_eq_lambda.npy"))
    _write_vector(out_dir / "act_lo.txt", np.load(dump_dir / f"{stem}_act_lo.npy"))
    _write_vector(out_dir / "act_hi.txt", np.load(dump_dir / f"{stem}_act_hi.npy"))

    meta = json.loads((dump_dir / f"{stem}_meta.json").read_text(encoding="utf-8"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
