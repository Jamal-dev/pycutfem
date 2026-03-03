"""
Assemble a stable, paper-facing set of validation figures for Duddu (2007, 2009).

This script does **not** rerun any simulations. It only:
  - copies key PNGs from existing benchmark output folders,
  - builds simple montages (paper vs our reproduction),
  - writes a small JSON/CSV summary of quantitative metrics.

Outputs are written under:
  examples/biofilms/benchmarks/dadu/validation_figs/

Run with the FEniCSX environment (matplotlib available):

  conda run -n fenicsx python examples/biofilms/benchmarks/dadu/make_validation_figs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r if row]


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _montage_rows(
    *,
    out_png: Path,
    rows: list[list[Path]],
    titles: list[list[str]] | None = None,
    dpi: int = 200,
) -> None:
    """
    Create a simple montage with matplotlib, given a grid of image paths.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"matplotlib is required to build montages ({exc}).")

    nrows = len(rows)
    ncols = max((len(r) for r in rows), default=0)
    if nrows == 0 or ncols == 0:
        raise ValueError("Empty montage grid.")

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 2.4 * nrows),
        constrained_layout=True,
    )
    if nrows == 1 and ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = [list(axs)]
    elif ncols == 1:
        axs = [[a] for a in axs]

    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i][j]
            ax.axis("off")
            if j >= len(rows[i]):
                continue
            img = plt.imread(str(rows[i][j]))
            ax.imshow(img)
            if titles is not None and i < len(titles) and j < len(titles[i]):
                tt = str(titles[i][j])
                if tt:
                    ax.set_title(tt, fontsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def _read_timeseries_xy(csv_path: Path, *, x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            try:
                xs.append(float(row[x_key]))
                ys.append(float(row[y_key]))
            except Exception:
                continue
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble paper-facing validation figures (Duddu 2007/2009).")
    ap.add_argument(
        "--outdir",
        type=str,
        default="examples/biofilms/benchmarks/dadu/validation_figs",
        help="Output directory for copied/montaged figures.",
    )
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Duddu (2007): 1D slab
    # --------------------------
    fig4_src = here / "results" / "duddu2007_fig4_table1" / "fig4_compare.png"
    if fig4_src.exists():
        _copy(fig4_src, outdir / "duddu2007_fig4_compare.png")

    table1_src = here / "results" / "duddu2007_fig4_table1" / "summary.csv"
    one_domain_slab_src = here / "results" / "duddu2007_one_domain_slab_speed_table1_match" / "summary.json"

    # --------------------------
    # Duddu (2007): 2D Fig.6 Example 2
    # --------------------------
    fig6_xfem_src = here / "results" / "duddu2007_fig6_example2_paper_match_qp_m20g200" / "compare_fig6.png"
    if fig6_xfem_src.exists():
        _copy(fig6_xfem_src, outdir / "duddu2007_fig6_paper_vs_xfem.png")

    one_domain_run = (
        here
        / "results"
        / "_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx80_dt0p2_q2_t28p6"
    )
    fig6_one_domain_src = one_domain_run / "fig6a_interface.png"
    ytop_src = one_domain_run / "y_top_compare.png"
    if fig6_one_domain_src.exists():
        _copy(fig6_one_domain_src, outdir / "duddu2007_fig6_one_domain_interface.png")
    if ytop_src.exists():
        _copy(ytop_src, outdir / "duddu2007_fig6_y_top_compare.png")

    # Growth mesh study (nx60 vs nx80) using the same reference XFEM y_top time series.
    one_domain_run_nx60 = (
        here
        / "results"
        / "_tune_oneDom_phiMix_alphaMix_DS120_vSextL2_gvS0p1_acM1_g5e-4_mobdeg_floor0p1_nx60_dt0p2_q2_t28p6"
    )
    xfem_ref = here / "results" / "duddu2007_fig6_example2_paper_match_qp_m20g200_y_top"
    xfem_ts = xfem_ref / "y_top_timeseries.csv"
    od60_ts = one_domain_run_nx60 / "y_top_timeseries.csv"
    od80_ts = one_domain_run / "y_top_timeseries.csv"
    if xfem_ts.exists() and od60_ts.exists() and od80_ts.exists():
        try:
            import matplotlib.pyplot as plt  # type: ignore

            tx, yx = _read_timeseries_xy(xfem_ts, x_key="t_days", y_key="y_top_mm")
            t60, y60 = _read_timeseries_xy(od60_ts, x_key="t_days", y_key="y_top_mm")
            t80, y80 = _read_timeseries_xy(od80_ts, x_key="t_days", y_key="y_top_mm")

            fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
            ax.plot(tx, yx, "-", lw=2.0, label="XFEM (reference)")
            ax.plot(t60, y60, "-", lw=1.5, label="one-domain (60×60)")
            ax.plot(t80, y80, "-", lw=1.5, label="one-domain (80×80)")
            ax.set_xlabel("t (days)")
            ax.set_ylabel(r"$y_{top}$ (mm)")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, loc="best")
            fig.savefig(outdir / "duddu2007_fig6_mesh_study_y_top.png", dpi=int(args.dpi))
            plt.close(fig)
        except Exception:
            pass

    # --------------------------
    # Duddu (2009): 1D mapping
    # --------------------------
    det1d_plot_src = here / "results" / "duddu2009_detachment_1d" / "plot_1d_thickness.png"
    if det1d_plot_src.exists():
        _copy(det1d_plot_src, outdir / "duddu2009_detachment_1d_thickness.png")

    # --------------------------
    # Duddu (2009): 2D Fig.6
    # --------------------------
    # Paper reference panels are stored as separate images; montage them into a single row.
    paper_a = here / "paper_figs" / "dadu_figure_6_a.png"
    paper_b = here / "paper_figs" / "dadu_figure_6_b.png"
    paper_c = here / "paper_figs" / "dadu_figure_6_c.png"
    if paper_a.exists() and paper_b.exists() and paper_c.exists():
        _montage_rows(
            out_png=outdir / "duddu2009_fig6_paper.png",
            rows=[[paper_a, paper_b, paper_c]],
            titles=[["shear", "height ($l^2$)", "poly"]],
            dpi=int(args.dpi),
        )

    fig6_2d_base = here / "results" / "fig6_noac_units_mu1_t7_ny30"
    fig6_2d_iface = fig6_2d_base / "fig6_noac_units_mu1_t7_ny30_interface_only.png"
    fig6_2d_thick = fig6_2d_base / "fig6_noac_units_mu1_t7_ny30_thickness_profiles.png"
    if fig6_2d_iface.exists():
        _copy(fig6_2d_iface, outdir / "duddu2009_fig6_one_domain_interface_ny30.png")
    if fig6_2d_thick.exists():
        _copy(fig6_2d_thick, outdir / "duddu2009_fig6_one_domain_thickness_ny30.png")

    # Optional: coarse-mesh run (only if present)
    fig6_2d_coarse = here / "results" / "fig6_noac_units_mu1_t7_nx60_ny15"
    iface_coarse = fig6_2d_coarse / "fig6_noac_units_mu1_t7_nx60_ny15_interface_only.png"
    thick_coarse = fig6_2d_coarse / "fig6_noac_units_mu1_t7_nx60_ny15_thickness_profiles.png"
    if iface_coarse.exists():
        _copy(iface_coarse, outdir / "duddu2009_fig6_one_domain_interface_ny15.png")
    if thick_coarse.exists():
        _copy(thick_coarse, outdir / "duddu2009_fig6_one_domain_thickness_ny15.png")

    # Detachment mesh sensitivity montage (interface-only panels)
    fine_iface = outdir / "duddu2009_fig6_one_domain_interface_ny30.png"
    coarse_iface = outdir / "duddu2009_fig6_one_domain_interface_ny15.png"
    if fine_iface.exists() and coarse_iface.exists():
        _montage_rows(
            out_png=outdir / "duddu2009_fig6_mesh_sensitivity.png",
            rows=[[fine_iface, coarse_iface]],
            titles=[["baseline (120×30)", "coarse (60×15)"]],
            dpi=int(args.dpi),
        )

    # Two-row montage: paper vs one-domain (baseline)
    paper_row = outdir / "duddu2009_fig6_paper.png"
    base_row = outdir / "duddu2009_fig6_one_domain_interface_ny30.png"
    if paper_row.exists() and base_row.exists():
        _montage_rows(
            out_png=outdir / "duddu2009_fig6_compare_paper_vs_one_domain.png",
            rows=[[paper_row], [base_row]],
            titles=[["paper (Duddu 2009)"], ["one-domain reproduction"]],
            dpi=int(args.dpi),
        )

    # --------------------------
    # Quantitative summary (JSON)
    # --------------------------
    metrics: dict[str, object] = {}

    # Duddu 2007 Table I (sharp-interface) + one-domain slab speed
    if table1_src.exists():
        metrics["duddu2007_table1_reproduced"] = _read_csv_rows(table1_src)
    if one_domain_slab_src.exists():
        metrics["duddu2007_table1_one_domain_slab"] = json.loads(one_domain_slab_src.read_text())

    # Duddu 2007 Fig6 y_top errors
    def _y_top_err(csv_path: Path) -> dict[str, float]:
        rows = _read_csv_rows(csv_path)
        abs_err = [float(r["abs_err_mm"]) for r in rows if "abs_err_mm" in r]
        return {
            "mae_mm": sum(abs_err) / len(abs_err),
            "max_mm": max(abs_err),
            "final_mm": abs_err[-1],
        }

    ytop_csv_80 = one_domain_run / "y_top_compare.csv"
    if ytop_csv_80.exists():
        metrics["duddu2007_fig6_y_top_nx80"] = _y_top_err(ytop_csv_80)
    ytop_csv_60 = one_domain_run_nx60 / "y_top_compare.csv"
    if ytop_csv_60.exists():
        metrics["duddu2007_fig6_y_top_nx60"] = _y_top_err(ytop_csv_60)

    # Duddu 2009 1D detachment mapping errors (baseline + coarse ny=120)
    def _det1d_err(path: Path) -> dict[str, float]:
        rows = _read_csv_rows(path)
        e = [abs(float(r["L_pde_mm"]) - float(r["L_ode_mm"])) for r in rows]
        return {"t_end_days": float(rows[-1]["t_days"]), "max_abs_err_mm": max(e), "mean_abs_err_mm": sum(e) / len(e)}

    det1d_base = here / "results" / "duddu2009_detachment_1d" / "model=shear_backend=cpp_timeseries.csv"
    det1d_ny120 = here / "results" / "duddu2009_detachment_1d_ny120" / "model=shear_backend=cpp_timeseries.csv"
    if det1d_base.exists():
        metrics["duddu2009_detachment_1d_shear"] = _det1d_err(det1d_base)
    if det1d_ny120.exists():
        metrics["duddu2009_detachment_1d_shear_ny120"] = _det1d_err(det1d_ny120)

    # Duddu 2009 Fig6 summary metrics
    for key, p in {
        "duddu2009_fig6_summary_ny30": fig6_2d_base / "summary.csv",
        "duddu2009_fig6_summary_ny15": fig6_2d_coarse / "summary.csv",
    }.items():
        if p.exists():
            metrics[key] = _read_csv_rows(p)

    (outdir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[ok] wrote {outdir/'validation_metrics.json'}")

    print(f"[ok] assembled validation figures under {outdir}")


if __name__ == "__main__":
    main()
