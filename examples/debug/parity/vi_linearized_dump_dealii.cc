#include <deal.II/base/exceptions.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace dealii;

namespace
{
  struct TripletMatrix
  {
    unsigned int n_rows = 0;
    unsigned int n_cols = 0;
    std::vector<std::tuple<unsigned int, unsigned int, double>> entries;
  };

  struct OwnedSparse
  {
    SparsityPattern    pattern;
    SparseMatrix<double> matrix;
  };

  TripletMatrix
  read_triplets(const std::string &path)
  {
    std::ifstream in(path);
    AssertThrow(in, ExcMessage("Could not open " + path));
    TripletMatrix out;
    unsigned int nnz = 0;
    in >> out.n_rows >> out.n_cols >> nnz;
    out.entries.reserve(nnz);
    for (unsigned int k = 0; k < nnz; ++k)
      {
        unsigned int i = 0;
        unsigned int j = 0;
        double       v = 0.0;
        in >> i >> j >> v;
        out.entries.emplace_back(i, j, v);
      }
    return out;
  }

  double
  parse_scalar_token(const std::string &token)
  {
    auto lower = token;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (lower == "inf" || lower == "+inf" || lower == "infinity" || lower == "+infinity")
      return std::numeric_limits<double>::infinity();
    if (lower == "-inf" || lower == "-infinity")
      return -std::numeric_limits<double>::infinity();
    if (lower == "nan" || lower == "+nan" || lower == "-nan")
      return std::numeric_limits<double>::quiet_NaN();
    return std::stod(token);
  }

  double
  read_scalar_token(std::istream &in, const std::string &path)
  {
    std::string token;
    in >> token;
    AssertThrow(in, ExcMessage("Could not read scalar token from " + path));
    return parse_scalar_token(token);
  }

  Vector<double>
  read_vector(const std::string &path)
  {
    std::ifstream in(path);
    AssertThrow(in, ExcMessage("Could not open " + path));
    unsigned int n = 0;
    in >> n;
    Vector<double> v(n);
    for (unsigned int i = 0; i < n; ++i)
      v[i] = read_scalar_token(in, path);
    return v;
  }

  void
  write_vector(const std::string &path, const Vector<double> &v)
  {
    std::ofstream out(path);
    AssertThrow(out, ExcMessage("Could not open " + path + " for writing"));
    out << v.size() << '\n';
    for (unsigned int i = 0; i < v.size(); ++i)
      out << std::scientific << std::setprecision(17) << v[i] << '\n';
  }

  void
  write_sparse_triplets(const std::string &path, const SparseMatrix<double> &A)
  {
    std::ofstream out(path);
    AssertThrow(out, ExcMessage("Could not open " + path + " for writing"));
    out << A.m() << ' ' << A.n() << ' ' << A.n_nonzero_elements() << '\n';
    for (unsigned int i = 0; i < A.m(); ++i)
      for (SparseMatrix<double>::const_iterator e = A.begin(i); e != A.end(i); ++e)
        out << i << ' ' << e->column() << ' ' << std::scientific << std::setprecision(17)
            << e->value() << '\n';
  }

  OwnedSparse
  build_sparse(const TripletMatrix &trip)
  {
    DynamicSparsityPattern dsp(trip.n_rows, trip.n_cols);
    for (const auto &[i, j, _] : trip.entries)
      dsp.add(i, j);
    OwnedSparse out;
    out.pattern.copy_from(dsp);
    out.matrix.reinit(out.pattern);
    for (const auto &[i, j, v] : trip.entries)
      out.matrix.add(i, j, v);
    return out;
  }

  double
  linf_norm(const Vector<double> &v)
  {
    double n = 0.0;
    for (unsigned int i = 0; i < v.size(); ++i)
      n = std::max(n, std::abs(v[i]));
    return n;
  }

  std::vector<double>
  sparse_row_abs_sum(const SparseMatrix<double> &A)
  {
    std::vector<double> row_sum(A.m(), 0.0);
    for (unsigned int i = 0; i < A.m(); ++i)
      for (SparseMatrix<double>::const_iterator e = A.begin(i); e != A.end(i); ++e)
        row_sum[i] += std::abs(e->value());
    return row_sum;
  }

  void
  add_diagonal_shift(SparseMatrix<double>       &A,
                     const std::vector<double>  &diag_scale,
                     const double                factor,
                     const std::vector<int>     &state,
                     const unsigned int          first_row,
                     const unsigned int          last_row)
  {
    const unsigned int end_row = std::min<unsigned int>(last_row, A.m());
    for (unsigned int i = first_row; i < end_row; ++i)
      {
        if (i < state.size() && state[i] != 0)
          continue;
        const double scale = std::max(diag_scale[i], 1.0e-12);
        A.add(i, i, factor * scale);
      }
  }

  struct Summary
  {
    unsigned int iterations = 0;
    bool         converged = false;
    unsigned int n_active_lo = 0;
    unsigned int n_active_hi = 0;
    double       g_inf = 0.0;
    double       inactive_res_inf = 0.0;
    double       active_gap_inf = 0.0;
    double       equality_inf = 0.0;
  };

  struct IterationRecord
  {
    unsigned int iter = 0;
    unsigned int n_active_lo = 0;
    unsigned int n_active_hi = 0;
    unsigned int delta_active = 0;
    std::string  linear_solver = "";
    double       shift_factor = 0.0;
    unsigned int gmres_steps = 0;
    double       rhs_inf = 0.0;
    double       linear_res_inf = 0.0;
    double       g_inf = 0.0;
    double       inactive_res_inf = 0.0;
    double       active_gap_inf = 0.0;
    double       equality_inf = 0.0;
    double       y_inf = 0.0;
    double       lambda_inf = 0.0;
  };

  void
  compute_vi_metrics(const SparseMatrix<double> &A,
                     const SparseMatrix<double> &B,
                     const Vector<double>       &rhs_base,
                     const Vector<double>       &b_eff,
                     const Vector<double>       &lo,
                     const Vector<double>       &hi,
                     const Vector<double>       &y,
                     const Vector<double>       &lambda,
                     const std::vector<int>     &state,
                     double                     &g_inf,
                     double                     &inactive_res_inf,
                     double                     &active_gap_inf,
                     double                     &equality_inf)
  {
    const unsigned int n = A.m();
    Vector<double> Ay(n);
    A.vmult(Ay, y);
    Vector<double> ByT(n);
    B.Tvmult(ByT, lambda);
    Vector<double> stationarity(n);
    stationarity = Ay;
    stationarity -= rhs_base;
    stationarity += ByT;

    Vector<double> eq_res(B.m());
    B.vmult(eq_res, y);
    eq_res -= b_eff;

    Vector<double> G(n);
    inactive_res_inf = 0.0;
    active_gap_inf   = 0.0;
    for (unsigned int i = 0; i < n; ++i)
      {
        if (state[i] == 1)
          {
            G[i] = y[i] - lo[i];
            active_gap_inf = std::max(active_gap_inf, std::abs(G[i]));
          }
        else if (state[i] == -1)
          {
            G[i] = y[i] - hi[i];
            active_gap_inf = std::max(active_gap_inf, std::abs(G[i]));
          }
        else
          {
            G[i] = stationarity[i];
            inactive_res_inf = std::max(inactive_res_inf, std::abs(G[i]));
          }
      }
    g_inf        = linf_norm(G);
    equality_inf = linf_norm(eq_res);
  }

  void
  write_json_double(std::ostream &out, const double value)
  {
    if (std::isfinite(value))
      out << std::scientific << value;
    else
      out << "null";
  }

  void
  write_history_json(const std::string                  &path,
                     const std::vector<IterationRecord> &history)
  {
    std::ofstream out(path);
    AssertThrow(out, ExcMessage("Could not open " + path + " for writing"));
    out << "[\n";
    for (unsigned int k = 0; k < history.size(); ++k)
      {
        const auto &r = history[k];
        out << "  {\n";
        out << "    \"iter\": " << r.iter << ",\n";
        out << "    \"n_active_lo\": " << r.n_active_lo << ",\n";
        out << "    \"n_active_hi\": " << r.n_active_hi << ",\n";
        out << "    \"delta_active\": " << r.delta_active << ",\n";
        out << "    \"linear_solver\": \"" << r.linear_solver << "\",\n";
        out << "    \"shift_factor\": ";
        write_json_double(out, r.shift_factor);
        out << ",\n";
        out << "    \"gmres_steps\": " << r.gmres_steps << ",\n";
        out << "    \"rhs_inf\": ";
        write_json_double(out, r.rhs_inf);
        out << ",\n";
        out << "    \"linear_res_inf\": ";
        write_json_double(out, r.linear_res_inf);
        out << ",\n";
        out << "    \"g_inf\": ";
        write_json_double(out, r.g_inf);
        out << ",\n";
        out << "    \"inactive_res_inf\": ";
        write_json_double(out, r.inactive_res_inf);
        out << ",\n";
        out << "    \"active_gap_inf\": ";
        write_json_double(out, r.active_gap_inf);
        out << ",\n";
        out << "    \"equality_inf\": ";
        write_json_double(out, r.equality_inf);
        out << ",\n";
        out << "    \"y_inf\": ";
        write_json_double(out, r.y_inf);
        out << ",\n";
        out << "    \"lambda_inf\": ";
        write_json_double(out, r.lambda_inf);
        out << "\n";
        out << "  }" << (k + 1 < history.size() ? "," : "") << "\n";
      }
    out << "]\n";
  }

  Summary
  solve_linearized_pdas(const SparseMatrix<double> &A,
                        const SparseMatrix<double> &B,
                        const Vector<double>       &x0,
                        const Vector<double>       &lo,
                        const Vector<double>       &hi,
                        const Vector<double>       &c,
                        const Vector<double>       &stat0,
                        const Vector<double>       &b_eff,
                        const Vector<double>       &lambda0,
                        Vector<double>             &y,
                        Vector<double>             &lambda,
                        std::vector<IterationRecord> &history)
  {
    const unsigned int n = A.m();
    const unsigned int m = B.m();
    AssertDimension(A.n(), n);
    AssertDimension(B.n(), n);
    AssertDimension(x0.size(), n);
    AssertDimension(lo.size(), n);
    AssertDimension(hi.size(), n);
    AssertDimension(c.size(), n);
    AssertDimension(stat0.size(), n);
    AssertDimension(b_eff.size(), m);

    y      = x0;
    lambda = lambda0;

    Vector<double> rhs_base(n);
    A.vmult(rhs_base, x0);
    rhs_base -= stat0;
    std::vector<double> A_diag_scale(n, 1.0e-12);
    for (unsigned int i = 0; i < n; ++i)
      {
        const double diag = std::abs(A.diag_element(i));
        A_diag_scale[i] = diag > 1.0e-12 ? diag : 1.0e-12;
      }

    std::vector<int> state(n, 0);
    std::vector<int> prev_state(n, 7);
    Summary          summary;
    history.clear();
    const bool trace = []() {
      const char *env = std::getenv("PYCUTFEM_VI_DEALII_TRACE");
      if (env == nullptr)
        return false;
      const std::string s(env);
      return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES";
    }();
    const char *dump_env = std::getenv("PYCUTFEM_VI_DEALII_DUMP_DIR");
    const std::filesystem::path dump_dir =
      dump_env != nullptr ? std::filesystem::path(dump_env) : std::filesystem::path();
    if (!dump_dir.empty())
      std::filesystem::create_directories(dump_dir);

    for (unsigned int it = 0; it < 50; ++it)
      {
        summary.iterations = it + 1;

        Vector<double> Ay(n);
        A.vmult(Ay, y);
        Vector<double> ByT(n);
        B.Tvmult(ByT, lambda);

        Vector<double> stationarity(n);
        stationarity = Ay;
        stationarity -= rhs_base;
        stationarity += ByT;

        for (unsigned int i = 0; i < n; ++i)
          {
            const bool lo_f = std::isfinite(lo[i]);
            const bool hi_f = std::isfinite(hi[i]);
            const double ind_lo = stationarity[i] - c[i] * (y[i] - lo[i]);
            const double ind_hi = stationarity[i] + c[i] * (hi[i] - y[i]);
            int new_state = 0;
            if (lo_f && ind_lo > 0.0)
              new_state = 1;
            if (hi_f && ind_hi < 0.0)
              new_state = -1;
            state[i] = new_state;
          }

        const unsigned int n_active_lo =
          static_cast<unsigned int>(std::count(state.begin(), state.end(), 1));
        const unsigned int n_active_hi =
          static_cast<unsigned int>(std::count(state.begin(), state.end(), -1));
        unsigned int delta_active = 0;
        for (unsigned int i = 0; i < n; ++i)
          if (state[i] != prev_state[i])
            ++delta_active;

        DynamicSparsityPattern dsp(n + m, n + m);
        for (unsigned int i = 0; i < n; ++i)
          {
            if (state[i] != 0)
              {
                dsp.add(i, i);
              }
            else
              {
                for (SparseMatrix<double>::const_iterator e = A.begin(i); e != A.end(i); ++e)
                  dsp.add(i, e->column());
                for (unsigned int j = 0; j < m; ++j)
                  if (std::abs(B.el(j, i)) > 0.0)
                    dsp.add(i, n + j);
              }
          }
        for (unsigned int j = 0; j < m; ++j)
          for (SparseMatrix<double>::const_iterator e = B.begin(j); e != B.end(j); ++e)
            dsp.add(n + j, e->column());

        SparsityPattern sp_aug;
        sp_aug.copy_from(dsp);
        SparseMatrix<double> M;
        M.reinit(sp_aug);
        Vector<double> rhs(n + m);

        for (unsigned int i = 0; i < n; ++i)
          {
            if (state[i] == 1)
              {
                M.set(i, i, 1.0);
                rhs[i] = lo[i];
              }
            else if (state[i] == -1)
              {
                M.set(i, i, 1.0);
                rhs[i] = hi[i];
              }
            else
              {
                for (SparseMatrix<double>::const_iterator e = A.begin(i); e != A.end(i); ++e)
                  M.add(i, e->column(), e->value());
                for (unsigned int j = 0; j < m; ++j)
                  {
                    const double bij = B.el(j, i);
                    if (std::abs(bij) > 0.0)
                      M.add(i, n + j, bij);
                  }
                rhs[i] = rhs_base[i];
              }
          }

        for (unsigned int j = 0; j < m; ++j)
          {
            for (SparseMatrix<double>::const_iterator e = B.begin(j); e != B.end(j); ++e)
              M.add(n + j, e->column(), e->value());
            rhs[n + j] = b_eff[j];
          }
        if (!dump_dir.empty())
          {
            const std::string base = "iter" + std::to_string(it + 1);
            write_sparse_triplets((dump_dir / (base + "_M.triplets")).string(), M);
            write_vector((dump_dir / (base + "_rhs.txt")).string(), rhs);
            Vector<double> state_vec(n);
            for (unsigned int i = 0; i < n; ++i)
              state_vec[i] = static_cast<double>(state[i]);
            write_vector((dump_dir / (base + "_state.txt")).string(), state_vec);
          }

        bool                solved = false;
        const std::array<double, 4> retry_shift = {{1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1}};
        Vector<double>             sol(n + m);
        double                     used_shift = 0.0;
        unsigned int               gmres_steps = 0;
        std::string                linear_solver = "";
        const double               rhs_inf = linf_norm(rhs);
        double                     linear_res_inf = std::numeric_limits<double>::infinity();
        for (const double factor : retry_shift)
          {
            try
              {
                SparseMatrix<double> M_solve;
                M_solve.reinit(M.get_sparsity_pattern());
                M_solve.copy_from(M);
                if (factor > 0.0)
                  {
                    add_diagonal_shift(M_solve, A_diag_scale, factor, state, 0, n);
                  }
                {
                  SparseDirectUMFPACK solver;
                  solver.initialize(M_solve);
                  Vector<double> trial(n + m);
                  solver.vmult(trial, rhs);
                  Vector<double> lin_res(rhs.size());
                  M_solve.vmult(lin_res, trial);
                  lin_res -= rhs;
                  sol = trial;
                  used_shift = factor;
                  gmres_steps = 0;
                  linear_res_inf = linf_norm(lin_res);
                  linear_solver = "umfpack";
                  solved = true;
                  break;
                }
              }
            catch (const SolverControl::NoConvergence &)
              {
                solved = false;
              }
            catch (const dealii::ExceptionBase &)
              {
                try
                  {
                    SparseMatrix<double> M_solve;
                    M_solve.reinit(M.get_sparsity_pattern());
                    M_solve.copy_from(M);
                    if (factor > 0.0)
                      {
                        add_diagonal_shift(M_solve, A_diag_scale, factor, state, 0, n);
                      }
                    SolverControl control(std::max<unsigned int>(2000, 20 * (n + m)),
                                          std::max(1.0e-12, 1.0e-10 * rhs.l2_norm()));
                    SolverGMRES<Vector<double>>::AdditionalData gmres_data;
                    gmres_data.max_n_tmp_vectors =
                      std::min<unsigned int>(400, std::max<unsigned int>(50, n + m));
                    SolverGMRES<Vector<double>> solver(control, gmres_data);
                    PreconditionIdentity prec;
                    Vector<double> trial(n + m);
                    solver.solve(M_solve, trial, rhs, prec);
                    Vector<double> lin_res(rhs.size());
                    M_solve.vmult(lin_res, trial);
                    lin_res -= rhs;
                    sol = trial;
                    used_shift = factor;
                    gmres_steps = control.last_step();
                    linear_res_inf = linf_norm(lin_res);
                    linear_solver = "gmres";
                    solved = true;
                    break;
                  }
                catch (const SolverControl::NoConvergence &)
                  {
                    solved = false;
                  }
                catch (const dealii::ExceptionBase &)
                  {
                    solved = false;
                  }
              }
          }
        if (!solved)
          {
            IterationRecord rec;
            rec.iter = it + 1;
            rec.n_active_lo = n_active_lo;
            rec.n_active_hi = n_active_hi;
            rec.delta_active = delta_active;
            rec.linear_solver = "failed";
            rec.shift_factor = std::numeric_limits<double>::quiet_NaN();
            rec.gmres_steps = 0;
            rec.rhs_inf = rhs_inf;
            rec.linear_res_inf = std::numeric_limits<double>::infinity();
            compute_vi_metrics(
              A,
              B,
              rhs_base,
              b_eff,
              lo,
              hi,
              y,
              lambda,
              state,
              rec.g_inf,
              rec.inactive_res_inf,
              rec.active_gap_inf,
              rec.equality_inf);
            rec.y_inf = linf_norm(y);
            rec.lambda_inf = linf_norm(lambda);
            history.push_back(rec);
            if (trace)
              {
                std::cerr << "[dealii-vi] it=" << (it + 1) << " active_lo=" << n_active_lo
                          << " active_hi=" << n_active_hi << " delta=" << delta_active
                          << " rhs_inf=" << std::scientific << rhs_inf
                          << " solve=failed"
                          << " g_inf=" << rec.g_inf
                          << " eq_inf=" << rec.equality_inf << '\n';
              }
            summary.converged = false;
            break;
          }

        for (unsigned int i = 0; i < n; ++i)
          y[i] = sol[i];
        for (unsigned int j = 0; j < m; ++j)
          lambda[j] = sol[n + j];
        if (!dump_dir.empty())
          {
            const std::string base = "iter" + std::to_string(it + 1);
            Vector<double> y_aug(n + m);
            for (unsigned int i = 0; i < n; ++i)
              y_aug[i] = y[i];
            for (unsigned int j = 0; j < m; ++j)
              y_aug[n + j] = lambda[j];
            write_vector((dump_dir / (base + "_sol.txt")).string(), y_aug);
          }

        IterationRecord rec;
        rec.iter = it + 1;
        rec.n_active_lo = n_active_lo;
        rec.n_active_hi = n_active_hi;
        rec.delta_active = delta_active;
        rec.linear_solver = linear_solver;
        rec.shift_factor = used_shift;
        rec.gmres_steps = gmres_steps;
        rec.rhs_inf = rhs_inf;
        rec.linear_res_inf = linear_res_inf;
        compute_vi_metrics(
          A,
          B,
          rhs_base,
          b_eff,
          lo,
          hi,
          y,
          lambda,
          state,
          rec.g_inf,
          rec.inactive_res_inf,
          rec.active_gap_inf,
          rec.equality_inf);
        rec.y_inf = linf_norm(y);
        rec.lambda_inf = linf_norm(lambda);
        history.push_back(rec);
        if (trace)
          {
            std::cerr << "[dealii-vi] it=" << (it + 1) << " active_lo=" << n_active_lo
                      << " active_hi=" << n_active_hi << " delta=" << delta_active
                      << " rhs_inf=" << std::scientific << rhs_inf
                      << " solver=" << linear_solver
                      << " shift=" << used_shift
                      << " gmres=" << gmres_steps
                      << " lin_res=" << linear_res_inf
                      << " g_inf=" << rec.g_inf
                      << " eq_inf=" << rec.equality_inf << '\n';
          }

        if (state == prev_state)
          {
            summary.converged = true;
            break;
          }
        prev_state = state;
      }

    summary.n_active_lo =
      static_cast<unsigned int>(std::count(state.begin(), state.end(), 1));
    summary.n_active_hi =
      static_cast<unsigned int>(std::count(state.begin(), state.end(), -1));
    compute_vi_metrics(
      A,
      B,
      rhs_base,
      b_eff,
      lo,
      hi,
      y,
      lambda,
      state,
      summary.g_inf,
      summary.inactive_res_inf,
      summary.active_gap_inf,
      summary.equality_inf);
    return summary;
  }
} // namespace


int main(int argc, char **argv)
{
  dealii::deal_II_exceptions::disable_abort_on_exception();

  if (argc != 3)
    {
      std::cerr << "usage: vi_linearized_dump_dealii <input_dir> <output_json>\n";
      return 2;
    }

  const std::filesystem::path input_dir(argv[1]);
  const std::filesystem::path output_json(argv[2]);

  const auto A_trip  = read_triplets((input_dir / "A_red.triplets").string());
  const auto B_trip  = read_triplets((input_dir / "B_red.triplets").string());
  const auto A_owned = build_sparse(A_trip);
  const auto B_owned = build_sparse(B_trip);
  const auto &A      = A_owned.matrix;
  const auto &B      = B_owned.matrix;
  const auto x0      = read_vector((input_dir / "x_red.txt").string());
  const auto lo      = read_vector((input_dir / "lo_red.txt").string());
  const auto hi      = read_vector((input_dir / "hi_red.txt").string());
  const auto c       = read_vector((input_dir / "c_red.txt").string());
  const auto stat0   = read_vector((input_dir / "stat_red.txt").string());
  const auto b_eff   = read_vector((input_dir / "eq_b_eff.txt").string());
  const auto lambda0 = read_vector((input_dir / "eq_lambda.txt").string());

  Vector<double> y;
  Vector<double> lambda;
  lambda = lambda0;
  std::vector<IterationRecord> history;
  const auto summary = solve_linearized_pdas(A, B, x0, lo, hi, c, stat0, b_eff, lambda0, y, lambda, history);

  std::filesystem::create_directories(output_json.parent_path());
  write_vector((output_json.parent_path() / "y_red.txt").string(), y);
  write_vector((output_json.parent_path() / "lambda_red.txt").string(), lambda);
  write_history_json((output_json.parent_path() / "history.json").string(), history);
  std::ofstream out(output_json);
  out << "{\n";
  out << "  \"iterations\": " << summary.iterations << ",\n";
  out << "  \"converged\": " << (summary.converged ? "true" : "false") << ",\n";
  out << "  \"n_active_lo\": " << summary.n_active_lo << ",\n";
  out << "  \"n_active_hi\": " << summary.n_active_hi << ",\n";
  out << "  \"g_inf\": ";
  write_json_double(out, summary.g_inf);
  out << ",\n";
  out << "  \"inactive_res_inf\": ";
  write_json_double(out, summary.inactive_res_inf);
  out << ",\n";
  out << "  \"active_gap_inf\": ";
  write_json_double(out, summary.active_gap_inf);
  out << ",\n";
  out << "  \"equality_inf\": ";
  write_json_double(out, summary.equality_inf);
  out << "\n";
  out << "}\n";
  return 0;
}
