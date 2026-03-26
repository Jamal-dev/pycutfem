#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

using namespace dealii;

namespace
{
  class SphereObstacle2D
  {
  public:
    explicit SphereObstacle2D(const double y_surface)
      : y_surface(y_surface)
    {}

    double value(const Point<2> &p) const
    {
      const double rad2 = 0.36 - (p[0] - 0.5) * (p[0] - 0.5);
      if (rad2 <= 0.0)
        return 1000.0;
      return -std::sqrt(rad2) + y_surface + 0.59;
    }

  private:
    const double y_surface;
  };


  class Step42ContactBox2D
  {
  public:
    Step42ContactBox2D(const unsigned int degree,
                       const unsigned int refinements,
                       const double       e_modulus,
                       const double       nu,
                       const double       c_scale,
                       const double       active_tol,
                       const std::string &output_dir)
      : fe(FE_Q<2>(QGaussLobatto<1>(degree + 1)) ^ 2)
      , dof_handler(triangulation)
      , degree(degree)
      , refinements(refinements)
      , e_modulus(e_modulus)
      , nu(nu)
      , lambda_lame(e_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
      , mu_lame(e_modulus / (2.0 * (1.0 + nu)))
      , c_scale(c_scale)
      , active_tol(active_tol)
      , output_dir(output_dir)
      , obstacle(1.0)
    {}

    void run()
    {
      make_grid();
      setup_system();
      solve_contact();
      output_results();
      write_summary();
    }

  private:
    void make_grid()
    {
      GridGenerator::hyper_rectangle(triangulation, Point<2>(0.0, 0.0), Point<2>(1.0, 1.0));
      for (const auto &cell : triangulation.active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              if (std::fabs(face->center()[1] - 1.0) < 1.0e-12)
                face->set_boundary_id(1);
              else if (std::fabs(face->center()[1] - 0.0) < 1.0e-12)
                face->set_boundary_id(6);
              else
                face->set_boundary_id(8);
            }

      triangulation.refine_global(refinements);
    }

    void setup_system()
    {
      dof_handler.distribute_dofs(fe);
      solution.reinit(dof_handler.n_dofs());
      lambda_vec.reinit(dof_handler.n_dofs());
      diag_mass.reinit(dof_handler.n_dofs());
      psi.reinit(dof_handler.n_dofs());
      active.assign(dof_handler.n_dofs(), false);

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);
      stiffness_matrix.reinit(sparsity_pattern);
      assemble_stiffness_matrix();
      assemble_top_mass_diagonal();
      setup_contact_support_points();
      setup_dirichlet_values();
    }

    void assemble_stiffness_matrix()
    {
      const QGauss<2> quadrature(degree + 1);
      FEValues<2>     fe_values(fe, quadrature, update_gradients | update_JxW_values);

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
      const unsigned int n_q = quadrature.size();
      const FEValuesExtractors::Vector displacement(0);

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;

          for (unsigned int q = 0; q < n_q; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const SymmetricTensor<2, 2> eps_i =
                  fe_values[displacement].symmetric_gradient(i, q);
                const SymmetricTensor<2, 2> sigma_i =
                  lambda_lame * trace(eps_i) * unit_symmetric_tensor<2>() +
                  2.0 * mu_lame * eps_i;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const SymmetricTensor<2, 2> eps_j =
                      fe_values[displacement].symmetric_gradient(j, q);
                    cell_matrix(i, j) += scalar_product(sigma_i, eps_j) * fe_values.JxW(q);
                  }
              }

          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              stiffness_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
        }
    }

    void assemble_top_mass_diagonal()
    {
      const QGaussLobatto<1> face_quadrature(degree + 1);
      FEFaceValues<2>        fe_face(fe, face_quadrature, update_values | update_JxW_values);
      const FEValuesExtractors::Vector displacement(0);

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
      const unsigned int n_q = face_quadrature.size();
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() && face->boundary_id() == 1)
            {
              fe_face.reinit(cell, face);
              cell->get_dof_indices(local_dof_indices);

              for (unsigned int q = 0; q < n_q; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const Tensor<1, 2> value_i = fe_face[displacement].value(i, q);
                    diag_mass(local_dof_indices[i]) += (value_i * value_i) * fe_face.JxW(q);
                  }
            }
    }

    void setup_contact_support_points()
    {
      const Quadrature<1> face_quadrature(fe.get_unit_face_support_points());
      FEFaceValues<2>     fe_face(fe, face_quadrature, update_quadrature_points);
      const FEValuesExtractors::Scalar y_displacement(1);

      std::vector<types::global_dof_index> face_dof_indices(fe.n_dofs_per_face());
      std::vector<bool>                    touched(dof_handler.n_dofs(), false);

      for (const auto &cell : dof_handler.active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() && face->boundary_id() == 1)
            {
              fe_face.reinit(cell, face);
              face->get_dof_indices(face_dof_indices);
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  const unsigned int gdof = face_dof_indices[q];
                  if (!fe.shape_function_belongs_to(q, y_displacement) || touched[gdof])
                    continue;
                  touched[gdof] = true;
                  top_uy_dofs.push_back(gdof);
                  const Point<2> p = fe_face.quadrature_point(q);
                  const double obstacle_y = obstacle.value(p);
                  psi(gdof) = (obstacle_y >= 100.0 ? -1.0e30 : obstacle_y - p[1]);
                }
            }
    }

    void setup_dirichlet_values()
    {
      Functions::ZeroFunction<2> zero(2);
      VectorTools::interpolate_boundary_values(dof_handler, 6, zero, dirichlet_values);

      const FEValuesExtractors::Scalar x_displacement(0);
      VectorTools::interpolate_boundary_values(
        dof_handler,
        8,
        zero,
        dirichlet_values,
        fe.component_mask(x_displacement));
    }

    void solve_contact()
    {
      const double c = c_scale * e_modulus;
      Vector<double> rhs(dof_handler.n_dofs());
      rhs = 0;

      for (unsigned int it = 1; it <= 80; ++it)
        {
          std::map<types::global_dof_index, double> fixed_values = dirichlet_values;
          for (const auto gdof : top_uy_dofs)
            if (active[gdof])
              fixed_values[gdof] = psi(gdof);

          SparseMatrix<double> system_matrix;
          system_matrix.reinit(sparsity_pattern);
          system_matrix.copy_from(stiffness_matrix);
          Vector<double> system_rhs(rhs);
          Vector<double> solution_trial(solution);
          MatrixTools::apply_boundary_values(fixed_values, system_matrix, solution_trial, system_rhs);

          SparseDirectUMFPACK solver;
          solver.initialize(system_matrix);
          solver.vmult(solution, system_rhs);

          stiffness_matrix.vmult(lambda_vec, solution);
          lambda_vec *= -1.0;

          std::vector<bool> active_new(active.size(), false);
          for (const auto gdof : top_uy_dofs)
            {
              const double indicator =
                lambda_vec(gdof) / diag_mass(gdof) + c * (solution(gdof) - psi(gdof));
              active_new[gdof] = indicator > active_tol;
            }

          iterations = it;
          if (active_new == active)
            {
              converged = true;
              active = active_new;
              break;
            }
          active = active_new;
        }
    }

    void output_results() const
    {
      DataOut<2> data_out;
      data_out.attach_dof_handler(dof_handler);

      Vector<double> contact_force(dof_handler.n_dofs());
      Vector<double> active_set_vec(dof_handler.n_dofs());
      for (const auto gdof : top_uy_dofs)
        {
          if (diag_mass(gdof) > 0.0)
            contact_force(gdof) = lambda_vec(gdof) / diag_mass(gdof);
          if (active[gdof])
            active_set_vec(gdof) = 1.0;
        }

      const std::vector<DataComponentInterpretation::DataComponentInterpretation> interp(
        2, DataComponentInterpretation::component_is_part_of_vector);
      data_out.add_data_vector(solution, std::vector<std::string>{"displacement", "displacement"}, DataOut<2>::type_dof_data, interp);
      data_out.add_data_vector(contact_force, std::vector<std::string>{"contact_force", "contact_force"}, DataOut<2>::type_dof_data, interp);
      data_out.add_data_vector(active_set_vec, std::vector<std::string>{"active_set", "active_set"}, DataOut<2>::type_dof_data, interp);
      data_out.build_patches();

      std::ofstream out(output_dir + "/solution.vtu");
      data_out.write_vtu(out);
    }

    void write_summary() const
    {
      double lambda_sum = 0.0;
      double gap_min = std::numeric_limits<double>::max();
      double gap_max = -std::numeric_limits<double>::max();
      unsigned int n_active = 0;
      for (const auto gdof : top_uy_dofs)
        {
          lambda_sum += lambda_vec(gdof);
          if (psi(gdof) > -1.0e20)
            {
              gap_min = std::min(gap_min, psi(gdof) - solution(gdof));
              gap_max = std::max(gap_max, psi(gdof) - solution(gdof));
            }
          if (active[gdof])
            ++n_active;
        }

      std::ofstream out(output_dir + "/summary.json");
      out << "{\n"
          << "  \"degree\": " << degree << ",\n"
          << "  \"refinements\": " << refinements << ",\n"
          << "  \"iterations\": " << iterations << ",\n"
          << "  \"converged\": " << (converged ? "true" : "false") << ",\n"
          << "  \"n_active\": " << n_active << ",\n"
          << "  \"contact_force_total_lumped\": " << lambda_sum << ",\n"
          << "  \"gap_min_top\": " << gap_min << ",\n"
          << "  \"gap_max_top\": " << gap_max << "\n"
          << "}\n";
    }

    Triangulation<2> triangulation;
    FESystem<2>      fe;
    DoFHandler<2>    dof_handler;

    const unsigned int degree;
    const unsigned int refinements;
    const double       e_modulus;
    const double       nu;
    const double       lambda_lame;
    const double       mu_lame;
    const double       c_scale;
    const double       active_tol;
    const std::string  output_dir;
    const SphereObstacle2D obstacle;

    SparseMatrix<double> stiffness_matrix;
    SparsityPattern      sparsity_pattern;
    Vector<double>       solution;
    Vector<double>       lambda_vec;
    Vector<double>       diag_mass;
    Vector<double>       psi;
    std::map<types::global_dof_index, double> dirichlet_values;
    std::vector<types::global_dof_index>       top_uy_dofs;
    std::vector<bool>                          active;
    unsigned int                               iterations = 0;
    bool                                       converged = false;
  };
} // namespace


int main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      unsigned int degree      = 1;
      unsigned int refinements = 3;
      std::string  output_dir  = "out/step42_contact_box_2d_dealii";
      if (argc >= 2)
        degree = static_cast<unsigned int>(std::stoi(argv[1]));
      if (argc >= 3)
        refinements = static_cast<unsigned int>(std::stoi(argv[2]));
      if (argc >= 4)
        output_dir = argv[3];

      std::filesystem::create_directories(output_dir);
      Step42ContactBox2D problem(degree, refinements, 200000.0, 0.3, 100.0, 0.0, output_dir);
      problem.run();
    }
  catch (const std::exception &exc)
    {
      std::cerr << exc.what() << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception." << std::endl;
      return 1;
    }

  return 0;
}
