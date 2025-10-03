/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_ACOUSTIC_CONSERVATION_LAWS_TEST_CASES_PLANE_WAVE_IN_DUCT_H_
#define APPLICATIONS_ACOUSTIC_CONSERVATION_LAWS_TEST_CASES_PLANE_WAVE_IN_DUCT_H_

#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <exadg/grid/mesh_movement_functions.h>

#include <exadg/acoustic_conservation_equations/postprocessor/postprocessor.h>

using namespace dealii;

namespace ExaDG::Acoustics
{
  template <int dim>
  struct PMLInfo
  {
    double speed_of_sound;
    double pml_thickness;
    dealii::Point<dim> point_on_plane;
    dealii::Point<dim> normal;
  };

  //// inverse distance damping
  template <int dim>
  class PMLDamping : public dealii::Function<dim>
  {
  public:
    PMLDamping(std::vector<PMLInfo<dim>> const pml_infos_in)
        : dealii::Function<dim>(dim), pml_infos(pml_infos_in)
    {
    }

    double
    value(dealii::Point<dim> const &p, unsigned int const i) const final
    {
      double result = 0.0;

      for (const auto &pml_info : pml_infos)
      {
        double const distance = (p - pml_info.point_on_plane) * pml_info.normal;
        double temp = 0.0;
        if (distance > -1.0e-8)
        {
          temp = pml_info.speed_of_sound / (pml_info.pml_thickness - distance) -
                 1.0 * pml_info.speed_of_sound / (pml_info.pml_thickness);
          temp *= pml_info.normal[i];
        }
        result += temp;
      }
      return result;
    }

  private:
    // quantities needed for pml: speed_of_sound, pml_thickness, p_on_plane, normal
    std::vector<PMLInfo<dim>> const pml_infos;
  };

  template <int dim>
  class CSVTrajectoryReader
  {
  public:
    bool
    parse_file(std::filesystem::path const &filename)
    {
      assert(std::filesystem::exists(filename));

      std::string line;
      std::ifstream file(filename);
      if (!file.is_open())
      {
        return false;
      }
      while (getline(file, line))
      {
        auto row =
            split_string<double>(line, ',', [](const std::string &s)
                                 { return std::stod(s); });
        times.push_back(row[0]);
        values.push_back(row[1]);

        dealii::Point<dim> p;
        for (unsigned int i = 0; i < dim; ++i)
          p[i] = row[2 + i];
        positions.push_back(p);
      }
      file.close();
      return true;
    }

  private:
    template <typename return_type>
    std::vector<return_type>
    split_string(std::string const &s,
                 char const c,
                 std::function<return_type(const std::string &)> string_to_return_type)
    {
      std::vector<return_type> result;

      auto it = s.begin();
      auto it_sep = it;
      while ((it_sep = std::find(it, s.end(), c)) != s.end())
      {
        const std::string comp{it, it_sep};

        result.push_back(string_to_return_type(comp));
        it = std::next(it_sep);
      }
      const std::string comp{it, it_sep};
      result.push_back(string_to_return_type(comp));

      return result;
    }

  public:
    std::vector<double> times;
    std::vector<double> values;
    std::vector<dealii::Point<dim>> positions;
  };

  template <int dim>
  class ReadBcPressure : public dealii::Function<dim>
  {
  public:
    ReadBcPressure(double radius, CSVTrajectoryReader<dim> const &reader)
        : dealii::Function<dim>(1, 0.0),
          boundaryVal(0.0),
          radius(radius),
          times(reader.times),
          values(reader.values),
          positions(reader.positions)
    {
    }

    void
    set_time(const double new_time) final
    {
      dealii::Function<dim>::set_time(new_time);

      auto it2 = std::upper_bound(times.begin(), times.end(), new_time);
      AssertThrow(std::distance(times.begin(), it2) > 0, dealii::ExcMessage("error message"));
      AssertThrow(it2 != times.end(),
                  dealii::ExcMessage("equal_range failed for " + std::to_string(new_time)));
      auto it1 = std::prev(it2);

      const double t1 = *it1;
      const double t2 = *it2;
      const double v1 = values[std::distance(times.begin(), it1)];
      const double v2 = values[std::distance(times.begin(), it2)];

      boundaryVal = v1 + (new_time - t1) / (t2 - t1) * (v2 - v1);

      const dealii::Point<dim> p1 = positions[std::distance(times.begin(), it1)];
      const dealii::Point<dim> p2 = positions[std::distance(times.begin(), it2)];

      currentPosition = p1 + (new_time - t1) / (t2 - t1) * (p2 - p1);
    }

    double
    value(dealii::Point<dim> const &p, unsigned int const) const final
    {
      const double currentR = (currentPosition - p).norm();
      if (currentR <= radius)
      {
        static bool written = false;
        if (!written)
        {
          // std::cout << "currentPosition " << currentPosition;
          // std::cout << " | boundaryVal " << boundaryVal << std::endl;
          written = true;
        }
        return boundaryVal;
      }
      return 0.0;
    }

  private:
    double boundaryVal;
    double radius;
    std::vector<double> times;
    std::vector<double> values;
    std::vector<dealii::Point<dim>> positions;
    dealii::Point<dim> currentPosition;
  };

  // template <int dim>
  // class PressureInlet : public dealii::Function<dim>
  // {
  // public:
  //   PressureInlet(double const period, unsigned int const number_of_periods)
  //       : dealii::Function<dim>(1, 0.0),
  //         period_(period),
  //         number_of_periods_(number_of_periods),
  //         omega_(2. * dealii::numbers::PI / period_)
  //   {
  //   }

  //   double
  //   value(dealii::Point<dim> const &, unsigned int const) const final
  //   {
  //     double const t = this->get_time();

  //     double result = 0.0;

  //     if (t < (double)number_of_periods_ * period_)
  //     {
  //       result = std::sin(omega_ * t);
  //     }

  //     return result;
  //   }

  // private:
  //   double const period_;
  //   unsigned int const number_of_periods_;
  //   double const omega_;
  // };

  template <int dim, typename Number>
  class Application : public ApplicationBase<dim, Number>
  {
  public:
    Application(std::string input_file, MPI_Comm const &comm)
        : ApplicationBase<dim, Number>(input_file, comm)
    {
    }

    void
    add_parameters(dealii::ParameterHandler &prm) final
    {
      ApplicationBase<dim, Number>::add_parameters(prm);
      prm.enter_subsection("Application");
      {
        prm.add_parameter("BoundaryValueFilename",
                          boundary_val_filename,
                          "File name for reading arbitrary Dirichlet boundary values.",
                          dealii::Patterns::FileName(),
                          true);

        prm.add_parameter("MeshFilename",
                          mesh_filename,
                          "File name for reading the mesh.",
                          dealii::Patterns::FileName(),
                          true);

        prm.add_parameter("EndTime",
                          end_time,
                          "End Time in seconds.",
                          dealii::Patterns::Double(1.0e-12));

        prm.add_parameter("SpeedOfSound",
                          speed_of_sound,
                          "Speed of sound.",
                          dealii::Patterns::Double());

        prm.add_parameter("SourceRadius",
                          target_radius,
                          "Radius of Source Term in m.",
                          dealii::Patterns::Double());
      }
      prm.leave_subsection();
    }

  private:
    CSVTrajectoryReader<dim> reader;

    void
    set_parameters() final
    {
      reader.parse_file(boundary_val_filename);

      this->param.formulation = Formulation::SkewSymmetric;
      this->param.right_hand_side = true;
      this->param.start_time = start_time;
      this->param.end_time = end_time;

      this->param.speed_of_sound = speed_of_sound;

      // std::cout << "start time " << start_time << " | end time " << end_time << std::endl;
      // std::cout << "speed of sound " << speed_of_sound << std::endl;

      this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
      this->param.cfl = 0.25;
      this->param.order_time_integrator = 2;
      this->param.start_with_low_order = true;
      this->param.adaptive_time_stepping = false;

      this->param.restarted_simulation = false;
      this->param.restart_data.write_restart = false;

      this->param.grid.triangulation_type = TriangulationType::Distributed;
      this->param.mapping_degree = 1;
      this->param.degree_p = this->param.degree_u;
      this->param.degree_u = this->param.degree_p;

      this->param.has_pml = true;
    }

    void
    create_grid(Grid<dim> &grid, std::shared_ptr<dealii::Mapping<dim>> &mapping) final
    {
      auto const lambda_create_triangulation =
          [&](dealii::Triangulation<dim, dim> &tria,
              std::vector<dealii::GridTools::PeriodicFacePair<
                  typename dealii::Triangulation<dim>::cell_iterator>> & /*periodic_face_pairs*/,
              unsigned int const global_refinements,
              std::vector<unsigned int> const & /* vector_local_refinements*/)
      {
        GridIn<dim>(tria).read_exodusii(mesh_filename, false);

        // std::set<unsigned int> material_id;
        for (const auto &cell : tria.active_cell_iterators())
        {
          // material_id.insert(cell->material_id());
          if (cell->material_id() == 2)
          {
            // static bool written = false;
            // if (!written)
            // {
            //   std::cout << "set material id to pml" << std::endl;
            //   written = true;
            // }
            cell->set_material_id(numbers::pml_material_id);
            // for (const auto &face : cell->face_iterators())
            //   if (face->at_boundary())
            //     face->set_boundary_id(99); // pml
          }
        }
        // for(auto const & m : material_id)
        //   std::cout << "material_id: " << m << std::endl;

        tria.refine_global(global_refinements);
        refine_triangulation_along_trajectory(tria, 2, 2.0 * target_radius);
      };

      GridUtilities::create_triangulation<dim>(
          grid, this->mpi_comm, this->param.grid, lambda_create_triangulation, {});

      GridUtilities::create_mapping(mapping,
                                    this->param.grid.element_type,
                                    this->param.mapping_degree);
    }

    void
    refine_triangulation_along_trajectory(dealii::Triangulation<dim> &tria,
                                          unsigned int const n_ref,
                                          double const radius) const
    {
      if (n_ref > 0)
      {
        for (unsigned int r = 0; r < n_ref; ++r)
        {
          for (auto const &cell : tria.active_cell_iterators())
            if (cell->is_locally_owned())
            {
              // for(const unsigned int i : dealii::GeometryInfo<dim>::vertex_indices())
              // {
              //   // std::cout<<"i: "<<i<<std::endl;
              //   auto const & v = cell->vertex(i);
              //   auto         it =
              //     std::find_if(reader.positions.begin(),
              //                  reader.positions.end(),
              //                  [&](auto const & p) { return (p - v).norm() < radius - 1.0e-6; });
              //   if(it != reader.positions.end())
              //   {
              //     cell->set_refine_flag();
              //   }
              // }
              if (auto it = std::find_if(reader.positions.begin(),
                                         reader.positions.end(),
                                         [&](auto const &p)
                                         { return cell->point_inside(p); });
                  it != reader.positions.end())
              {
                cell->set_refine_flag();
              }
            }

          tria.execute_coarsening_and_refinement();
        }
      }
    }

    void
    set_boundary_descriptor() final
    {
      // 0 not defined: ABC
      this->boundary_descriptor->admittance_bc.insert(
          std::make_pair(0, std::make_shared<dealii::Functions::ConstantFunction<dim>>(1.0)));
      // 1 sound hard
      this->boundary_descriptor->admittance_bc.insert(
          std::make_pair(1, std::make_shared<dealii::Functions::ConstantFunction<dim>>(0.0)));
      // 2 pml_outer_1: abc
      this->boundary_descriptor->admittance_bc.insert(
          std::make_pair(2, std::make_shared<dealii::Functions::ConstantFunction<dim>>(1.0)));
      // 3 pml_outer_2: abc
      this->boundary_descriptor->admittance_bc.insert(
          std::make_pair(3, std::make_shared<dealii::Functions::ConstantFunction<dim>>(1.0)));
    }

    void
    set_field_functions() final
    {
      this->field_functions->initial_solution_pressure.reset(
          new dealii::Functions::ZeroFunction<dim>(1));
      this->field_functions->initial_solution_velocity.reset(
          new dealii::Functions::ZeroFunction<dim>(dim));

      // std::cout << "target_radius: " << target_radius << std::endl;
      this->field_functions->right_hand_side =
          std::make_shared<ReadBcPressure<dim>>(target_radius, reader);

      std::vector<PMLInfo<dim>> pml_infos;
      PMLInfo<dim> pml_info;
      pml_info.speed_of_sound = this->param.speed_of_sound;
      pml_info.pml_thickness = 0.2;

      pml_info.point_on_plane = {1.0, 0.0, 0.0}; // PML right
      pml_info.normal = {1.0, 0.0, 0.0};
      pml_infos.emplace_back(pml_info);

      pml_info.point_on_plane = {0.0, 1.0, 0.0}; // PML top
      pml_info.normal = {0.0, 1.0, 0.0};
      pml_infos.emplace_back(pml_info);

      this->field_functions->pml_damping.reset(new PMLDamping<dim>(pml_infos));
    }

    std::shared_ptr<PostProcessorBase<dim, Number>>
    create_postprocessor() final
    {
      PostProcessorData<dim> pp_data;

      // write output for visualization of results
      pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
      pp_data.output_data.time_control_data.start_time = start_time;
      pp_data.output_data.time_control_data.end_time = end_time;

      pp_data.output_data.time_control_data.trigger_interval =
          (this->param.end_time - start_time) / 200.0;

      pp_data.output_data.directory = this->output_parameters.directory + "vtu/";
      pp_data.output_data.filename = this->output_parameters.filename;
      pp_data.output_data.write_velocity = true;
      pp_data.output_data.write_pressure = true;
      pp_data.output_data.write_processor_id = true;
      pp_data.output_data.write_boundary_IDs = true;
      pp_data.output_data.write_higher_order = true;
      pp_data.output_data.degree = 2;

      // pointwise output
      // pp_data.pointwise_output_data.time_control_data.is_active = false;
      // pp_data.pointwise_output_data.time_control_data.start_time = start_time;
      // pp_data.pointwise_output_data.time_control_data.end_time = this->param.end_time;
      // pp_data.pointwise_output_data.time_control_data.trigger_interval =
      //     (this->param.end_time - start_time) / 3000.0;
      // pp_data.pointwise_output_data.directory =
      //     this->output_parameters.directory + "pointwise_output/";
      // pp_data.pointwise_output_data.filename = this->output_parameters.filename;
      // pp_data.pointwise_output_data.write_pressure = true;
      // pp_data.pointwise_output_data.write_velocity = false;
      // pp_data.pointwise_output_data.update_points_before_evaluation = false;
      // pp_data.pointwise_output_data.evaluation_points.push_back(
      //     dealii::Point<dim>(1.0, 1.0, 1.0));

      std::shared_ptr<PostProcessorBase<dim, Number>> pp;
      pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

      return pp;
    }

    double length_ = 1.0;
    double height_ = 0.1;
    double period_ = 1.0;
    unsigned int number_of_periods_ = 1;
    double speed_of_sound = 343.0;
    // unsigned int n_elements_pml = 2;
    // double pml_length = 0.2;
    // double const theta = 0.0 * 0.25 * dealii::numbers::PI;

    std::string boundary_val_filename = "";
    std::string mesh_filename = "";
    double target_radius = 0.02;

    double const start_time = 0.0;
    double end_time = 0.1;
  };
} // namespace ExaDG::Acoustics

#include <exadg/acoustic_conservation_equations/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_ACOUSTIC_CONSERVATION_LAWS_TEST_CASES_PLANE_WAVE_IN_DUCT_H_ */
