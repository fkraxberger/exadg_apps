/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_VIBRATING_MEMBRANE_H_
#define APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_VIBRATING_MEMBRANE_H_

#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <exadg/grid/grid_utilities.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>


using namespace dealii;

namespace ExaDG
{
namespace Acoustics
{

template<int dim>
class AnalyticalBcPressure : public dealii::Function<dim>
{
public:
  AnalyticalBcPressure(const double freq, const double ampl)
    : dealii::Function<dim>(1, 0.0), freq(freq), ampl(ampl) // 1...scalar, 0.0...startzeitpunkt
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const) const final
  {
    double const t  = this->get_time();
    double const pi = dealii::numbers::PI;

    double result = ampl * std::sin(2.0 * pi * freq * t);

    return result;
  }

private:
  double const freq, ampl;
};

template<typename return_type>
std::vector<return_type>
split_string(std::string const &                             s,
             char const                                      c,
             std::function<return_type(const std::string &)> string_to_return_type)
{
  std::vector<return_type> result;

  auto it     = s.begin();
  auto it_sep = it;
  while((it_sep = std::find(it, s.end(), c)) != s.end())
  {
    const std::string comp{it, it_sep};

    result.push_back(string_to_return_type(comp));
    it = std::next(it_sep);
  }
  const std::string comp{it, it_sep};
  result.push_back(string_to_return_type(comp));
  
  return result;
}


template<int dim>
class ReadBcPressure : public dealii::Function<dim>
{
public:
  ReadBcPressure(double radius, const std::string boundary_val_filename)
    : dealii::Function<dim>(1, 0.0), boundary_val_filename(boundary_val_filename), boundaryVal(0.0), radius(radius)
  {
    std::string   line;
    std::ifstream file(boundary_val_filename);
    if(file.is_open())
    {
      while(getline(file, line))
      {
        auto row =
          split_string<double>(line, ',', [](const std::string & s) { return std::stod(s); });
        times.push_back(row[0]);
        values.push_back(row[1]);

        dealii::Point<dim> p;
        for(unsigned int i = 0; i < dim; ++i)
          p[i] = row[2 + i];
        positions.push_back(p);
      }
      file.close();
    }
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

    // std::cout<<"currentPosition "<<currentPosition ;
    // std::cout<<" | boundaryVal "<<boundaryVal<<std::endl;
  }


  double
  value(dealii::Point<dim> const & p, unsigned int const) const final
  {
    const double currentR = (currentPosition - p).norm();
    if(currentR < radius)
    {
      return boundaryVal;
    }
    return 0.0;
  }

private:
  std::string const               boundary_val_filename;
  std::vector<double>             times;
  std::vector<double>             values;
  double                          boundaryVal;
  std::vector<dealii::Point<dim>> positions;
  double                          radius;
  dealii::Point<dim>              currentPosition;
};



template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      // MATHEMATICAL MODEL
      prm.add_parameter("Formulation", this->param.formulation, "Formulation.");

      // PHYSICAL QUANTITIES
      prm.add_parameter("SpeedOfSound",
                        this->param.speed_of_sound,
                        "Speed of sound.",
                        dealii::Patterns::Double());

      // TEMPORAL DISCRETIZATION
      prm.add_parameter("TimeIntegrationScheme",
                        this->param.calculation_of_time_step_size,
                        "How to calculate time step size.");

      prm.add_parameter("UserSpecifiedTimeStepSize",
                        this->param.time_step_size,
                        "UserSpecified Timestep size.",
                        dealii::Patterns::Double());

      prm.add_parameter("CFL", this->param.cfl, "CFL number.", dealii::Patterns::Double());

      prm.add_parameter("OrderTimeIntegrator",
                        this->param.order_time_integrator,
                        "Order of time integration.",
                        dealii::Patterns::Integer(1));

      // APPLICATION SPECIFIC
      // prm.add_parameter("RuntimeInNumberOfPeriods",
      //                   number_of_periods,
      //                   "Number of temporal oscillations during runtime.",
      //                   dealii::Patterns::Double(1.0e-12));

      prm.add_parameter("EndTime",
                        end_time,
                        "End Time in seconds.",
                        dealii::Patterns::Double(1.0e-12));

      prm.add_parameter("Amplitude",
                        ampl,
                        "Amplitude of pressure excitation.",
                        dealii::Patterns::Double());

      prm.add_parameter("Frequency",
                        freq,
                        "Frequency of pressure excitation.",
                        dealii::Patterns::Double(1.0e-12));

      prm.add_parameter("Density", density, "Density.", dealii::Patterns::Double());
      prm.add_parameter("BoundaryValueFilename",
                        boundary_val_filename,
                        "File name for reading arbitrary Dirichlet boundary values.",
                        dealii::Patterns::FileName(),
                        true);
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // PHYSICAL QUANTITIES
    this->param.right_hand_side = true;
    this->param.start_time      = start_time;
    this->param.end_time      = end_time;

    // TEMPORAL DISCRETIZATION
    this->param.start_with_low_order = true;

    // output of solver information
    this->param.solver_info_data.interval_time = (this->param.end_time - this->param.start_time);

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 1;
    this->param.degree_p                = this->param.degree_u;
    this->param.degree_u                = this->param.degree_p;
  }

  void
  create_grid(Grid<dim> & grid, std::shared_ptr<dealii::Mapping<dim>> & mapping) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> & tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & /*periodic_face_pairs*/,
          unsigned int const global_refinements,
          std::vector<unsigned int> const & /* vector_local_refinements*/)
    {

      // GridIn<dim>(tria).read_exodusii("2D-extruded_test1_20m_v02.e", false);
      GridIn<dim>(tria).read_exodusii("../applications/acoustic_conservation_laws/valley/cube_1m.e", false);

      // GridIn writes ExodusII sideset_ids into manifold ids. We want to use it as boundary IDs and
      // have flat manifolds:
      // GridIn<dim>(tria).read_exodusii(
      //   "/storage_3_nfs/TrafficNoise_3D/01_Mesh/2D-valley_extruded/2D-extruded_test1_20m.e",
      //   true);
      // for(const auto & face : tria.active_face_iterators())
      //   if(face->at_boundary())
      //     face->set_boundary_id(face->manifold_id());
      // tria.set_all_manifold_ids_on_boundary(dealii::numbers::flat_manifold_id);

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation<dim>(
      grid, this->mpi_comm, this->param.grid, lambda_create_triangulation, {});

    GridUtilities::create_mapping(mapping,
                                  this->param.grid.element_type,
                                  this->param.mapping_degree);
  }

  void
  set_boundary_descriptor() final
  {
    // this->boundary_descriptor->pressure_dbc.insert(
    //   std::make_pair(1, std::make_shared<AnalyticalBcPressure<dim>>(freq, ampl)));

    // this->boundary_descriptor->pressure_dbc.insert(
    //   std::make_pair(1, std::make_shared<ReadBcPressure<dim>>(filename)));


    // for(int i = 1; i < 4; i++)
    // {
    //   // this->boundary_descriptor->pressure_dbc.insert(
    //   //   std::make_pair(i, std::make_shared<AnalyticalBcPressure<dim>>(freq, ampl)));
    //   this->boundary_descriptor->admittance_bc.insert(
    //     std::make_pair(i, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // }
    // for(int i = 4; i < 7; i++)
    // {
    //   // this->boundary_descriptor->pressure_dbc.insert(
    //   //   std::make_pair(i, std::make_shared<AnalyticalBcPressure<dim>>(freq, ampl)));
    //   this->boundary_descriptor->admittance_bc.insert(
    //     std::make_pair(i, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    // }

    // // Soundhard
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(2, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(3, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(4, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));

    // // ABC
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(1, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(5, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(6, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));

    // ID = 0 ??
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(0, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // // Wald
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(1, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(6, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // // Bahn, Strasse
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(2, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(4, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // // Wiese
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(3, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(5, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    // // ABC
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(7, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(8, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(9, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(10, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    // this->boundary_descriptor->admittance_bc.insert(std::make_pair(11, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));


    this->boundary_descriptor->admittance_bc.insert(std::make_pair(1, std::make_shared<Functions::ConstantFunction<dim>>(0.0)));
    this->boundary_descriptor->admittance_bc.insert(std::make_pair(2, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    this->boundary_descriptor->admittance_bc.insert(std::make_pair(3, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    this->boundary_descriptor->admittance_bc.insert(std::make_pair(4, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    this->boundary_descriptor->admittance_bc.insert(std::make_pair(5, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));
    this->boundary_descriptor->admittance_bc.insert(std::make_pair(6, std::make_shared<Functions::ConstantFunction<dim>>(1.0)));

  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_pressure =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);

    this->field_functions->initial_solution_velocity =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);

    // this->field_functions->right_hand_side =
    //   std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);

    double target_radius = 1.0; //0.1
    this->field_functions->right_hand_side =
      std::make_shared<ReadBcPressure<dim>>(target_radius, boundary_val_filename);
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // std::cout<<"END TIME "<<end_time<<std::endl;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active  = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time = start_time;
    pp_data.output_data.time_control_data.end_time = end_time;
    pp_data.output_data.time_control_data.trigger_interval =
      (end_time - start_time) / 40.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_pressure     = true;
    pp_data.output_data.write_velocity     = true;
    pp_data.output_data.write_boundary_IDs = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    // pointwise output
    pp_data.pointwise_output_data.time_control_data.is_active  = false;
    pp_data.pointwise_output_data.time_control_data.start_time = start_time;
    pp_data.pointwise_output_data.time_control_data.end_time   = this->param.end_time;
    pp_data.pointwise_output_data.time_control_data.trigger_interval =
      (this->end_time - start_time) / 1000.0;
    pp_data.pointwise_output_data.directory =
      this->output_parameters.directory + "pointwise_output/";
    pp_data.pointwise_output_data.filename       = this->output_parameters.filename;
    pp_data.pointwise_output_data.write_pressure = true;
    pp_data.pointwise_output_data.write_velocity = true;
    pp_data.pointwise_output_data.update_points_before_evaluation = false;
    // pp_data.pointwise_output_data.evaluation_points.push_back(
    //   dealii::Point<dim>(0.5 * (right - left), 0.5 * (right - left)));

    // sound energy calculation
    pp_data.sound_energy_data.time_control_data.is_active  = false;
    pp_data.sound_energy_data.time_control_data.start_time = this->param.start_time;
    pp_data.sound_energy_data.time_control_data.end_time = this->param.end_time;
    pp_data.sound_energy_data.density                      = density;
    pp_data.sound_energy_data.speed_of_sound               = this->param.speed_of_sound;
    pp_data.sound_energy_data.time_control_data.trigger_every_time_steps = 1;
    pp_data.sound_energy_data.directory = this->output_parameters.directory + "sound_energy/";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // problem specific parameters like physical dimensions, etc.
  double      freq              = 1.0;
  double      ampl              = 1.0;
  double      number_of_periods = 1.0;
  double      density           = 1.0;
  std::string boundary_val_filename          = "";

  double
  compute_period_duration()
  {
    return (1 / freq);
  }

  double const left  = 0.0;
  double const right = 1.0;

  double const start_time = 0.0;
  double end_time = 0.1;
};

} // namespace Acoustics

} // namespace ExaDG

#include <exadg/acoustic_conservation_equations/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_VIBRATING_MEMBRANE_H_ */
