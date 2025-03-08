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

namespace ExaDG::Acoustics {
    template<int dim>
    struct PMLInfo
    {
      double             speed_of_sound;
      double             pml_thickness;
      dealii::Point<dim> point_on_plane;
      dealii::Point<dim> normal;
    };

    //// inverse distance damping
    template<int dim>
    class PMLDamping : public dealii::Function<dim>
    {
    public:
      PMLDamping(std::vector<PMLInfo<dim>> const pml_infos_in) : pml_infos(pml_infos_in)
      {
      }

    double
    value(dealii::Point<dim> const & p, unsigned int const i) const final
    {
      double result = 0.0;

      for(const auto & pml_info : pml_infos)
      {
        double const distance = (p - pml_info.point_on_plane) * pml_info.normal;
        double       temp     = 0.0;
        if(distance > -1.0e-8)
        {
          temp = pml_info.speed_of_sound / (pml_info.pml_thickness - distance) -
                 0.0 * pml_info.speed_of_sound / (pml_info.pml_thickness);
          temp *= pml_info.normal[i];
        }
        result += temp;
      }
      std::cerr<<result<<std::endl;
      return result;
    }

    private:
      // quantities needed for pml: speed_of_sound, pml_thickness, p_on_plane, normal
      std::vector<PMLInfo<dim>> const pml_infos;
    };

    template<int dim>
    class PressureInlet : public dealii::Function<dim> {
    public:
        PressureInlet(double const period, unsigned int const number_of_periods)
            : dealii::Function<dim>(1, 0.0),
              period_(period),
              number_of_periods_(number_of_periods),
              omega_(2. * dealii::numbers::PI / period_) {
        }

        double
        value(dealii::Point<dim> const &, unsigned int const) const final {
            double const t = this->get_time();

            double result = 0.0;

            if (t < (double) number_of_periods_ * period_) {
                result = std::sin(omega_ * t);
            }

            return result;
        }

    private:
        double const period_;
        unsigned int const number_of_periods_;
        double const omega_;
    };


    template<int dim, typename Number>
    class Application : public ApplicationBase<dim, Number> {
    public:
        Application(std::string input_file, MPI_Comm const &comm)
            : ApplicationBase<dim, Number>(input_file, comm) {
        }

        void
        add_parameters(dealii::ParameterHandler &prm) final {
            ApplicationBase<dim, Number>::add_parameters(prm);
        }

    private:
        void
        set_parameters() final {
            this->param.formulation = Formulation::SkewSymmetric;
            this->param.right_hand_side = false;
            this->param.start_time = start_time_;
            this->param.end_time = start_time_ + ((double) number_of_periods_ * period_) + 2.5 * length_ /
                                   speed_of_sound_;
            this->param.speed_of_sound = speed_of_sound_;

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

            this->param.has_pml         = true;
        }

        void
        create_grid(Grid<dim> &grid, std::shared_ptr<dealii::Mapping<dim> > &mapping) final {
            auto const lambda_create_triangulation =
                    [&](dealii::Triangulation<dim, dim> &tria,
                        std::vector<dealii::GridTools::PeriodicFacePair<
                            typename dealii::Triangulation<dim>::cell_iterator> > & /*periodic_face_pairs*/,
                        unsigned int const /*global_refinements*/,
                        std::vector<unsigned int> const & /* vector_local_refinements*/) {
                unsigned int n_elements_x = static_cast<unsigned int>(std::ceil((length_) / height_));
                std::vector<unsigned int> subdivisions(dim, 1);
                subdivisions[0] = n_elements_x;
                dealii::Point<dim> p1;
                dealii::Point<dim> p2;
                p1[0] = 0.0;
                p2[0] = length_;
                for (uint d = 1; d < dim; ++d) {
                    p1[d] = -0.5 * height_;
                    p2[d] = 0.5 * height_;
                }

                double const bonus_domain_length = height_ * std::tan(theta);
                dealii::Triangulation<dim> domain;
                dealii::GridGenerator::subdivided_hyper_rectangle(domain, subdivisions, p1, p2);

                dealii::GridTools::transform(
                    [&](const dealii::Point<dim> &p) {
                        if (p[0] > length_ - 1e-6 && p[2] < -0.5 * height_ + 1e-6) {
                            auto p_new = p;
                            p_new[0] += bonus_domain_length;
                            return p_new;
                        }
                        return p;
                    },
                    domain);

                for (const auto &face: domain.active_face_iterators())
                    if (face->at_boundary()) {
                        if (face->center()[0] < 1e-6)
                            face->set_boundary_id(1); // left
                        else if ((std::abs(face->center()[1]) > (0.5 * height_) - 1.0e-6) ||
                                 (dim == 3 && std::abs(face->center()[2]) > (0.5 * height_) - 1.0e-6))
                            face->set_boundary_id(2); // wall
                        else
                            face->set_boundary_id(3); // right
                    }

                for (const auto &cell: domain.active_cell_iterators()) {
                    cell->set_material_id(0);
                }

                dealii::Triangulation<dim> pml;
                subdivisions[0] =
                        n_elements_pml * (unsigned int) std::pow(2, this->param.grid.n_refine_global);

                std::cout << "\nn_pml_elements: " << subdivisions[0] << "\n\n";

                p1[0] = length_;
                p2[0] = length_ + pml_length;
                dealii::GridGenerator::subdivided_hyper_rectangle(pml, subdivisions, p1, p2);

                dealii::GridTools::transform(
                    [&](const dealii::Point<dim> &p) {
                        if (p[0] > length_ - 1e-6 && p[2] < -0.5 * height_ + 1e-6) {
                            auto p_new = p;
                            p_new[0] += bonus_domain_length;
                            return p_new;
                        }
                        return p;
                    },
                    pml);

                for (const auto &face: pml.active_face_iterators())
                    if (face->at_boundary())
                        face->set_boundary_id(99); // pml

                for (const auto &cell: pml.active_cell_iterators())
                    cell->set_material_id(numbers::pml_material_id);

                dealii::GridGenerator::merge_triangulations(
                    domain, pml, tria, 1e-6, true, true);
            };

            GridUtilities::create_triangulation<dim>(
                grid, this->mpi_comm, this->param.grid, lambda_create_triangulation, {});

            GridUtilities::create_mapping(mapping,
                                          this->param.grid.element_type,
                                          this->param.mapping_degree);
        }

        void
        set_boundary_descriptor() final {
            // INLET
            this->boundary_descriptor->pressure_dbc.insert(
                std::make_pair(1, new PressureInlet<dim>(period_, number_of_periods_)));
            // WALL
            this->boundary_descriptor->admittance_bc.insert(
                std::make_pair(2, std::make_shared<dealii::Functions::ConstantFunction<dim> >(0.0)));
            // PML
            this->boundary_descriptor->pressure_dbc.insert(
                std::make_pair(99, new dealii::Functions::ZeroFunction<dim>(dim)));
        }

        void
        set_field_functions() final {
            this->field_functions->initial_solution_pressure.reset(
                new dealii::Functions::ZeroFunction<dim>(1));
            this->field_functions->initial_solution_velocity.reset(
                new dealii::Functions::ZeroFunction<dim>(dim));
            this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));

            std::vector<PMLInfo<dim>> pml_infos;
            PMLInfo<dim>              pml_info;
            pml_info.speed_of_sound = speed_of_sound_;
            pml_info.pml_thickness  = pml_length;
            pml_info.point_on_plane = {length_, 0.5 * height_, 0.5 * height_};
            pml_info.normal         = {1.0, 0.0, 0.0};
            pml_infos.emplace_back(pml_info);
            this->field_functions->pml_damping.reset(new PMLDamping<dim>(pml_infos));
        }

        std::shared_ptr<PostProcessorBase<dim, Number> >
        create_postprocessor() final {
            PostProcessorData<dim> pp_data;

            // write output for visualization of results
            pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
            pp_data.output_data.time_control_data.start_time = start_time_;

            pp_data.output_data.time_control_data.trigger_interval =
                    (this->param.end_time - start_time_) / 20.0;

            pp_data.output_data.directory = this->output_parameters.directory + "vtu/";
            pp_data.output_data.filename = this->output_parameters.filename;
            pp_data.output_data.write_velocity = false;
            pp_data.output_data.write_pressure = true;
            pp_data.output_data.write_processor_id = true;
            pp_data.output_data.write_boundary_IDs = true;
            pp_data.output_data.write_higher_order = true;
            pp_data.output_data.degree = this->param.degree_p;

            std::shared_ptr<PostProcessorBase<dim, Number> > pp;
            pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

            return pp;
        }


        double const start_time_ = 0.0;

        double length_ = 1.0;
        double height_ = 0.1;
        double period_ = 1.0;
        unsigned int number_of_periods_ = 1;
        double speed_of_sound_ = 1.0;
        double density_ = 1.0;
        unsigned int n_elements_pml = 5;
        double pml_length = 0.3;
        double const theta = 0.0 * 0.25 * dealii::numbers::PI;
    };
}

#include <exadg/acoustic_conservation_equations/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_ACOUSTIC_CONSERVATION_LAWS_TEST_CASES_PLANE_WAVE_IN_DUCT_H_ */
