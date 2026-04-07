#include "solver.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(poisson_solver, m) 
{
    m.doc() = "2D Poisson equation solver using Gauss-Seidel and SOR methods with multigrid refinement.";

    py::class_<PoissonSolver2D>(m, "PoissonSolver2D")
        .def(py::init<int, int, bool, int>(), 
             py::arg("n_power"), 
             py::arg("log_every") = 10, 
             py::arg("track_history") = false, 
             py::arg("history_stride") = 10)

        // main solvers
        .def("solve_gauss_seidel", &PoissonSolver2D::solve_gauss_seidel, 
             py::arg("max_iters"), 
             py::arg("tol"))
        .def("solve_sor", &PoissonSolver2D::solve_sor, 
             py::arg("max_iters"), 
             py::arg("tol"), 
             py::arg("omega"))
        .def("solve_with_refinement", &PoissonSolver2D::solve_with_refinement, 
             py::arg("lower_n"), 
             py::arg("max_iters"), 
             py::arg("tol"), 
             py::arg("omega"))

        // getters
        .def("get_energy_history", &PoissonSolver2D::get_energy_history)
        .def("get_residual_history", &PoissonSolver2D::get_residual_history)
     .def("get_refinement_timing_history", &PoissonSolver2D::get_refinement_timing_history)
        .def("get_solution", &PoissonSolver2D::get_solution)
        .def("get_solution_history_refinement", &PoissonSolver2D::get_solution_history_refinement)
        .def("compute_energy", &PoissonSolver2D::compute_energy)

        // setters
        .def("set_track_solution_history", &PoissonSolver2D::set_track_solution_history)
        .def("set_log_every", &PoissonSolver2D::set_log_every)
        .def("set_history_stride", &PoissonSolver2D::set_history_stride)
        .def("set_solution_from_coarse", &PoissonSolver2D::set_solution_from_coarse, 
             py::arg("coarse_sol"), 
             py::arg("coarse_n_power"));
}
