#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "FEM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FEM_solver, m)
{
    m.doc() = "FEM advection-diffusion solver (Q1 elements, Crank-Nicolson)";

    // ------------------------------------------------------------------
    //  SimResult  — plain data struct returned by the high-level runners
    // ------------------------------------------------------------------
    py::class_<SimResult>(m, "SimResult")
        .def_readonly("time",     &SimResult::time,
                      "Simulation time at each snapshot")
        .def_readonly("u_min",    &SimResult::u_min,
                      "Minimum nodal value at each snapshot")
        .def_readonly("u_max",    &SimResult::u_max,
                      "Maximum nodal value at each snapshot")
        .def_readonly("center_x", &SimResult::center_x,
                      "Mass-weighted centroid x at each snapshot")
        .def_readonly("center_y", &SimResult::center_y,
                      "Mass-weighted centroid y at each snapshot")
        .def_readonly("x_nodes",  &SimResult::x_nodes,
                      "x coordinate of every node (fixed)")
        .def_readonly("y_nodes",  &SimResult::y_nodes,
                      "y coordinate of every node (fixed)")
        .def_readonly("u_final",  &SimResult::u_final,
                      "Nodal solution vector at the final snapshot");

    // ------------------------------------------------------------------
    //  FEMSolver
    // ------------------------------------------------------------------
    py::class_<FEMSolver>(m, "FEMSolver")
        .def(py::init<int, double>(),
             py::arg("N"), py::arg("L"),
             R"pbdoc(
             Create a Q1 FEM solver on an N×N periodic mesh of side L.

             Parameters
             ----------
             N : int
                 Number of cells per axis.  DOF count = N^2.
             L : float
                 Domain side length.
             )pbdoc")

        // --- setup ---
        .def("assemble", &FEMSolver::assemble,
             py::arg("vx"), py::arg("vy"), py::arg("D"),
             "Assemble O, S, C matrices for the given parameters.")

        .def("set_initial_condition", &FEMSolver::setInitialCondition,
             py::arg("cx"), py::arg("cy"), py::arg("k") = 2.0,
             "Set Gaussian initial condition centred at (cx,cy) with exponent k.")

        // --- manual time stepping ---
        .def("step", &FEMSolver::step, py::arg("dt"),
             "Advance solution by one Crank-Nicolson step of size dt.")

        // --- diagnostics ---
        .def("min_u",    &FEMSolver::minU,    "Minimum nodal value.")
        .def("max_u",    &FEMSolver::maxU,    "Maximum nodal value.")
        .def("center_x", &FEMSolver::centerX, "Mass-weighted centroid x.")
        .def("center_y", &FEMSolver::centerY, "Mass-weighted centroid y.")
        .def_property_readonly("time", &FEMSolver::time, "Current simulation time.")
        .def_property_readonly("x_nodes", &FEMSolver::xNodes,
                               "Node x coordinates (list).")
        .def_property_readonly("y_nodes", &FEMSolver::yNodes,
                               "Node y coordinates (list).")
        .def_property_readonly("u",
            [](const FEMSolver& s) {
                // Return a copy as a plain Python list so NumPy can take it
                const auto& v = s.u();
                return std::vector<double>(v.data(), v.data() + v.size());
            },
            "Current nodal solution vector.")

        // --- high-level runners ---
        .def("run_pure_advection", &FEMSolver::runPureAdvection,
             py::arg("vx"), py::arg("vy"), py::arg("dt"),
             py::arg("steps_per_snapshot") = 10,
             R"pbdoc(
             Run pure-advection (D=0) until the density centre returns to its
             initial position a second time.

             Returns a SimResult with time-series diagnostics and the final
             field snapshot.
             )pbdoc")

        .def("run_pure_diffusion", &FEMSolver::runPureDiffusion,
             py::arg("D"), py::arg("t_end"), py::arg("dt"),
             py::arg("steps_per_snapshot") = 10,
             R"pbdoc(
             Run pure-diffusion (vx=vy=0) up to t_end.

             Returns SimResult with min/max u vs time.
             )pbdoc")

        .def("run_advection_diffusion", &FEMSolver::runAdvectionDiffusion,
             py::arg("vx"), py::arg("vy"), py::arg("D"),
             py::arg("t_end"), py::arg("dt"),
             py::arg("steps_per_snapshot") = 10,
             R"pbdoc(
             Run combined advection-diffusion up to t_end.

             Returns SimResult including centroid trajectory and final field.
             )pbdoc");
}
