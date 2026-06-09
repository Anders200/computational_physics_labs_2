#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <spdlog/spdlog.h>

#include "solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(membrane, m)
{
     spdlog::info("membrane Python module loaded");
    m.doc() = R"pbdoc(
        membrane – CUDA-accelerated 2-D FEM wave-equation solver

        Solves
            d²Ψ/dt² = v² ∇²Ψ
        on a square membrane of side L with Dirichlet boundary conditions
        and a sinusoidally driven centre node.

        The spatial discretisation uses bilinear Q4 finite elements on a
        uniform (N+1)×(N+1) grid.  Time integration uses the explicit
        leapfrog (Verlet) scheme:

            O c(t+Δt) = (-v²Δt² S + 2 O) c(t) – O c(t–Δt)

        where S is the stiffness matrix and O the overlap (mass) matrix.
        The linear system  O x = rhs  is solved at every step via
        cuSolverSp's direct LU factorisation.
    )pbdoc";

    py::class_<Solver>(m, "Solver")

        .def(py::init<double, double, double, double, std::size_t>(),
             py::arg("L"),
             py::arg("v"),
             py::arg("omega"),
             py::arg("dt"),
             py::arg("N"),
             R"pbdoc(
                 Create a new Solver.

                 Parameters
                 ----------
                 L     : float  – side length of the membrane
                 v     : float  – wave speed (c)
                 omega : float  – angular driving frequency ω
                 dt    : float  – time step Δt
                 N     : int    – number of mesh intervals per side
                                  (total nodes = (N+1)²)
             )pbdoc")

        .def("initialize_mesh", &Solver::initialize_mesh,
             R"pbdoc(
                 Assemble the finite-element matrices (S and O),
                 set up boundary conditions, and upload everything
                 to the GPU.  Must be called once before any time stepping.
             )pbdoc")

        .def("step", &Solver::step,
             "Advance the simulation by one time step Δt.")

        .def("advance", &Solver::advance,
             py::arg("n_steps"),
             "Advance the simulation by n_steps time steps.")

        .def("get_time",          &Solver::get_time,
             "Return the current simulation time.")

        .def("get_nodes_per_side",&Solver::get_nodes_per_side,
             "Return the number of nodes along one side: N+1.")

        .def("get_total_nodes",   &Solver::get_total_nodes,
             "Return the total number of nodes: (N+1)².")

        .def("get_field",         &Solver::get_field,
             R"pbdoc(
                 Return the current displacement field Ψ as a flat Python list
                 of length (N+1)².  Reshape to (N+1, N+1) for plotting:

                     import numpy as np
                     psi = np.array(s.get_field()).reshape(N+1, N+1)
             )pbdoc")

        .def("get_x_coords",      &Solver::get_x_coords,
             "Return the x-coordinates of all nodes (length = total_nodes).")

        .def("get_y_coords",      &Solver::get_y_coords,
             "Return the y-coordinates of all nodes (length = total_nodes).")

        .def("get_driving_node",  &Solver::get_driving_node,
             "Return the flat index of the centre (driven) node.")

        .def("field_norm",        &Solver::field_norm,
             "Return the L2 norm of the current displacement field.")

        .def("field_max",         &Solver::field_max,
             "Return the maximum absolute value in the current field.")

        .def("__repr__", [](const Solver& s) {
            return "<membrane.Solver  time=" + std::to_string(s.get_time()) +
                   "  nodes=" + std::to_string(s.get_total_nodes()) + ">";
        });

    m.def("cfl_dt", &Solver::cfl_dt,
          py::arg("L"), py::arg("v"), py::arg("N"),
          "Compute the CFL-stable time step: h / (v * sqrt(2)) where h = L/N.");
}
