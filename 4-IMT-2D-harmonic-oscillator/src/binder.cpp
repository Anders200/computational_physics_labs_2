#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "itm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(itm, m)
{
    m.doc() = "Imaginary Time Method solver for the 2D anisotropic harmonic oscillator";

    py::class_<Solver>(m, "Solver")
        .def(py::init<size_t, double, double, double, size_t, double>(),
             py::arg("N")            = 100,
             py::arg("alpha_factor") = 0.9,
             py::arg("omega_x")     = 1.0,
             py::arg("omega_y")     = 2.0,
             py::arg("max_iter")    = 10000,
             py::arg("tol")         = 1e-10,
             R"doc(
             Parameters
             ----------
             N            : grid points per side (N*dx = 4)
             alpha_factor : multiplier on alpha_c  (0.9 = safe, >1 diverges)
             omega_x/y    : trap frequencies (atomic units)
             max_iter     : hard iteration cap
             tol          : convergence threshold on |ΔE|
             )doc")

        .def("solve", &Solver::solve,
             py::arg("state_idx") = 0,
             R"doc(
             Compute eigenstates 0 … state_idx via ITM + Gram-Schmidt.
             Returns the per-iteration energy history of the *last* state solved.
             )doc")

        .def("get_potential",     &Solver::get_potential,
             "Flat N×N potential array (row-major)")
        .def("get_wavefunction",  &Solver::get_wavefunction,
             "Flat N×N wavefunction of the last converged state (row-major)")
        .def("get_state_energies",&Solver::get_state_energies,
             "Converged energies for every state found so far")
        .def("get_states",        &Solver::get_states,
             "All converged wavefunctions (list of flat N×N arrays, row-major)")
        .def("set_state_energies", &Solver::set_state_energies,
             "Replace the stored state energies from Python")
        .def("set_states",        &Solver::set_states,
             "Replace the stored converged wavefunctions from Python")
        .def("clear_states",      &Solver::clear_states,
             "Clear stored energies and wavefunctions")
        .def("num_states",        &Solver::num_states,
             "Number of stored states")
        .def("grid_size",         &Solver::grid_size, "N")
        .def("grid_dx",           &Solver::grid_dx,   "Grid spacing dx = 4/N");
}
