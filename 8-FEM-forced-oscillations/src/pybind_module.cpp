#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FD_solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fd_solver, m)
{
    m.doc() = "Finite difference solver bindings";

    py::class_<FD_solver>(m, "FD_solver")
        .def(py::init<size_t, double, double, double, double>(),
             py::arg("N"), py::arg("L"), py::arg("v"), py::arg("damping"), py::arg("omega"))
        .def("step", &FD_solver::step)
        .def("evolve", &FD_solver::evolve, py::arg("t_final"))
        .def("get_energy_history", &FD_solver::get_energy_history)
        .def("get_displacement_history", &FD_solver::get_displacement_history);
}
