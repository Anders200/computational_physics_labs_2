#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "TdSE.hpp"
#include <pybind11/complex.h>

namespace py = pybind11;

PYBIND11_MODULE(TdSE, m)
{
    m.doc() = "Time dependent shroedinger equation based on previous solution from itm module";

    py::class_<tdse>(m, "tdse")
        .def(py::init<size_t, double, double, double, double>(),
             py::arg("N")            = 100,
             py::arg("dt")          = 0.1,
             py::arg("omega_x")     = 1.0,
             py::arg("omega_y")     = 1.001,
             py::arg("m") = 1.0,
             R"doc(
             Parameters
             ----------
             N            : grid points per side (N*dx = 4)
             dt 
             omega_x/y    : trap frequencies (atomic units)
             )doc")
        .def("set_initial_state",
            &tdse::set_initial_state,
            py::arg("psi0"),
            py::arg("psi1"),
            py::arg("psi2"))
        .def("evolve",
            &tdse::evolve,
            py::arg("total_steps"))
        .def("get_norm_history", &tdse::get_norm_history)
        .def("get_energy_history", &tdse::get_energy_history)
        .def("get_x_avg_history", &tdse::get_x_avg_history)
        .def("get_y_avg_history", &tdse::get_y_avg_history)
        .def("get_psi", &tdse::get_psi);

}
