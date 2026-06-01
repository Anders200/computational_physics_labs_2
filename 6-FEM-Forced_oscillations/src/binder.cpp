#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FEM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FEM, m) {
    m.doc() = "FEM solver: 2-D Schrödinger eigenvalue problem with forced oscillations";

    py::enum_<BasisType>(m, "BasisType")
        .value("Bilinear",  BasisType::Bilinear)
        .value("Quadratic", BasisType::Quadratic)
        .export_values();

    py::class_<FEM>(m, "FEM")
        .def(py::init<double, size_t, double, double, BasisType>(),
             py::arg("L"), py::arg("N"),
             py::arg("rc_x") = 0.0, py::arg("rc_y") = 0.0,
             py::arg("basis") = BasisType::Bilinear)
        .def("build_mesh", &FEM::build_mesh)
        .def("assemble_stiffness_matrix", &FEM::assemble_stiffness_matrix)
        .def("assemble_overlap_matrix", &FEM::assemble_overlap_matrix)
        .def("apply_bc_eigenvalue", &FEM::apply_bc_eigenvalue)
        .def("get_num_nodes", &FEM::get_num_nodes)
        .def("get_csr_row", &FEM::get_csr_row, py::return_value_policy::reference_internal)
        .def("get_csr_col", &FEM::get_csr_col, py::return_value_policy::reference_internal)
        .def("get_csr_val", &FEM::get_csr_val, py::return_value_policy::reference_internal)
        .def("get_overlap_csr_row", &FEM::get_overlap_csr_row, py::return_value_policy::reference_internal)
        .def("get_overlap_csr_col", &FEM::get_overlap_csr_col, py::return_value_policy::reference_internal)
        .def("get_overlap_csr_val", &FEM::get_overlap_csr_val, py::return_value_policy::reference_internal)
        .def("assemble_potential_matrix", &FEM::assemble_potential_matrix, py::arg("eF"))
        .def("get_potential_csr_row", &FEM::get_potential_csr_row, py::return_value_policy::reference_internal)
        .def("get_potential_csr_col", &FEM::get_potential_csr_col, py::return_value_policy::reference_internal)
        .def("get_potential_csr_val", &FEM::get_potential_csr_val, py::return_value_policy::reference_internal);
}
