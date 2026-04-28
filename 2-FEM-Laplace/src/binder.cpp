#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FEM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FEM, m) {
    m.doc() = "FEM Laplace solver with Bilinear and Quadratic basis functions";

    py::enum_<BasisType>(m, "BasisType")
        .value("Bilinear",  BasisType::Bilinear)
        .value("Quadratic", BasisType::Quadratic)
        .export_values();

    py::class_<FEM>(m, "FEM")
        .def(py::init<double, size_t, double, double, BasisType>(),
             py::arg("L"),
             py::arg("N"),
             py::arg("rc_x"),
             py::arg("rc_y"),
             py::arg("basis") = BasisType::Bilinear,
             "Initialize FEM solver\n"
             "Args:\n"
             "  L: Domain size [0, L] × [0, L]\n"
             "  N: Number of elements per side\n"
             "  rc_x, rc_y: Singularity centre coordinates\n"
             "  basis: BasisType.Bilinear or BasisType.Quadratic")

        // ── Pipeline ────────────────────────────────────────────────────
        .def("build_mesh",                 &FEM::build_mesh)
        .def("assemble_stiffness_matrix",  &FEM::assemble_stiffness_matrix)
        .def("apply_boundary_conditions",  &FEM::apply_boundary_conditions)
        .def("solve",                      &FEM::solve,
             "Sparse Conjugate Gradient solve on GPU (works for any N)")

        // ── Evaluation ──────────────────────────────────────────────────
        .def("evaluate",
             &FEM::evaluate, py::arg("x"), py::arg("y"),
             "Evaluate the FEM solution at point (x, y)")

        .def("exact_solution",
             &FEM::exact_solution, py::arg("x"), py::arg("y"),
             "Evaluate the analytical Coulomb potential at point (x, y)")

        // ── Logging ─────────────────────────────────────────────────────
        .def("set_log_level",
             &FEM::set_log_level, py::arg("level"),
             "Set spdlog log level (e.g. spdlog::level::debug)")

        // ── Scalar getters ──────────────────────────────────────────────
        .def("get_a",         &FEM::get_a,         "Element side length a = L/N")
        .def("get_N",         &FEM::get_N,         "Number of elements per side")
        .def("get_L",         &FEM::get_L,         "Domain size L")
        .def("get_num_nodes", &FEM::get_num_nodes, "Total number of nodes")

        // ── Solution vectors ────────────────────────────────────────────
        .def("get_coefficients",
             &FEM::get_coefficients,
             "Solution coefficient vector c (one entry per node)",
             py::return_value_policy::reference_internal)

        .def("get_rhs",
             &FEM::get_rhs,
             "Right-hand side vector b",
             py::return_value_policy::reference_internal)

        // ── Sparse matrix (CSR) — replaces the old dense getter ─────────
        // To reconstruct in Python:
        //   import scipy.sparse as sp, numpy as np
        //   A = sp.csr_matrix((np.array(fem.get_csr_val()),
        //                      np.array(fem.get_csr_col()),
        //                      np.array(fem.get_csr_row())),
        //                     shape=(n, n))
        .def("get_csr_row",
             &FEM::get_csr_row,
             "CSR row-pointer array (length num_nodes+1)",
             py::return_value_policy::reference_internal)

        .def("get_csr_col",
             &FEM::get_csr_col,
             "CSR column-index array (length nnz)",
             py::return_value_policy::reference_internal)

        .def("get_csr_val",
             &FEM::get_csr_val,
             "CSR value array (length nnz)",
             py::return_value_policy::reference_internal)

        // ── Mesh data ───────────────────────────────────────────────────
        .def("get_nlg",
             &FEM::get_nlg,
             "Node local-to-global map: nlg[k*nodes_per_element + m] = global index",
             py::return_value_policy::reference_internal)

        .def("get_nodeX",
             &FEM::get_nodeX,
             "x-coordinates of all nodes",
             py::return_value_policy::reference_internal)

        .def("get_nodeY",
             &FEM::get_nodeY,
             "y-coordinates of all nodes",
             py::return_value_policy::reference_internal);
}
