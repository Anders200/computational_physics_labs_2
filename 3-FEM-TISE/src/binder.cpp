#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FEM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FEM, m) {
    m.doc() = "FEM solver: 2-D Laplace (Poisson) and Schrödinger eigenvalue problem";

    py::enum_<BasisType>(m, "BasisType")
        .value("Bilinear",  BasisType::Bilinear)
        .value("Quadratic", BasisType::Quadratic)
        .export_values();

    py::class_<FEM>(m, "FEM")
        .def(py::init<double, size_t, double, double, BasisType>(),
             py::arg("L"),
             py::arg("N"),
             py::arg("rc_x") = 0.0,
             py::arg("rc_y") = 0.0,
             py::arg("basis") = BasisType::Bilinear,
             "Initialise FEM solver.\n"
             "Args:\n"
             "  L      : domain size [0,L]×[0,L]\n"
             "  N      : number of elements per side\n"
             "  rc_x,y : singularity centre (Laplace/Poisson mode only)\n"
             "  basis  : BasisType.Bilinear or BasisType.Quadratic")

        .def("build_mesh",                &FEM::build_mesh)
        .def("assemble_stiffness_matrix", &FEM::assemble_stiffness_matrix)

        .def("apply_boundary_conditions", &FEM::apply_boundary_conditions,
             "Dirichlet BC for Laplace/Poisson: set boundary values from exact_solution()")
        .def("solve", &FEM::solve,
             "Sparse Conjugate Gradient solve on GPU (Laplace/Poisson mode)")

        .def("assemble_overlap_matrix", &FEM::assemble_overlap_matrix,
             "Assemble the overlap (mass) matrix O_{ij} = <h_i|h_j>.\n"
             "Call after assemble_stiffness_matrix().")
        .def("apply_bc_eigenvalue", &FEM::apply_bc_eigenvalue,
             "Apply homogeneous Dirichlet BCs for the EVP H c = E O c.\n"
             "Zeros row+col of both K and O for boundary nodes;\n"
             "sets K_{ii}=-2 and O_{ii}=1 so the spurious eigenvalue is -1.")

        .def("evaluate",
             &FEM::evaluate, py::arg("x"), py::arg("y"),
             "Evaluate the FEM solution (or eigenvector) at point (x, y)")
        .def("exact_solution",
             &FEM::exact_solution, py::arg("x"), py::arg("y"),
             "Evaluate the analytical Coulomb potential at (x, y)  [Laplace mode]")

        .def("set_coefficients",
             &FEM::set_coefficients, py::arg("c"),
             "Set the coefficient vector (e.g. an eigenvector from Python)\n"
             "so that evaluate() renders the corresponding wave function.")

        .def("set_log_level",
             &FEM::set_log_level, py::arg("level"))

        .def("get_a",         &FEM::get_a)
        .def("get_N",         &FEM::get_N)
        .def("get_L",         &FEM::get_L)
        .def("get_num_nodes", &FEM::get_num_nodes)

        .def("get_coefficients",
             &FEM::get_coefficients,
             py::return_value_policy::reference_internal)
        .def("get_rhs",
             &FEM::get_rhs,
             py::return_value_policy::reference_internal)

        .def("get_csr_row",
             &FEM::get_csr_row,
             "CSR row-pointer array for the stiffness matrix K (len num_nodes+1)",
             py::return_value_policy::reference_internal)
        .def("get_csr_col",
             &FEM::get_csr_col,
             "CSR column-index array for K (len nnz)",
             py::return_value_policy::reference_internal)
        .def("get_csr_val",
             &FEM::get_csr_val,
             "CSR value array for K (len nnz)",
             py::return_value_policy::reference_internal)

        .def("get_overlap_csr_row",
             &FEM::get_overlap_csr_row,
             "CSR row-pointer array for the overlap matrix O (len num_nodes+1)",
             py::return_value_policy::reference_internal)
        .def("get_overlap_csr_col",
             &FEM::get_overlap_csr_col,
             "CSR column-index array for O (len nnz)",
             py::return_value_policy::reference_internal)
        .def("get_overlap_csr_val",
             &FEM::get_overlap_csr_val,
             "CSR value array for O (len nnz)",
             py::return_value_policy::reference_internal)

        .def("get_nlg",
             &FEM::get_nlg,
             py::return_value_policy::reference_internal)
        .def("get_nodeX",
             &FEM::get_nodeX,
             py::return_value_policy::reference_internal)
        .def("get_nodeY",
             &FEM::get_nodeY,
             py::return_value_policy::reference_internal);
}
