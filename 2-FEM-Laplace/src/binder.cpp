#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FEM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FEM, m) {
    m.doc() = "FEM Laplace solver with Bilinear and Quadratic basis functions";

    // Export BasisType enum
    py::enum_<BasisType>(m, "BasisType")
        .value("Bilinear", BasisType::Bilinear)
        .value("Quadratic", BasisType::Quadratic)
        .export_values();

    // Export FEM class
    py::class_<FEM>(m, "FEM")
        // Constructor
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
             "  rc_x, rc_y: Singularity center coordinates\n"
             "  basis: BasisType.Bilinear or BasisType.Quadratic")

        // Main pipeline methods
        .def("build_mesh",
             &FEM::build_mesh,
             "Build the finite element mesh")

        .def("assemble_stiffness_matrix",
             &FEM::assemble_stiffness_matrix,
             "Assemble the global stiffness matrix")

        .def("apply_boundary_conditions",
             &FEM::apply_boundary_conditions,
             "Apply Dirichlet boundary conditions")

        .def("solve",
             &FEM::solve,
             "Solve the linear system using Gaussian elimination")

        // Solution evaluation
        .def("evaluate",
             &FEM::evaluate,
             py::arg("x"),
             py::arg("y"),
             "Evaluate the FEM solution at point (x, y)")

        .def("exact_solution",
             &FEM::exact_solution,
             py::arg("x"),
             py::arg("y"),
             "Evaluate the analytical Coulomb potential at point (x, y)")
        .def("set_log_level",
             &FEM::set_log_level,
             py::arg("level"),
             "Set spdlog log level (e.g., spdlog::level::info)")

        // Getters
        .def("get_a",
             &FEM::get_a,
             "Get element side length a = L/N")

        .def("get_N",
             &FEM::get_N,
             "Get number of elements per side")

        .def("get_L",
             &FEM::get_L,
             "Get domain size L")

        .def("get_num_nodes",
             &FEM::get_num_nodes,
             "Get total number of nodes")

        .def("get_coefficients",
             &FEM::get_coefficients,
             "Get solution coefficients vector",
             py::return_value_policy::reference_internal)

        .def("get_rhs",
             &FEM::get_rhs,
             "Get right-hand side vector",
             py::return_value_policy::reference_internal)

        .def("get_stiffness_matrix",
             &FEM::get_stiffness_matrix,
             "Get global stiffness matrix",
             py::return_value_policy::reference_internal)

        .def("get_nlg",
             &FEM::get_nlg,
             "Get node-local-to-global mapping array",
             py::return_value_policy::reference_internal)

        .def("get_nodeX",
             &FEM::get_nodeX,
             "Get x-coordinates of all nodes",
             py::return_value_policy::reference_internal)

        .def("get_nodeY",
             &FEM::get_nodeY,
             "Get y-coordinates of all nodes",
             py::return_value_policy::reference_internal);
}
