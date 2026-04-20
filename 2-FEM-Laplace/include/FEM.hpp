#pragma once
#include <vector>
#include <cstddef>
#include "spdlog/spdlog.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

enum class BasisType { Bilinear, Quadratic };

class FEM
{
public:
    FEM(double L, size_t N, double rc_x, double rc_y,
        BasisType basis = BasisType::Bilinear);
    
    ~FEM();  // Destructor to clean up CUDA memory

    // Main pipeline — call in this order
    void build_mesh();
    void assemble_stiffness_matrix();
    void apply_boundary_conditions();
    void solve();

    // Evaluate the FEM solution at an arbitrary point inside [0,L]x[0,L]
    double evaluate(double x, double y) const;

    // Analytical 2-D Coulomb potential: Ψ = -1/(2π) ln|r - rc|
    double exact_solution(double x, double y) const;

    // ---- Getters --------------------------------------------------------
    double                              get_a()                const { return a; }
    size_t                              get_N()                const { return N; }
    double                              get_L()                const { return L; }
    size_t                              get_num_nodes()        const { return num_nodes; }
    const std::vector<double>&          get_coefficients()     const { return coefficients; }
    const std::vector<double>&          get_rhs()              const { return rhs; }
    const std::vector<double>&       get_stiffness_matrix() const { return stiffness_matrix; }
    
    // Helper: access stiffness matrix element at row i, col j (row-major indexing)
    inline double& S(size_t i, size_t j) { return stiffness_matrix[i * num_nodes + j]; }
    inline const double& S(size_t i, size_t j) const { return stiffness_matrix[i * num_nodes + j]; }
    // nlg[k * nodes_per_element + local_idx] = global node index
    const std::vector<size_t>&          get_nlg()              const { return nlg; }
    const std::vector<double>&          get_nodeX()            const { return nodeX_; }
    const std::vector<double>&          get_nodeY()            const { return nodeY_; }

    void set_log_level(spdlog::level::level_enum level) { spdlog::set_level(level); }

private:
    // ---- Problem parameters ---------------------------------------------
    double     L;
    size_t     N;
    double     rc_x, rc_y;
    double     a;                   // element side length = L / N
    int        nodes_per_element;   // 4 (bilinear) or 9 (quadratic)
    BasisType  basis_;
    size_t     num_nodes;

    // ---- Mesh data ------------------------------------------------------
    // nlg[k * nodes_per_element + m] = global index of local node m in element k
    std::vector<size_t> nlg;
    std::vector<double> nodeX_;
    std::vector<double> nodeY_;

    // ---- FEM arrays -----------------------------------------------------
    std::vector<double>              coefficients;
    std::vector<double>              rhs;
    std::vector<double>              stiffness_matrix;  // 1D row-major: stiffness_matrix[i*num_nodes + j]

    // Precomputed local stiffness (same for every element on a uniform mesh)
    std::vector<std::vector<double>> localStiffness_;
    
    // CUDA/CUBLAS data
    double* d_A = nullptr;           // Device stiffness matrix (flat)
    double* d_b = nullptr;           // Device RHS vector
    double* d_x = nullptr;           // Device solution vector
    cublasHandle_t cublas_handle;    // CUBLAS handle

    // ---- Internal helpers -----------------------------------------------
    void   initialize();
    void   compute_local_stiffness();

    int    find_node(double x, double y) const;   // -1 if absent
    int    add_node (double x, double y);          // returns new global index

    bool   is_boundary_node(size_t i) const;

    // Reference-space shape function and its ξ-derivatives (FD, Δ=0.1)
    double shape_func       (int node_idx, double xi1, double xi2) const;
    double shape_deriv_xi1  (int node_idx, double xi1, double xi2) const;
    double shape_deriv_xi2  (int node_idx, double xi1, double xi2) const;
};
