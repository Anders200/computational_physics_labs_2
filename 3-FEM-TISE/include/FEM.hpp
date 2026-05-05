#pragma once
#include <vector>
#include <map>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "spdlog/spdlog.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>

enum class BasisType { Bilinear, Quadratic };

class FEM
{
public:
    FEM(double L, size_t N, double rc_x, double rc_y,
        BasisType basis = BasisType::Bilinear);
    ~FEM();

    void build_mesh();
    void assemble_stiffness_matrix();
    void apply_boundary_conditions();
    void solve();

    void assemble_overlap_matrix();
    void apply_bc_eigenvalue();

    double evaluate       (double x, double y) const;
    double exact_solution (double x, double y) const;

    void set_coefficients(const std::vector<double>& c) { coefficients = c; }

    double              get_a()         const { return a; }
    size_t              get_N()         const { return N; }
    double              get_L()         const { return L; }
    size_t              get_num_nodes() const { return num_nodes; }

    const std::vector<double>& get_coefficients() const { return coefficients; }
    const std::vector<double>& get_rhs()          const { return rhs; }
    const std::vector<size_t>& get_nlg()          const { return nlg; }
    const std::vector<double>& get_nodeX()        const { return nodeX_; }
    const std::vector<double>& get_nodeY()        const { return nodeY_; }

    const std::vector<int>&    get_csr_row() const { return csr_row_; }
    const std::vector<int>&    get_csr_col() const { return csr_col_; }
    const std::vector<double>& get_csr_val() const { return csr_val_; }

    const std::vector<int>&    get_overlap_csr_row() const { return overlap_csr_row_; }
    const std::vector<int>&    get_overlap_csr_col() const { return overlap_csr_col_; }
    const std::vector<double>& get_overlap_csr_val() const { return overlap_csr_val_; }

    void set_log_level(spdlog::level::level_enum lvl) { spdlog::set_level(lvl); }

    int    cg_max_iter = 20000;
    double cg_tol      = 1e-10;

private:
    double    L, a;
    size_t    N;
    double    rc_x, rc_y;
    int       nodes_per_element;
    BasisType basis_;
    size_t    num_nodes;

    std::vector<size_t> nlg;
    std::vector<double> nodeX_, nodeY_;

    std::vector<double> coefficients;
    std::vector<double> rhs;

    std::map<std::pair<int,int>, double> triplets_;
    std::vector<int>    csr_row_;
    std::vector<int>    csr_col_;
    std::vector<double> csr_val_;

    inline void triplet_add(int r, int c, double v) { triplets_[{r,c}] += v; }
    void build_csr_from_triplets();

    std::vector<std::vector<double>> localStiffness_;
    void compute_local_stiffness();

    std::map<std::pair<int,int>, double> overlap_triplets_;
    std::vector<int>    overlap_csr_row_;
    std::vector<int>    overlap_csr_col_;
    std::vector<double> overlap_csr_val_;

    inline void overlap_triplet_add(int r, int c, double v) { overlap_triplets_[{r,c}] += v; }
    void build_overlap_csr_from_triplets();

    std::vector<std::vector<double>> localOverlap_;
    void compute_local_overlap();

    int*    d_csr_row = nullptr;
    int*    d_csr_col = nullptr;
    double* d_csr_val = nullptr;
    double* d_x       = nullptr;
    double* d_b       = nullptr;
    double* d_r       = nullptr;
    double* d_p       = nullptr;
    double* d_Ap      = nullptr;

    cusparseHandle_t     cusparse_handle = nullptr;
    cublasHandle_t       cublas_handle   = nullptr;
    cusparseSpMatDescr_t mat_descr       = nullptr;
    cusparseDnVecDescr_t vec_p_descr     = nullptr;
    cusparseDnVecDescr_t vec_Ap_descr    = nullptr;
    void*                spmv_buffer     = nullptr;

    void upload_sparse_matrix_to_gpu();
    void free_gpu();

    // ---- Mesh helpers -------------------------------------------------------
    void initialize();
    int  find_node(double x, double y) const;
    int  add_node (double x, double y);
    bool is_boundary_node(size_t i) const;

    // ---- Reference-space shape functions ------------------------------------
    double shape_func      (int m, double xi1, double xi2) const;
    double shape_deriv_xi1 (int m, double xi1, double xi2) const;
    double shape_deriv_xi2 (int m, double xi1, double xi2) const;
};
