#include "FEM.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>

// ============================================================
//  Helpers
// ============================================================
static constexpr double PI = M_PI;

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); \
} while(0)

#define CUSPARSE_CHECK(x) do { \
    cusparseStatus_t e = (x); \
    if (e != CUSPARSE_STATUS_SUCCESS) \
        throw std::runtime_error("cuSPARSE error " + std::to_string(e)); \
} while(0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t e = (x); \
    if (e != CUBLAS_STATUS_SUCCESS) \
        throw std::runtime_error("cuBLAS error " + std::to_string(e)); \
} while(0)

// ============================================================
//  1-D basis functions
// ============================================================
static inline double f1(double xi) { return (1.0 - xi) * 0.5; }
static inline double f2(double xi) { return (1.0 + xi) * 0.5; }
static inline double q1(double xi) { return xi * (xi - 1.0) * 0.5; }
static inline double q2(double xi) { return (1.0 - xi) * (1.0 + xi); }
static inline double q3(double xi) { return xi * (xi + 1.0) * 0.5; }

// ============================================================
//  Constructor / Destructor
// ============================================================
FEM::FEM(double L, size_t N, double rc_x, double rc_y, BasisType basis)
    : L(L), N(N), rc_x(rc_x), rc_y(rc_y), basis_(basis)
{
    nodes_per_element = (basis_ == BasisType::Bilinear) ? 4 : 9;
    num_nodes = (basis_ == BasisType::Bilinear) ? (N+1)*(N+1)
                                                 : (2*N+1)*(2*N+1);
    a = L / static_cast<double>(N);

    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
    spdlog::info("FEM created: N={}, basis={}, nodes={}", N,
                 basis_ == BasisType::Bilinear ? "Bilinear" : "Quadratic",
                 num_nodes);
}

FEM::~FEM() { free_gpu(); }

void FEM::free_gpu()
{
    if (spmv_buffer)   { cudaFree(spmv_buffer);   spmv_buffer  = nullptr; }
    if (vec_Ap_descr)  { cusparseDestroyDnVec(vec_Ap_descr); vec_Ap_descr = nullptr; }
    if (vec_p_descr)   { cusparseDestroyDnVec(vec_p_descr);  vec_p_descr  = nullptr; }
    if (mat_descr)     { cusparseDestroySpMat(mat_descr);     mat_descr    = nullptr; }
    if (d_csr_val)     { cudaFree(d_csr_val);  d_csr_val = nullptr; }
    if (d_csr_col)     { cudaFree(d_csr_col);  d_csr_col = nullptr; }
    if (d_csr_row)     { cudaFree(d_csr_row);  d_csr_row = nullptr; }
    if (d_Ap)          { cudaFree(d_Ap);        d_Ap      = nullptr; }
    if (d_p)           { cudaFree(d_p);         d_p       = nullptr; }
    if (d_r)           { cudaFree(d_r);         d_r       = nullptr; }
    if (d_b)           { cudaFree(d_b);         d_b       = nullptr; }
    if (d_x)           { cudaFree(d_x);         d_x       = nullptr; }
    if (cusparse_handle) { cusparseDestroy(cusparse_handle); cusparse_handle = nullptr; }
    if (cublas_handle)   { cublasDestroy(cublas_handle);     cublas_handle   = nullptr; }
}

// ============================================================
//  Exact solution
// ============================================================
double FEM::exact_solution(double x, double y) const
{
    double dx = x - rc_x, dy = y - rc_y;
    return -1.0 / (2.0 * PI) * std::log(std::sqrt(dx*dx + dy*dy));
}

// ============================================================
//  Shape functions  (reference coords ξ ∈ [-1,1])
// ============================================================
double FEM::shape_func(int m, double xi1, double xi2) const
{
    if (basis_ == BasisType::Bilinear) {
        switch (m) {
            case 0: return f1(xi1)*f1(xi2);
            case 1: return f2(xi1)*f1(xi2);
            case 2: return f1(xi1)*f2(xi2);
            case 3: return f2(xi1)*f2(xi2);
        }
    } else {
        switch (m) {
            case 0: return q1(xi1)*q1(xi2);
            case 1: return q3(xi1)*q1(xi2);
            case 2: return q1(xi1)*q3(xi2);
            case 3: return q3(xi1)*q3(xi2);
            case 4: return q2(xi1)*q1(xi2);
            case 5: return q3(xi1)*q2(xi2);
            case 6: return q1(xi1)*q2(xi2);
            case 7: return q2(xi1)*q3(xi2);
            case 8: return q2(xi1)*q2(xi2);
        }
    }
    return 0.0;
}

double FEM::shape_deriv_xi1(int m, double xi1, double xi2) const
{
    constexpr double D = 0.1;
    return (shape_func(m, xi1+D, xi2) - shape_func(m, xi1-D, xi2)) / (2.0*D);
}

double FEM::shape_deriv_xi2(int m, double xi1, double xi2) const
{
    constexpr double D = 0.1;
    return (shape_func(m, xi1, xi2+D) - shape_func(m, xi1, xi2-D)) / (2.0*D);
}

// ============================================================
//  Node helpers
// ============================================================
int FEM::find_node(double x, double y) const
{
    constexpr double TOL = 1e-10;
    for (size_t i = 0; i < nodeX_.size(); ++i)
        if (std::abs(nodeX_[i]-x) < TOL && std::abs(nodeY_[i]-y) < TOL)
            return static_cast<int>(i);
    return -1;
}
int FEM::add_node(double x, double y)
{
    nodeX_.push_back(x);
    nodeY_.push_back(y);
    return static_cast<int>(nodeX_.size()) - 1;
}
bool FEM::is_boundary_node(size_t i) const
{
    constexpr double TOL = 1e-10;
    double x = nodeX_[i], y = nodeY_[i];
    return x < TOL || x > L-TOL || y < TOL || y > L-TOL;
}

// ============================================================
//  initialize()
// ============================================================
void FEM::initialize()
{
    nlg.resize(N * N * static_cast<size_t>(nodes_per_element));
    nodeX_.clear();  nodeX_.reserve(num_nodes);
    nodeY_.clear();  nodeY_.reserve(num_nodes);
    coefficients.assign(num_nodes, 0.0);
    rhs.assign(num_nodes, 0.0);
    triplets_.clear();
    csr_row_.clear(); csr_col_.clear(); csr_val_.clear();
}

// ============================================================
//  Local stiffness matrix (Gauss quadrature, same for all elements)
// ============================================================
void FEM::compute_local_stiffness()
{
    const double sq30  = std::sqrt(30.0);
    const double sq65  = std::sqrt(6.0/5.0);
    const double inner = std::sqrt(3.0/7.0 - 2.0/7.0*sq65);
    const double outer = std::sqrt(3.0/7.0 + 2.0/7.0*sq65);

    const double w[4]   = { (18.0+sq30)/36.0, (18.0+sq30)/36.0,
                             (18.0-sq30)/36.0, (18.0-sq30)/36.0 };
    const double gam[4] = { -inner, +inner, +outer, -outer };

    localStiffness_.assign(nodes_per_element,
                           std::vector<double>(nodes_per_element, 0.0));

    for (int l = 0; l < 4; ++l)
        for (int n = 0; n < 4; ++n) {
            double ww = w[l]*w[n];
            for (int j = 0; j < nodes_per_element; ++j) {
                double dj1 = shape_deriv_xi1(j, gam[l], gam[n]);
                double dj2 = shape_deriv_xi2(j, gam[l], gam[n]);
                for (int i = 0; i < nodes_per_element; ++i)
                    localStiffness_[j][i] +=
                        ww * (dj1*shape_deriv_xi1(i, gam[l], gam[n])
                            + dj2*shape_deriv_xi2(i, gam[l], gam[n]));
            }
        }
}

// ============================================================
//  build_mesh()
// ============================================================
void FEM::build_mesh()
{
    initialize();
    spdlog::info("Building mesh: N={} nodes_per_element={} total_nodes={}",
                 N, nodes_per_element, num_nodes);

    const double offsets[9][2] = {
        {0.0,    0.0   }, {a,    0.0   }, {0.0,  a    }, {a,    a    },
        {a/2.0,  0.0   }, {a,    a/2.0 }, {0.0,  a/2.0}, {a/2.0, a  },
        {a/2.0,  a/2.0 }
    };

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            size_t k = i*N + j;
            double ax = j*a, ay = i*a;
            for (int m = 0; m < nodes_per_element; ++m) {
                double x = ax + offsets[m][0], y = ay + offsets[m][1];
                int ex = find_node(x, y);
                nlg[k*nodes_per_element + m] =
                    (ex != -1) ? static_cast<size_t>(ex)
                               : static_cast<size_t>(add_node(x, y));
            }
        }

    if (nodeX_.size() != num_nodes)
        throw std::runtime_error(
            "Node count mismatch: expected " + std::to_string(num_nodes) +
            ", got " + std::to_string(nodeX_.size()));

    spdlog::info("Mesh built: {} nodes, {} elements.", nodeX_.size(), N*N);
    compute_local_stiffness();
}

// ============================================================
//  assemble_stiffness_matrix()
//  Accumulate into the triplet map; no dense allocation.
// ============================================================
void FEM::assemble_stiffness_matrix()
{
    spdlog::info("Assembling sparse stiffness matrix...");
    triplets_.clear();

    for (size_t k = 0; k < N*N; ++k)
        for (int i1 = 0; i1 < nodes_per_element; ++i1) {
            int g1 = static_cast<int>(nlg[k*nodes_per_element + i1]);
            for (int i2 = 0; i2 < nodes_per_element; ++i2) {
                int g2 = static_cast<int>(nlg[k*nodes_per_element + i2]);
                triplet_add(g1, g2, localStiffness_[i1][i2]);
            }
        }

    build_csr_from_triplets();
    spdlog::info("Sparse matrix: {} nodes, {} nonzeros (density {:.4f}%)",
                 num_nodes, csr_val_.size(),
                 100.0*csr_val_.size() / (double)(num_nodes*num_nodes));
}

void FEM::build_csr_from_triplets()
{
    int n = static_cast<int>(num_nodes);
    csr_row_.assign(n + 1, 0);
    csr_col_.clear();
    csr_val_.clear();

    // Count entries per row
    for (auto& [key, _] : triplets_)
        csr_row_[key.first + 1]++;
    // Prefix sum
    for (int i = 0; i < n; ++i)
        csr_row_[i+1] += csr_row_[i];

    int nnz = static_cast<int>(triplets_.size());
    csr_col_.resize(nnz);
    csr_val_.resize(nnz);

    std::vector<int> pos(csr_row_.begin(), csr_row_.begin() + n);
    for (auto& [key, val] : triplets_) {
        int p = pos[key.first]++;
        csr_col_[p] = key.second;
        csr_val_[p] = val;
    }
    triplets_.clear();   // free host memory
}

// ============================================================
//  apply_boundary_conditions()
//  Correct symmetric enforcement: subtract BC columns from interior RHS.
// ============================================================
void FEM::apply_boundary_conditions()
{
    spdlog::info("Applying Dirichlet BCs...");

    std::vector<bool>   is_bc(num_nodes, false);
    std::vector<double> bc_val(num_nodes, 0.0);
    for (size_t i = 0; i < num_nodes; ++i)
        if (is_boundary_node(i)) {
            is_bc[i]  = true;
            bc_val[i] = exact_solution(nodeX_[i], nodeY_[i]);
        }

    std::fill(rhs.begin(), rhs.end(), 0.0);

    // Walk CSR to subtract boundary column contributions from interior RHS,
    // and to zero off-diagonal BC entries (row and column simultaneously).
    // We rebuild CSR in-place: keep diagonal of BC rows, zero all others.
    for (int row = 0; row < static_cast<int>(num_nodes); ++row) {
        for (int idx = csr_row_[row]; idx < csr_row_[row+1]; ++idx) {
            int col = csr_col_[idx];
            if (is_bc[col] && !is_bc[row]) {
                // Move A[row][col] * bc_val[col] to RHS
                rhs[row] -= csr_val_[idx] * bc_val[col];
                csr_val_[idx] = 0.0;
            }
            if (is_bc[row]) {
                // Zero entire boundary row except diagonal
                csr_val_[idx] = (row == col) ? 1.0 : 0.0;
            }
        }
    }

    // Also zero the symmetric (col=BC, row=interior) part — already done above
    // via the is_bc[col] branch. Set BC RHS.
    for (size_t i = 0; i < num_nodes; ++i)
        if (is_bc[i]) rhs[i] = bc_val[i];

    spdlog::info("BCs applied.");
}

// ============================================================
//  upload_sparse_matrix_to_gpu()
// ============================================================
void FEM::upload_sparse_matrix_to_gpu()
{
    int   n   = static_cast<int>(num_nodes);
    int   nnz = static_cast<int>(csr_val_.size());

    CUDA_CHECK(cudaMalloc(&d_csr_row, (n+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_col, nnz   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_val, nnz   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_csr_row, csr_row_.data(), (n+1)*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col, csr_col_.data(), nnz  *sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_val, csr_val_.data(), nnz  *sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,       rhs.data(),       n   *sizeof(double), cudaMemcpyHostToDevice));

    // Initial guess x = 0
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));

    // Create cuSPARSE descriptors for SpMV: Ap = A * p
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_descr, n, n, nnz,
        d_csr_row, d_csr_col, d_csr_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_p_descr,  n, d_p,  CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_Ap_descr, n, d_Ap, CUDA_R_64F));

    // Allocate SpMV buffer
    double one = 1.0, zero = 0.0;
    size_t buf_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, mat_descr, vec_p_descr, &zero, vec_Ap_descr,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));

    if (buf_size > 0)
        CUDA_CHECK(cudaMalloc(&spmv_buffer, buf_size));

    spdlog::info("Sparse matrix uploaded to GPU ({} nnz, ~{:.1f} MB).",
                 nnz, nnz * 8.0 / 1e6);
}

// ============================================================
//  solve()  — Conjugate Gradient on GPU
//
//  Works for any N because memory scales as O(nnz) ~ O(num_nodes),
//  not O(num_nodes²).
//
//  CG is valid here because the stiffness matrix is symmetric
//  positive definite (after BC enforcement it stays SPD).
//
//  Algorithm:
//    r₀ = b - A x₀  (x₀ = 0, so r₀ = b)
//    p₀ = r₀
//    for k = 0, 1, 2, ...
//      α_k   = (rᵀr) / (pᵀAp)
//      x_k+1 = x_k  + α_k p_k
//      r_k+1 = r_k  - α_k A p_k
//      β_k   = (r_k+1ᵀ r_k+1) / (rᵀr)
//      p_k+1 = r_k+1 + β_k p_k
// ============================================================
void FEM::solve()
{
    spdlog::info("Uploading sparse matrix to GPU...");
    upload_sparse_matrix_to_gpu();

    int    n   = static_cast<int>(num_nodes);
    double one = 1.0, m_one = -1.0, zero = 0.0;

    // r = b  (x₀ = 0)
    CUDA_CHECK(cudaMemcpy(d_r, d_b, n*sizeof(double), cudaMemcpyDeviceToDevice));
    // p = r
    CUDA_CHECK(cudaMemcpy(d_p, d_r, n*sizeof(double), cudaMemcpyDeviceToDevice));

    double rr_old;
    CUBLAS_CHECK(cublasDdot(cublas_handle, n, d_r, 1, d_r, 1, &rr_old));

    double b_norm;
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_b, 1, &b_norm));
    double tol_abs = cg_tol * b_norm;

    spdlog::info("CG starting: n={}, ||b||={:.4e}, tol={:.4e}",
                 n, b_norm, tol_abs);

    int iter = 0;
    for (; iter < cg_max_iter; ++iter) {
        // Ap = A * p
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat_descr, vec_p_descr, &zero, vec_Ap_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer));

        // α = rᵀr / (pᵀAp)
        double pAp;
        CUBLAS_CHECK(cublasDdot(cublas_handle, n, d_p, 1, d_Ap, 1, &pAp));
        double alpha = rr_old / pAp;

        // x = x + α p
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, n, &alpha, d_p, 1, d_x, 1));

        // r = r - α Ap
        double m_alpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, n, &m_alpha, d_Ap, 1, d_r, 1));

        double rr_new;
        CUBLAS_CHECK(cublasDdot(cublas_handle, n, d_r, 1, d_r, 1, &rr_new));

        if (iter % 200 == 0)
            spdlog::debug("CG iter {:5d}: ||r||={:.4e}", iter, std::sqrt(rr_new));

        if (std::sqrt(rr_new) < tol_abs) {
            iter++;
            break;
        }

        // β = rr_new / rr_old
        double beta = rr_new / rr_old;

        // p = r + β p
        CUBLAS_CHECK(cublasDscal(cublas_handle, n, &beta, d_p, 1));
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, n, &one,  d_r, 1, d_p, 1));

        rr_old = rr_new;
    }
    spdlog::info("CG converged in {} iterations.", iter);

    CUDA_CHECK(cudaMemcpy(coefficients.data(), d_x,
                          n*sizeof(double), cudaMemcpyDeviceToHost));
}

// ============================================================
//  evaluate(x, y)
// ============================================================
double FEM::evaluate(double x, double y) const
{
    int ej = std::clamp((int)(x/a), 0, (int)N-1);
    int ei = std::clamp((int)(y/a), 0, (int)N-1);
    size_t k = (size_t)ei * N + (size_t)ej;

    double xi1 = 2.0*(x - ej*a)/a - 1.0;
    double xi2 = 2.0*(y - ei*a)/a - 1.0;

    double result = 0.0;
    for (int m = 0; m < nodes_per_element; ++m)
        result += coefficients[nlg[k*nodes_per_element + m]]
                  * shape_func(m, xi1, xi2);
    return result;
}
