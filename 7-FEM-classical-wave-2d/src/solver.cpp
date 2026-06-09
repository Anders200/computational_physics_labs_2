#include "solver.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <spdlog/spdlog.h>

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(_e) +                  \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } while (0)

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t _s = (call);                                          \
        if (_s != CUSPARSE_STATUS_SUCCESS)                                     \
            throw std::runtime_error(std::string("cuSPARSE error code ") +    \
                                     std::to_string(static_cast<int>(_s)) +   \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } while (0)

struct Solver::CudaResources
{
    cublasHandle_t   cublas_handle   = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    double* d_S_val  = nullptr;  int* d_S_col  = nullptr;  int* d_S_row  = nullptr;
    double* d_O_val  = nullptr;  int* d_O_col  = nullptr;  int* d_O_row  = nullptr;

    cusparseSpMatDescr_t spmat_S = nullptr;
    cusparseSpMatDescr_t spmat_O = nullptr;

    double* d_cPrev    = nullptr;
    double* d_cCur     = nullptr;
    double* d_cNext    = nullptr;
    double* d_rhs      = nullptr;
    double* d_tmp      = nullptr;

    cusparseDnVecDescr_t dvec_cPrev = nullptr;
    cusparseDnVecDescr_t dvec_cCur  = nullptr;
    cusparseDnVecDescr_t dvec_rhs   = nullptr;
    cusparseDnVecDescr_t dvec_tmp   = nullptr;
    cusparseDnVecDescr_t dvec_cNext = nullptr;

    void*  spmv_buf_S   = nullptr;  std::size_t spmv_buf_S_sz   = 0;
    void*  spmv_buf_O1  = nullptr;  std::size_t spmv_buf_O1_sz  = 0;
    void*  spmv_buf_O2  = nullptr;  std::size_t spmv_buf_O2_sz  = 0;

    int n_nodes = 0;

    ~CudaResources()
    {
        if (spmat_S)      cusparseDestroySpMat(spmat_S);
        if (spmat_O)      cusparseDestroySpMat(spmat_O);
        if (dvec_cPrev)   cusparseDestroyDnVec(dvec_cPrev);
        if (dvec_cCur)    cusparseDestroyDnVec(dvec_cCur);
        if (dvec_rhs)     cusparseDestroyDnVec(dvec_rhs);
        if (dvec_tmp)     cusparseDestroyDnVec(dvec_tmp);
        if (dvec_cNext)   cusparseDestroyDnVec(dvec_cNext);
        cudaFree(d_S_val);  cudaFree(d_S_col);  cudaFree(d_S_row);
        cudaFree(d_O_val);  cudaFree(d_O_col);  cudaFree(d_O_row);
        cudaFree(d_cPrev);  cudaFree(d_cCur);
        cudaFree(d_cNext);  cudaFree(d_rhs);  cudaFree(d_tmp);
        cudaFree(spmv_buf_S);
        cudaFree(spmv_buf_O1);
        cudaFree(spmv_buf_O2);

        if (cublas_handle)   cublasDestroy(cublas_handle);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
    }
};
Solver::Solver(double L, double v, double F_omega, double dt, std::size_t N)
    : m_L(L), m_v(v), m_F_omega(F_omega), m_dt(dt),
      m_current_t(0.0), m_N(N),
      m_n_nodes((N + 1) * (N + 1))
{
        spdlog::info("Solver created: L={}, v={}, omega={}, dt={}, N={} (nodes={})",
                                 m_L, m_v, m_F_omega, m_dt, m_N, m_n_nodes);
}

Solver::~Solver()
{
    delete m_cuda;
}

double Solver::cfl_dt(double L, double v, std::size_t N)
{
    double h = L / static_cast<double>(N);
    return h / (v * std::sqrt(2.0));
}

void Solver::initialize_mesh()
{
    spdlog::info("Initializing mesh (N={}, nodes={})", m_N, m_n_nodes);
    double h = m_L / static_cast<double>(m_N);
    m_xCoords.resize(m_n_nodes);
    m_yCoords.resize(m_n_nodes);
    for (std::size_t i = 0; i <= m_N; ++i)
        for (std::size_t j = 0; j <= m_N; ++j) {
            std::size_t idx = node_id(i, j);
            m_xCoords[idx] = j * h;
            m_yCoords[idx] = i * h;
        }

    m_is_constrained.assign(m_n_nodes, false);
    m_boundaryNodes.clear();
    for (std::size_t i = 0; i <= m_N; ++i)
        for (std::size_t j = 0; j <= m_N; ++j) {
            if (i == 0 || i == m_N || j == 0 || j == m_N) {
                std::size_t idx = node_id(i, j);
                m_boundaryNodes.push_back(idx);
                m_is_constrained[idx] = true;
            }
        }

    m_driving_node = node_id(m_N / 2, m_N / 2);
    m_is_constrained[m_driving_node] = true;

    build_matrices();

    {
        const int n = static_cast<int>(m_n_nodes);
        m_O_eigen.resize(n, n);
        m_O_eigen.reserve(static_cast<int>(m_O_values.size()));

        for (std::size_t r = 0; r < m_n_nodes; ++r) {
            int start = m_O_rowPointers[r];
            int end = m_O_rowPointers[r + 1];
            for (int k = start; k < end; ++k) {
                m_O_eigen.insert(static_cast<int>(r), m_O_colIndices[k]) = m_O_values[k];
            }
        }
        m_O_eigen.makeCompressed();
        m_O_llt.compute(m_O_eigen);
        if (m_O_llt.info() != Eigen::Success) {
            throw std::runtime_error("Eigen SimplicialLLT factorization failed for O");
        }
        m_O_factorized = true;
    }

    m_cPrevious.assign(m_n_nodes, 0.0);
    m_cCurrent .assign(m_n_nodes, 0.0);
    m_cNext    .assign(m_n_nodes, 0.0);

    upload_to_device();
    spdlog::info("Mesh initialized and GPU resources uploaded");
}

void Solver::build_matrices()
{
    spdlog::info("Building FEM matrices (S and O)");
    const std::size_t NNodes = m_n_nodes;
    std::vector<std::vector<std::pair<std::size_t,double>>> S_coo(NNodes);
    std::vector<std::vector<std::pair<std::size_t,double>>> O_coo(NNodes);

    auto add_entry = [&](std::vector<std::vector<std::pair<std::size_t,double>>>& mat,
                         std::size_t r, std::size_t c, double val) {
        for (auto& p : mat[r])
            if (p.first == c) { p.second += val; return; }
        mat[r].emplace_back(c, val);
    };

    double h = m_L / static_cast<double>(m_N);

    double s_fac = 1.0 / (6.0 * h);
    double o_fac = (h * h) / 36.0;

    static const double S_local[4][4] = {
        { 4, -1, -2, -1},
        {-1,  4, -1, -2},
        {-2, -1,  4, -1},
        {-1, -2, -1,  4}
    };
    static const double O_local[4][4] = {
        {4, 2, 1, 2},
        {2, 4, 2, 1},
        {1, 2, 4, 2},
        {2, 1, 2, 4}
    };

    for (std::size_t ei = 0; ei < m_N; ++ei)
        for (std::size_t ej = 0; ej < m_N; ++ej)
        {
            std::size_t loc[4] = {
                node_id(ei,   ej  ),
                node_id(ei,   ej+1),
                node_id(ei+1, ej+1),
                node_id(ei+1, ej  )
            };
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    add_entry(S_coo, loc[a], loc[b], s_fac * S_local[a][b]);
                    add_entry(O_coo, loc[a], loc[b], o_fac * O_local[a][b]);
                }
        }

    auto coo_to_csr = [&](const std::vector<std::vector<std::pair<std::size_t,double>>>& coo,
                          std::vector<double>& val,
                          std::vector<int>&    col,
                          std::vector<int>&    row_ptr)
    {
        row_ptr.resize(NNodes + 1, 0);
        for (std::size_t r = 0; r < NNodes; ++r) {
            std::vector<std::pair<std::size_t,double>> entries = coo[r];
            std::sort(entries.begin(), entries.end(),
                      [](auto& a, auto& b){ return a.first < b.first; });

            std::size_t k = 0;
            while (k < entries.size()) {
                std::size_t c = entries[k].first;
                double v = 0.0;
                while (k < entries.size() && entries[k].first == c) {
                    v += entries[k].second;
                    ++k;
                }
                if (v != 0.0) {
                    col.push_back(static_cast<int>(c));
                    val.push_back(v);
                }
            }

            row_ptr[r + 1] = static_cast<int>(col.size());
        }
    };

    coo_to_csr(S_coo, m_S_values, m_S_colIndices, m_S_rowPointers);
    coo_to_csr(O_coo, m_O_values, m_O_colIndices, m_O_rowPointers);

    spdlog::info("Matrices built: nnz(S)={}, nnz(O)={}",
                 m_S_values.size(), m_O_values.size());
}

void Solver::upload_to_device()
{
    spdlog::info("Uploading matrices/vectors to GPU");
    delete m_cuda;
    m_cuda = new CudaResources();
    auto& r = *m_cuda;
    r.n_nodes = static_cast<int>(m_n_nodes);
    int n = r.n_nodes;

    cublasCreate(&r.cublas_handle);
    cusparseCreate(&r.cusparse_handle);
    int nnz_S = static_cast<int>(m_S_values.size());
    CUDA_CHECK(cudaMalloc(&r.d_S_val, nnz_S * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_S_col, nnz_S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&r.d_S_row, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(r.d_S_val, m_S_values.data(),     nnz_S * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r.d_S_col, m_S_colIndices.data(), nnz_S * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r.d_S_row, m_S_rowPointers.data(),(n+1) * sizeof(int),    cudaMemcpyHostToDevice));

    int nnz_O = static_cast<int>(m_O_values.size());
    CUDA_CHECK(cudaMalloc(&r.d_O_val, nnz_O * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_O_col, nnz_O * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&r.d_O_row, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(r.d_O_val, m_O_values.data(),     nnz_O * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r.d_O_col, m_O_colIndices.data(), nnz_O * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r.d_O_row, m_O_rowPointers.data(),(n+1) * sizeof(int),    cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&r.d_cPrev, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_cCur,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_cNext, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_rhs,   n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r.d_tmp,   n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(r.d_cPrev, m_cPrevious.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(r.d_cCur,  m_cCurrent.data(),  n * sizeof(double), cudaMemcpyHostToDevice));

    cusparseCreateCsr(&r.spmat_S,
                      n, n, nnz_S,
                      r.d_S_row, r.d_S_col, r.d_S_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateCsr(&r.spmat_O,
                      n, n, nnz_O,
                      r.d_O_row, r.d_O_col, r.d_O_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateDnVec(&r.dvec_cPrev, n, r.d_cPrev, CUDA_R_64F);
    cusparseCreateDnVec(&r.dvec_cCur,  n, r.d_cCur,  CUDA_R_64F);
    cusparseCreateDnVec(&r.dvec_rhs,   n, r.d_rhs,   CUDA_R_64F);
    cusparseCreateDnVec(&r.dvec_tmp,   n, r.d_tmp,   CUDA_R_64F);
    cusparseCreateDnVec(&r.dvec_cNext, n, r.d_cNext, CUDA_R_64F);

    double dummy1 = 1.0, dummy0 = 0.0;

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        r.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &dummy1, r.spmat_S, r.dvec_cCur, &dummy0, r.dvec_rhs,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &r.spmv_buf_S_sz));
    CUDA_CHECK(cudaMalloc(&r.spmv_buf_S, r.spmv_buf_S_sz));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        r.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &dummy1, r.spmat_O, r.dvec_cCur, &dummy0, r.dvec_rhs,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &r.spmv_buf_O1_sz));
    CUDA_CHECK(cudaMalloc(&r.spmv_buf_O1, r.spmv_buf_O1_sz));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        r.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &dummy1, r.spmat_O, r.dvec_cPrev, &dummy0, r.dvec_tmp,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &r.spmv_buf_O2_sz));
    CUDA_CHECK(cudaMalloc(&r.spmv_buf_O2, r.spmv_buf_O2_sz));
}

void Solver::applyBoundaryConditions(std::vector<double>& c, double t)
{
    for (std::size_t idx : m_boundaryNodes)
        c[idx] = 0.0;
    c[m_driving_node] = std::sin(m_F_omega * t);
}

void Solver::step()
{
    auto& r = *m_cuda;
    int n = r.n_nodes;

    double alpha, beta;

    double s_scale = -(m_v * m_v) * (m_dt * m_dt);
    beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMV(
        r.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &s_scale, r.spmat_S, r.dvec_cCur, &beta, r.dvec_rhs,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, r.spmv_buf_S));

    beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMV(
        r.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &(alpha = 1.0), r.spmat_O, r.dvec_cCur, &beta, r.dvec_tmp,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, r.spmv_buf_O1));

    alpha = 2.0;
    cublasDaxpy(r.cublas_handle, n, &alpha, r.d_tmp, 1, r.d_rhs, 1);

    beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMV(
        r.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &(alpha = 1.0), r.spmat_O, r.dvec_cPrev, &beta, r.dvec_tmp,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, r.spmv_buf_O2));

    alpha = -1.0;
    cublasDaxpy(r.cublas_handle, n, &alpha, r.d_tmp, 1, r.d_rhs, 1);

    {
        if (!m_O_factorized) {
            throw std::runtime_error("O factorization not available");
        }

        std::vector<double> h_rhs(n);
        CUDA_CHECK(cudaMemcpy(h_rhs.data(), r.d_rhs, n * sizeof(double), cudaMemcpyDeviceToHost));

        Eigen::Map<const Eigen::VectorXd> rhs(h_rhs.data(), n);
        Eigen::Map<Eigen::VectorXd> x(m_cNext.data(), n);
        x = m_O_llt.solve(rhs);
        if (m_O_llt.info() != Eigen::Success) {
            throw std::runtime_error("Eigen SimplicialLLT solve failed for O");
        }

        double t_next = m_current_t + m_dt;
        applyBoundaryConditions(m_cNext, t_next);

        CUDA_CHECK(cudaMemcpy(r.d_cNext, m_cNext.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    }

    std::swap(r.d_cPrev, r.d_cCur);
    std::swap(r.d_cCur,  r.d_cNext);

    cusparseDestroyDnVec(r.dvec_cPrev);
    cusparseDestroyDnVec(r.dvec_cCur);
    cusparseCreateDnVec(&r.dvec_cPrev, n, r.d_cPrev, CUDA_R_64F);
    cusparseCreateDnVec(&r.dvec_cCur,  n, r.d_cCur,  CUDA_R_64F);

    m_cPrevious = m_cCurrent;
    m_cCurrent  = m_cNext;

    m_current_t += m_dt;
}

void Solver::advance(int n_steps)
{
    for (int i = 0; i < n_steps; ++i)
        step();
}

std::vector<double> Solver::get_field() const
{
    return m_cCurrent;
}

double Solver::field_norm() const
{
    double s = 0.0;
    for (double v : m_cCurrent) s += v * v;
    return std::sqrt(s);
}

double Solver::field_max() const
{
    double mx = 0.0;
    for (double v : m_cCurrent) mx = std::max(mx, std::abs(v));
    return mx;
}
