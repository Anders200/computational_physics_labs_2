#include "FEM.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>


//  Constants
static constexpr double PI = M_PI;

//  Bilinear 1-D basis functions  (reference coords ξ ∈ [-1,1])
//      f1(ξ) = (1-ξ)/2     f2(ξ) = (1+ξ)/2

static inline double f1(double xi) { return (1.0 - xi) * 0.5; }
static inline double f2(double xi) { return (1.0 + xi) * 0.5; }

//  Biquadratic 1-D basis functions
//      q1(ξ) = ξ(ξ-1)/2    q2(ξ) = 1-ξ²    q3(ξ) = ξ(ξ+1)/2
static inline double q1(double xi) { return xi * (xi - 1.0) * 0.5; }
static inline double q2(double xi) { return (1.0 - xi) * (1.0 + xi); }
static inline double q3(double xi) { return xi * (xi + 1.0) * 0.5; }

//  Constructor
FEM::FEM(double L, size_t N, double rc_x, double rc_y, BasisType basis)
    : L(L), N(N), rc_x(rc_x), rc_y(rc_y), basis_(basis)
{
    if (basis_ == BasisType::Bilinear) {
        nodes_per_element = 4;
        num_nodes = (N + 1) * (N + 1);
    } else {
        nodes_per_element = 9;
        num_nodes = (2 * N + 1) * (2 * N + 1);
    }
    a = L / static_cast<double>(N);
    
    // Initialize CUBLAS
    cublasCreate(&cublas_handle);
    spdlog::debug("CUBLAS initialized for solving.");
}

//  Destructor
FEM::~FEM()
{
    // Clean up CUDA memory
    if (d_A) cudaFree(d_A);
    if (d_b) cudaFree(d_b);
    if (d_x) cudaFree(d_x);
    
    // Destroy CUBLAS handle
    cublasDestroy(cublas_handle);
    
    spdlog::debug("CUDA memory cleaned up.");
}

//  Exact (analytical) solution
//  Ψ(r) = -1/(2π) * ln|r - rc|
double FEM::exact_solution(double x, double y) const
{
    double dx = x - rc_x;
    double dy = y - rc_y;
    return -1.0 / (2.0 * PI) * std::log(std::sqrt(dx*dx + dy*dy));
}

//  Reference-space shape functions
//
//  Local node ordering: 
//
//  Bilinear (nodes 0-3):
//    0 → (-1,-1)   1 → (+1,-1)
//    2 → (-1,+1)   3 → (+1,+1)
//
//  Biquadratic (nodes 0-8, adds):
//    4 → ( 0,-1)   5 → (+1, 0)
//    6 → (-1, 0)   7 → ( 0,+1)   8 → (0,0)



double FEM::shape_func(int m, double xi1, double xi2) const
{
    if (basis_ == BasisType::Bilinear) {
        switch (m) {
            case 0: return f1(xi1) * f1(xi2);   // (-1,-1)
            case 1: return f2(xi1) * f1(xi2);   // (+1,-1)
            case 2: return f1(xi1) * f2(xi2);   // (-1,+1)
            case 3: return f2(xi1) * f2(xi2);   // (+1,+1)
        }
    } else {
        switch (m) {
            case 0: return q1(xi1) * q1(xi2);   // (-1,-1)
            case 1: return q3(xi1) * q1(xi2);   // (+1,-1)
            case 2: return q1(xi1) * q3(xi2);   // (-1,+1)
            case 3: return q3(xi1) * q3(xi2);   // (+1,+1)
            case 4: return q2(xi1) * q1(xi2);   // ( 0,-1)
            case 5: return q3(xi1) * q2(xi2);   // (+1, 0)
            case 6: return q1(xi1) * q2(xi2);   // (-1, 0)
            case 7: return q2(xi1) * q3(xi2);   // ( 0,+1)
            case 8: return q2(xi1) * q2(xi2);   // ( 0, 0)
        }
    }
    return 0.0;
}

// Central finite-difference derivatives in reference space (Δ = 0.1)
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

//  Node lookup / insertion
int FEM::find_node(double x, double y) const
{
    constexpr double TOL = 1e-10;
    for (size_t i = 0; i < nodeX_.size(); ++i)
        if (std::abs(nodeX_[i] - x) < TOL && std::abs(nodeY_[i] - y) < TOL)
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
    return x < TOL || x > L - TOL || y < TOL || y > L - TOL;
}

//  initialize()  — allocate all arrays (but NOT the node lists,
//                  which grow dynamically via add_node)
void FEM::initialize()
{
    nlg.resize(N * N * static_cast<size_t>(nodes_per_element));
    nodeX_.clear();  nodeX_.reserve(num_nodes);
    nodeY_.clear();  nodeY_.reserve(num_nodes);
    coefficients.assign(num_nodes, 0.0);
    rhs.assign(num_nodes, 0.0);
    stiffness_matrix.assign(num_nodes * num_nodes, 0.0);  // 1D row-major allocation
}

//  compute_local_stiffness()
//
//  s_ji = ∫∫ (∂h_j/∂ξ1 · ∂h_i/∂ξ1 + ∂h_j/∂ξ2 · ∂h_i/∂ξ2) dξ1 dξ2
//
//  Evaluated with 4-point Gauss–Legendre quadrature in each direction.
//
//  Weights:  w1=w2=(18+√30)/36,  w3=w4=(18-√30)/36
//  Nodes:    γ1=-√(3/7-2/7·√(6/5)),  γ2=+√(3/7-2/7·√(6/5))
//            γ3=+√(3/7+2/7·√(6/5)),  γ4=-√(3/7+2/7·√(6/5))
//
//  The Jacobian factor (a/2)² from the area element and the (2/a)² from
//  the chain rule exactly cancel for a uniform square mesh, so no 'a'
//  appears in the formula.
void FEM::compute_local_stiffness()
{
    const double sq30  = std::sqrt(30.0);
    const double sq65  = std::sqrt(6.0 / 5.0);
    const double inner = std::sqrt(3.0/7.0 - 2.0/7.0 * sq65);
    const double outer = std::sqrt(3.0/7.0 + 2.0/7.0 * sq65);

    const double w[4]   = { (18.0+sq30)/36.0, (18.0+sq30)/36.0,
                             (18.0-sq30)/36.0, (18.0-sq30)/36.0 };
    const double gam[4] = { -inner, +inner, +outer, -outer };

    localStiffness_.assign(nodes_per_element,
                           std::vector<double>(nodes_per_element, 0.0));

    for (int l = 0; l < 4; ++l) {
        for (int n = 0; n < 4; ++n) {
            double xi1 = gam[l], xi2 = gam[n];
            double ww  = w[l] * w[n];
            for (int j = 0; j < nodes_per_element; ++j) {
                double dj1 = shape_deriv_xi1(j, xi1, xi2);
                double dj2 = shape_deriv_xi2(j, xi1, xi2);
                for (int i = 0; i < nodes_per_element; ++i) {
                    double di1 = shape_deriv_xi1(i, xi1, xi2);
                    double di2 = shape_deriv_xi2(i, xi1, xi2);
                    localStiffness_[j][i] += ww * (dj1*di1 + dj2*di2);
                }
            }
        }
    }
    spdlog::debug("Local stiffness matrix precomputed ({0}×{0}).",
                  nodes_per_element);
}

//  Real-space offsets from the bottom-left anchor of element k.
//  The ordering must be consistent with shape_func() above.
//
//  offset[m] = (Δx, Δy) of local node m from the element's (0,0) corner.
void FEM::build_mesh()
{
    initialize();
    spdlog::info("Building mesh: N={}, nodes_per_element={}, total_nodes={}",
                 N, nodes_per_element, num_nodes);

    // Real-space offset from bottom-left corner for each local node.
    // Only the first 4 entries are used for Bilinear.
    const double offsets[9][2] = {
        {0.0,   0.0  },   // 0  (-1,-1)
        {a,     0.0  },   // 1  (+1,-1)
        {0.0,   a    },   // 2  (-1,+1)
        {a,     a    },   // 3  (+1,+1)
        {a/2.0, 0.0  },   // 4  ( 0,-1)  -- quadratic only
        {a,     a/2.0},   // 5  (+1, 0)  -- quadratic only
        {0.0,   a/2.0},   // 6  (-1, 0)  -- quadratic only
        {a/2.0, a    },   // 7  ( 0,+1)  -- quadratic only
        {a/2.0, a/2.0}    // 8  ( 0, 0)  -- quadratic only
    };

    // Iterate over elements (row i, column j in the grid)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            size_t k  = i * N + j;       // element index
            double ax = j * a;           // bottom-left x
            double ay = i * a;           // bottom-left y

            for (int m = 0; m < nodes_per_element; ++m) {
                double x = ax + offsets[m][0];
                double y = ay + offsets[m][1];
                int existing = find_node(x, y);
                nlg[k * nodes_per_element + m] =
                    (existing != -1) ? static_cast<size_t>(existing)
                                     : static_cast<size_t>(add_node(x, y));
            }
        }
    }

    if (nodeX_.size() != num_nodes) {
        throw std::runtime_error(
            "Node count mismatch: expected " + std::to_string(num_nodes) +
            ", got "                         + std::to_string(nodeX_.size()));
    }
    spdlog::info("Mesh built: {} nodes, {} elements.", nodeX_.size(), N*N);

    compute_local_stiffness();
}

//  assemble_stiffness_matrix()
//  S(nlg(k,i1), nlg(k,i2)) += s_local(i1, i2)   ∀ k, i1, i2
void FEM::assemble_stiffness_matrix()
{
    spdlog::info("Assembling global stiffness matrix...");
    // Reset to zero
    std::fill(stiffness_matrix.begin(), stiffness_matrix.end(), 0.0);

    for (size_t k = 0; k < N * N; ++k) {
        for (int i1 = 0; i1 < nodes_per_element; ++i1) {
            size_t g1 = nlg[k * nodes_per_element + i1];
            for (int i2 = 0; i2 < nodes_per_element; ++i2) {
                size_t g2 = nlg[k * nodes_per_element + i2];
                S(g1, g2) += localStiffness_[i1][i2];
            }
        }
    }
    spdlog::info("Global stiffness matrix assembled.");
}

//  apply_boundary_conditions()
//
//  For every boundary node i:
//    • zero row i of S, set S[i][i] = 1
//    • set rhs[i] = Ψ_exact(p_i)
//  Interior rhs entries remain 0 (no internal charges).
void FEM::apply_boundary_conditions()
{
    spdlog::info("Applying Dirichlet boundary conditions (rc=({},{})).",
                 rc_x, rc_y);
    std::fill(rhs.begin(), rhs.end(), 0.0);

    for (size_t i = 0; i < num_nodes; ++i) {
        if (is_boundary_node(i)) {
            double psi = exact_solution(nodeX_[i], nodeY_[i]);
            // Zero out row i
            for (size_t j = 0; j < num_nodes; ++j)
                S(i, j) = 0.0;
            // Set diagonal to 1
            S(i, i) = 1.0;
            rhs[i] = psi;
        }
    }
    spdlog::info("Boundary conditions applied.");
}

//  solve()
//  Use cuSOLVER LU decomposition + CUBLAS substitution for faster solving on GPU
void FEM::solve()
{
    spdlog::info("Solving linear system using cuSOLVER (n={})...", num_nodes);
    int n = static_cast<int>(num_nodes);
    
    // Stiffness matrix is already 1D row-major, make a working copy
    std::vector<double> A_flat = stiffness_matrix;
    
    // Allocate device memory
    if (d_A) cudaFree(d_A);
    if (d_b) cudaFree(d_b);
    if (d_x) cudaFree(d_x);
    
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_A, A_flat.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Create cuSOLVER handle
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);
    
    // Allocate workspace
    int lwork;
    cusolverDnDgetrf_bufferSize(cusolver_handle, n, n, d_A, n, &lwork);
    double* d_work;
    int* d_info;
    int* d_ipiv;
    cudaMalloc(&d_work, lwork * sizeof(double));
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_ipiv, n * sizeof(int));
    
    // LU decomposition: PA = LU
    cusolverDnDgetrf(cusolver_handle, n, n, d_A, n, d_work, d_ipiv, d_info);
    
    // Solve Ax = b using LU factors: A = PLU, so we solve LUx = P^T b
    cusolverDnDgetrs(cusolver_handle, CUBLAS_OP_N, n, 1, d_A, n, d_ipiv, d_x, n, d_info);
    
    // Copy solution back to host
    cudaMemcpy(coefficients.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Check for errors
    int info_h;
    cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (info_h != 0) {
        cudaFree(d_work);
        cudaFree(d_info);
        cudaFree(d_ipiv);
        cusolverDnDestroy(cusolver_handle);
        throw std::runtime_error(
            "cuSOLVER error at step " + std::to_string(info_h));
    }
    
    // Clean up
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_ipiv);
    cusolverDnDestroy(cusolver_handle);
    
    spdlog::info("System solved successfully using GPU (cuSOLVER).");
}

//  evaluate(x, y)
//
//  Locates the element containing (x,y), maps to reference
//  coordinates, and evaluates Ψ = Σ c_{nlg(k,m)} · h_m(ξ).
double FEM::evaluate(double x, double y) const
{
    // Grid indices of the containing element
    int ej = static_cast<int>(x / a);
    int ei = static_cast<int>(y / a);
    ej = std::clamp(ej, 0, static_cast<int>(N) - 1);
    ei = std::clamp(ei, 0, static_cast<int>(N) - 1);

    size_t k = static_cast<size_t>(ei) * N + static_cast<size_t>(ej);

    // Map to reference coordinates ξ ∈ [-1,1]
    double ax  = ej * a;
    double ay  = ei * a;
    double xi1 = 2.0 * (x - ax) / a - 1.0;
    double xi2 = 2.0 * (y - ay) / a - 1.0;

    double result = 0.0;
    for (int m = 0; m < nodes_per_element; ++m) {
        size_t gn = nlg[k * nodes_per_element + m];
        result += coefficients[gn] * shape_func(m, xi1, xi2);
    }
    return result;
}
