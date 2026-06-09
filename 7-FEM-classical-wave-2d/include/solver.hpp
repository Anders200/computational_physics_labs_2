#pragma once

#include <vector>
#include <cstddef>
#include <string>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

class Solver
{
public:
    Solver(double L, double v, double F_omega, double dt, std::size_t N);
    ~Solver();

    void initialize_mesh();

    void step();

    void advance(int n_steps);

    double get_time() const { return m_current_t; }
    std::size_t get_nodes_per_side() const { return m_N + 1; }
    std::size_t get_total_nodes() const { return (m_N + 1) * (m_N + 1); }
    
    std::vector<double> get_field() const;
    const std::vector<double>& get_x_coords() const { return m_xCoords; }
    const std::vector<double>& get_y_coords() const { return m_yCoords; }

    std::size_t get_driving_node() const { return m_driving_node; }

    double field_norm() const;
    double field_max() const;

    static double cfl_dt(double L, double v, std::size_t N);
 

private:
    double      m_L;
    double      m_v;
    double      m_F_omega;
    double      m_dt;
    double      m_current_t;
    std::size_t m_N;
    std::size_t m_n_nodes;

    std::size_t              m_driving_node;
    std::vector<std::size_t> m_boundaryNodes;
    std::vector<bool>        m_is_constrained;
    std::vector<double>      m_xCoords;
    std::vector<double>      m_yCoords;

    std::vector<double> m_cPrevious;
    std::vector<double> m_cCurrent;
    std::vector<double> m_cNext;

    std::vector<double>      m_S_values;
    std::vector<int>         m_S_colIndices;
    std::vector<int>         m_S_rowPointers;

    std::vector<double>      m_O_values;
    std::vector<int>         m_O_colIndices;
    std::vector<int>         m_O_rowPointers;

    Eigen::SparseMatrix<double> m_O_eigen;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> m_O_llt;
    bool m_O_factorized = false;

    struct CudaResources;
    CudaResources* m_cuda = nullptr;

    void build_matrices();
    void upload_to_device();
    void applyBoundaryConditions(std::vector<double>& c, double t);

    inline std::size_t node_id(std::size_t i, std::size_t j) const
    { return i * (m_N + 1) + j; }
};
