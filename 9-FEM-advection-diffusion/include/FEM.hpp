#pragma once

#include <vector>
#include <string>
#include <functional>
#include <Eigen/Sparse>
#include <Eigen/Dense>


using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using VecXd = Eigen::VectorXd;

struct SimResult {
    std::vector<double> time;
    std::vector<double> u_min;
    std::vector<double> u_max;
    std::vector<double> center_x;
    std::vector<double> center_y;
    // snapshot of the full field at the last stored time
    std::vector<double> x_nodes;
    std::vector<double> y_nodes;
    std::vector<double> u_final;
};

class FEMSolver {
public:
    FEMSolver(int N, double L);

    void assemble(double vx, double vy, double D);

    void setInitialCondition(double cx, double cy, double k = 2.0);

    void step(double dt);


    double minU()      const;
    double maxU()      const;
    double centerX()   const;   ///< mass-weighted centroid x
    double centerY()   const;   ///< mass-weighted centroid y
    double time()      const { return t_; }

    /// Raw node coordinates
    const std::vector<double>& xNodes() const { return x_; }
    const std::vector<double>& yNodes() const { return y_; }

    /// Current coefficient vector 
    const VecXd& u() const { return u_; }

    SimResult runPureAdvection(double vx, double vy, double dt,
                               int    stepsPerSnapshot = 10);

    SimResult runPureDiffusion(double D, double t_end, double dt,
                               int    stepsPerSnapshot = 10);

    SimResult runAdvectionDiffusion(double vx, double vy, double D,
                                    double t_end,  double dt,
                                    int    stepsPerSnapshot = 10);

private:
    int    N_;          
    double L_;         
    int    n_nodes_;  

    std::vector<double> x_, y_;   

    SpMat O_;   // overlap 
    SpMat S_;   // stiffness matrix 
    SpMat C_;   // advection matrix
    SpMat A_;   // LHS of CN step  
    SpMat B_;   // RHS factor     
    double dt_cached_ = -1.0;

    // state
    VecXd u_;
    double t_ = 0.0;

    // parameters cached during assemble
    double vx_ = 0, vy_ = 0, D_ = 0;

    Eigen::SparseLU<SpMat> solver_;
    bool solver_ready_ = false;

    int globalIdx(int i, int j) const;

    void buildOverlapStiffness_();
    void buildAdvection_(double vx, double vy);
    void buildCN_(double dt);

    /// 2×2 Gauss quadrature weights and points on [-1,1]
    static const double gp_[2];
    static const double gw_[2];

    static void shapeFunctions_(double xi, double eta,
                                 double N[4],
                                 double dNdxi[4], double dNdeta[4]);
};
