#include "FEM.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

//  Gauss quadrature data  (2-point rule on [-1,1])
const double FEMSolver::gp_[2] = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
const double FEMSolver::gw_[2] = {1.0, 1.0};

//  Nodes ordered as: 0=(−1,−1), 1=(+1,−1), 2=(+1,+1), 3=(−1,+1)
void FEMSolver::shapeFunctions_(double xi, double eta,
                                 double N[4],
                                 double dNdxi[4], double dNdeta[4])
{
    N[0] = 0.25 * (1 - xi) * (1 - eta);
    N[1] = 0.25 * (1 + xi) * (1 - eta);
    N[2] = 0.25 * (1 + xi) * (1 + eta);
    N[3] = 0.25 * (1 - xi) * (1 + eta);

    dNdxi[0] = -0.25 * (1 - eta);
    dNdxi[1] =  0.25 * (1 - eta);
    dNdxi[2] =  0.25 * (1 + eta);
    dNdxi[3] = -0.25 * (1 + eta);

    dNdeta[0] = -0.25 * (1 - xi);
    dNdeta[1] = -0.25 * (1 + xi);
    dNdeta[2] =  0.25 * (1 + xi);
    dNdeta[3] =  0.25 * (1 - xi);
}

FEMSolver::FEMSolver(int N, double L)
    : N_(N), L_(L), n_nodes_(N * N)
{
    if (N < 2)
        throw std::invalid_argument("N must be >= 2");

    double h = L_ / static_cast<double>(N_);
    x_.resize(n_nodes_);
    y_.resize(n_nodes_);
    for (int j = 0; j < N_; ++j)
        for (int i = 0; i < N_; ++i) {
            int g = globalIdx(i, j);
            x_[g] = i * h;
            y_[g] = j * h;
        }

    u_.resize(n_nodes_);
    u_.setZero();

    spdlog::info("FEMSolver created: N={}, L={}, DOF={}", N_, L_, n_nodes_);
}

int FEMSolver::globalIdx(int i, int j) const
{
    i = ((i % N_) + N_) % N_;
    j = ((j % N_) + N_) % N_;
    return j * N_ + i;
}

void FEMSolver::assemble(double vx, double vy, double D)
{
    vx_ = vx;  vy_ = vy;  D_ = D;
    solver_ready_ = false;
    dt_cached_    = -1.0;

    buildOverlapStiffness_();
    buildAdvection_(vx, vy);

    spdlog::info("Assembly complete  (vx={}, vy={}, D={})", vx, vy, D);
}

void FEMSolver::buildOverlapStiffness_()
{
    spdlog::info("Building overlap and stiffness matrices ...");

    double h   = L_ / static_cast<double>(N_);
    double h2  = h * h;

    std::vector<Eigen::Triplet<double>> tO, tS;
    tO.reserve(16 * N_ * N_);
    tS.reserve(16 * N_ * N_);

    for (int ej = 0; ej < N_; ++ej) {
        for (int ei = 0; ei < N_; ++ei) {
            int total    = N_ * N_;
            int elem_idx = ej * N_ + ei;
            if (total >= 10 && elem_idx % (total / 10) == 0)
                spdlog::info("  Overlap/Stiffness assembly: {:3d}%",
                             100 * elem_idx / total);

            int dofs[4] = {
                globalIdx(ei,   ej  ),
                globalIdx(ei+1, ej  ),
                globalIdx(ei+1, ej+1),
                globalIdx(ei,   ej+1)
            };

            double Ke[4][4] = {};   // stiffness
            double Me[4][4] = {};   // overlap / mass

            // 2×2 Gauss quadrature
            for (int qi = 0; qi < 2; ++qi) {
                for (int qj = 0; qj < 2; ++qj) {
                    double xi  = gp_[qi];
                    double eta = gp_[qj];
                    double w   = gw_[qi] * gw_[qj];

                    double N4[4], dNdxi[4], dNdeta[4];
                    shapeFunctions_(xi, eta, N4, dNdxi, dNdeta);

                    // Jacobian for uniform rectangular element: J = h/2 * I
                    // det(J) = (h/2)^2,  J^{-1} = 2/h * I
                    double detJ = 0.25 * h2;
                    double inv2h = 2.0 / h;    // = 1/Jxx scaled

                    for (int a = 0; a < 4; ++a) {
                        for (int b = 0; b < 4; ++b) {
                            // Mass
                            Me[a][b] += w * N4[a] * N4[b] * detJ;

                            // stiffness
                            double dNadx = dNdxi[a]  * inv2h;
                            double dNady = dNdeta[a] * inv2h;
                            double dNbdx = dNdxi[b]  * inv2h;
                            double dNbdy = dNdeta[b] * inv2h;
                            Ke[a][b] += w * (dNadx * dNbdx + dNady * dNbdy) * detJ;
                        }
                    }
                }
            }

            // Scatter into global triplets
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    tO.emplace_back(dofs[a], dofs[b], Me[a][b]);
                    tS.emplace_back(dofs[a], dofs[b], Ke[a][b]);
                }
        }
    }
    spdlog::info("  Overlap/Stiffness assembly: 100%");

    O_.resize(n_nodes_, n_nodes_);
    S_.resize(n_nodes_, n_nodes_);
    O_.setFromTriplets(tO.begin(), tO.end());
    S_.setFromTriplets(tS.begin(), tS.end());

    spdlog::info("  O nnz={}, S nnz={}", O_.nonZeros(), S_.nonZeros());
}

void FEMSolver::buildAdvection_(double vx, double vy)
{
    spdlog::info("Building advection matrix (vx={}, vy={}) ...", vx, vy);

    double h  = L_ / static_cast<double>(N_);
    double h2 = h * h;

    std::vector<Eigen::Triplet<double>> tC;
    tC.reserve(16 * N_ * N_);


    for (int ej = 0; ej < N_; ++ej) {
        for (int ei = 0; ei < N_; ++ei) {
            int total    = N_ * N_;
            int elem_idx = ej * N_ + ei;
            if (total >= 10 && elem_idx % (total / 10) == 0)
                spdlog::info("  Advection assembly: {:3d}%",
                             100 * elem_idx / total);

            int dofs[4] = {
                globalIdx(ei,   ej  ),
                globalIdx(ei+1, ej  ),
                globalIdx(ei+1, ej+1),
                globalIdx(ei,   ej+1)
            };

            double Ce[4][4] = {};

            for (int qi = 0; qi < 2; ++qi) {
                for (int qj = 0; qj < 2; ++qj) {
                    double xi  = gp_[qi];
                    double eta = gp_[qj];
                    double w   = gw_[qi] * gw_[qj];

                    double N4[4], dNdxi[4], dNdeta[4];
                    shapeFunctions_(xi, eta, N4, dNdxi, dNdeta);

                    double detJ  = 0.25 * h2;
                    double inv2h = 2.0 / h;

                    for (int a = 0; a < 4; ++a) {
                        for (int b = 0; b < 4; ++b) {
                            double dNbdx = dNdxi[b]  * inv2h;
                            double dNbdy = dNdeta[b] * inv2h;
                            Ce[a][b] += w * N4[a]
                                          * (vx * dNbdx + vy * dNbdy)
                                          * detJ;
                        }
                    }
                }
            }

            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    tC.emplace_back(dofs[a], dofs[b], Ce[a][b]);
        }
    }
    spdlog::info("  Advection assembly: 100%");

    C_.resize(n_nodes_, n_nodes_);
    C_.setFromTriplets(tC.begin(), tC.end());
    spdlog::info("  C nnz={}", C_.nonZeros());
}

void FEMSolver::buildCN_(double dt)
{
    spdlog::info("Building Crank-Nicolson system (dt={}) ...", dt);

    
    // Weak form: O du/dt = -C u - D S u
    SpMat CS = C_ + D_ * S_;

    A_ = O_ + (dt / 2.0) * CS;
    B_ = O_ - (dt / 2.0) * CS;

    A_.makeCompressed();

    spdlog::info("Factorising LHS matrix ...");
    solver_.compute(A_);
    if (solver_.info() != Eigen::Success)
        throw std::runtime_error("SparseLU factorisation failed");

    solver_ready_ = true;
    dt_cached_    = dt;
    spdlog::info("Crank-Nicolson system ready");
}

void FEMSolver::setInitialCondition(double cx, double cy, double k)
{
    for (int g = 0; g < n_nodes_; ++g)
        u_[g] = std::exp(-k * ((x_[g] - cx) * (x_[g] - cx)
                             + (y_[g] - cy) * (y_[g] - cy)));
    t_ = 0.0;
    spdlog::info("Initial condition set: Gaussian at ({},{}) k={}", cx, cy, k);
}

void FEMSolver::step(double dt)
{
    if (!solver_ready_ || dt != dt_cached_)
        buildCN_(dt);

    VecXd rhs = B_ * u_;
    u_        = solver_.solve(rhs);

    if (solver_.info() != Eigen::Success)
        throw std::runtime_error("Linear solve failed");

    t_ += dt;
}

double FEMSolver::minU() const { return u_.minCoeff(); }
double FEMSolver::maxU() const { return u_.maxCoeff(); }

double FEMSolver::centerX() const
{
    const double twopi_over_L = 2.0 * M_PI / L_;
    double sc = 0.0, ss = 0.0, den = 0.0;
    for (int g = 0; g < n_nodes_; ++g) {
        double v = std::max(u_[g], 0.0);
        double theta = x_[g] * twopi_over_L;
        sc  += v * std::cos(theta);
        ss  += v * std::sin(theta);
        den += v;
    }
    if (den < 1e-30) return L_ / 2.0;
    double angle = std::atan2(ss / den, sc / den);
    if (angle < 0.0) angle += 2.0 * M_PI;
    return angle / twopi_over_L;
}

double FEMSolver::centerY() const
{
    const double twopi_over_L = 2.0 * M_PI / L_;
    double sc = 0.0, ss = 0.0, den = 0.0;
    for (int g = 0; g < n_nodes_; ++g) {
        double v = std::max(u_[g], 0.0);
        double theta = y_[g] * twopi_over_L;
        sc  += v * std::cos(theta);
        ss  += v * std::sin(theta);
        den += v;
    }
    if (den < 1e-30) return L_ / 2.0;
    double angle = std::atan2(ss / den, sc / den);
    if (angle < 0.0) angle += 2.0 * M_PI;
    return angle / twopi_over_L;
}

SimResult FEMSolver::runPureAdvection(double vx, double vy, double dt,
                                       int stepsPerSnapshot)
{
    spdlog::info("=== runPureAdvection: vx={}, vy={}, dt={} ===", vx, vy, dt);
    assemble(vx, vy, 0.0);
    setInitialCondition(L_ / 2.0, L_ / 2.0, 2.0);
    buildCN_(dt);

    double speed = std::hypot(vx, vy);
    if (speed < 1e-12)
        throw std::invalid_argument("velocity is zero in runPureAdvection");

    double T_period = L_ / speed;
    double t_end    = 2.0 * T_period;
    spdlog::info("Period T={:.4f}, running to t={:.4f}", T_period, t_end);

    SimResult res;
    res.x_nodes = x_;
    res.y_nodes = y_;

    int step_count = 0;
    while (t_ < t_end - 0.5 * dt) {
        if (step_count % stepsPerSnapshot == 0) {
            res.time    .push_back(t_);
            res.u_min   .push_back(minU());
            res.u_max   .push_back(maxU());
            res.center_x.push_back(centerX());
            res.center_y.push_back(centerY());
        }
        step(dt);
        ++step_count;
        if (step_count % 200 == 0)
            spdlog::info("  t={:.3f} / {:.3f}  min={:.4f}  max={:.4f}",
                         t_, t_end, minU(), maxU());
    }
    // final snapshot
    res.time    .push_back(t_);
    res.u_min   .push_back(minU());
    res.u_max   .push_back(maxU());
    res.center_x.push_back(centerX());
    res.center_y.push_back(centerY());
    res.u_final.assign(u_.data(), u_.data() + n_nodes_);

    spdlog::info("=== runPureAdvection done, steps={} ===", step_count);
    return res;
}

SimResult FEMSolver::runPureDiffusion(double D, double t_end, double dt,
                                       int stepsPerSnapshot)
{
    spdlog::info("=== runPureDiffusion: D={}, t_end={}, dt={} ===", D, t_end, dt);
    assemble(0.0, 0.0, D);
    setInitialCondition(L_ / 2.0, L_ / 2.0, 2.0);
    buildCN_(dt);

    SimResult res;
    res.x_nodes = x_;
    res.y_nodes = y_;

    int step_count = 0;
    while (t_ < t_end - 0.5 * dt) {
        if (step_count % stepsPerSnapshot == 0) {
            res.time    .push_back(t_);
            res.u_min   .push_back(minU());
            res.u_max   .push_back(maxU());
            res.center_x.push_back(centerX());
            res.center_y.push_back(centerY());
        }
        step(dt);
        ++step_count;
        if (step_count % 200 == 0)
            spdlog::info("  t={:.3f} / {:.3f}  min={:.6f}  max={:.6f}",
                         t_, t_end, minU(), maxU());
    }
    res.time    .push_back(t_);
    res.u_min   .push_back(minU());
    res.u_max   .push_back(maxU());
    res.center_x.push_back(centerX());
    res.center_y.push_back(centerY());
    res.u_final.assign(u_.data(), u_.data() + n_nodes_);

    spdlog::info("=== runPureDiffusion done, steps={} ===", step_count);
    return res;
}

SimResult FEMSolver::runAdvectionDiffusion(double vx, double vy, double D,
                                            double t_end, double dt,
                                            int stepsPerSnapshot)
{
    spdlog::info("=== runAdvectionDiffusion: vx={}, vy={}, D={}, t_end={}, dt={} ===",
                 vx, vy, D, t_end, dt);
    assemble(vx, vy, D);
    setInitialCondition(L_ / 2.0, L_ / 2.0, 2.0);
    buildCN_(dt);

    SimResult res;
    res.x_nodes = x_;
    res.y_nodes = y_;

    int step_count = 0;
    while (t_ < t_end - 0.5 * dt) {
        if (step_count % stepsPerSnapshot == 0) {
            res.time    .push_back(t_);
            res.u_min   .push_back(minU());
            res.u_max   .push_back(maxU());
            res.center_x.push_back(centerX());
            res.center_y.push_back(centerY());
        }
        step(dt);
        ++step_count;
        if (step_count % 200 == 0)
            spdlog::info("  t={:.3f} / {:.3f}  min={:.4f}  max={:.4f}",
                         t_, t_end, minU(), maxU());
    }
    res.time    .push_back(t_);
    res.u_min   .push_back(minU());
    res.u_max   .push_back(maxU());
    res.center_x.push_back(centerX());
    res.center_y.push_back(centerY());
    res.u_final.assign(u_.data(), u_.data() + n_nodes_);

    spdlog::info("=== runAdvectionDiffusion done, steps={} ===", step_count);
    return res;
}
