#ifndef SOLVER_HPP
#define SOLVER_HPP


#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>

class PoissonSolver2D {
public:
    PoissonSolver2D(int n_power, int log_every = 10, bool track_history = false, int history_stride = 10);
    
    void solve_gauss_seidel(int max_iters, double tol);
    void solve_sor(int max_iters, double tol, double omega);
    void solve_with_refinement(int lower_n, int max_iters, double tol, double omega);

    std::vector<double> get_energy_history() const;
    std::vector<double> get_residual_history() const;
    std::vector<double> get_refinement_timing_history() const;
    std::vector<std::vector<double>> get_solution() const;
    std::vector<std::vector<std::vector<double>>> get_solution_history_refinement() const;

    void set_track_solution_history(bool track) { this->track_solution_history = track; }
    void set_log_every(int log_every) { this->log_every = log_every; }
    void set_history_stride(int stride) { this->history_stride = stride; }
    void set_solution_from_coarse(const std::vector<std::vector<double>>& coarse_sol, int coarse_n_power);

    double compute_energy() const;
    double compute_max_residual() const;

private:
    int n_power;
    int N;
    double dx;
    int log_every;
    int history_stride;

    std::vector<double> u;
    std::vector<double> f;

    bool track_solution_history;
    std::vector<std::vector<std::vector<double>>> solution_history_refinement;
    std::vector<double> energy_history;
    std::vector<double> residual_history;
    std::vector<double> refinement_timing_history; // ms per refinement level

    inline int idx(int i, int j) const { return j * (N+1) + i; }
    
    void initialize();
    void compute_rhs();
    void sor_step(double omega);
    double compute_residual_at(int i, int j) const;


};

#endif // SOLVER_HPP
