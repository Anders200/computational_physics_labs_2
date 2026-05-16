#pragma once
#include <cmath>
#include <vector>
#include <functional>
#include <spdlog/spdlog.h>

static constexpr double H_BAR = 1.0;

class Solver
{
public:
    Solver(size_t N,
           double alpha_factor,
           double omega_x,
           double omega_y,
           size_t max_iter = 5000,
           double tol      = 1e-9,
           double m = 1.0,
           int seed = 42
           );

    std::vector<double> solve(size_t state_idx = 0);

    const std::vector<double>& get_potential()    const { return V;   }
    const std::vector<double>& get_wavefunction() const { return psi; }
    const std::vector<double>& get_state_energies() const { return state_energies; }
    const std::vector<std::vector<double>>& get_states() const { return found_states; }

    size_t grid_size() const { return N; }
    double grid_dx()   const { return dx; }

private:
    size_t N;
    double alpha_factor;
    double omega_x, omega_y;
    double m     = 1.0;
    double dx, dy;           
    double alpha;            
    size_t max_iter;
    double tol;
    int seed;

    inline size_t idx(size_t i, size_t j) const { return i * N + j; }

    std::vector<double> V;
    std::vector<double> psi;
    std::vector<double> Hpsi;

    std::vector<std::vector<double>> found_states;
    std::vector<double>              state_energies;


    void   init_potential();
    void   init_wavefunction();        
    void   apply_hamiltonian();
    void   normalize();
    double compute_energy() const;
    void   orthogonalize();
    void   iterate_one_step();
};
