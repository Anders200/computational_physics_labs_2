#pragma once
#include <complex>
#include <vector>
#include <spdlog/spdlog.h>

static constexpr double H_BAR = 1.0;

class tdse {
public:
    tdse(size_t N, 
               double dt, 
               double omega_x = 1.0, 
               double omega_y = 1.001, 
               double m = 1.0);

    void set_initial_state(const std::vector<double>& psi0, 
                            const std::vector<double>& psi1, 
                            const std::vector<double>& psi2);

    void evolve(size_t total_steps);

    const std::vector<double>& get_norm_history()    const { return norm_history; }
    const std::vector<double>& get_energy_history()  const { return energy_history; }
    const std::vector<double>& get_x_avg_history()   const { return x_avg_history; }
    const std::vector<double>& get_y_avg_history()   const { return y_avg_history; }
    std::vector<std::complex<double>> get_psi() const { return psi; }

private:
    size_t N;
    double dx, dy;
    double dt;
    double m, omega_x, omega_y;

    std::vector<double> V;                         
    std::vector<std::complex<double>> psi;
    std::vector<std::complex<double>> psi_new;
    
    void resize_history_vectors(size_t total_steps);

    std::vector<double> norm_history;
    std::vector<double> energy_history;
    std::vector<double> x_avg_history;
    std::vector<double> y_avg_history;

    inline size_t idx(size_t i, size_t j) const { return i * N + j; }

    void init_potential();
    
    void apply_hamiltonian(const std::vector<std::complex<double>>& in, 
                           std::vector<std::complex<double>>& out);

    void step_crank_nicolson();

    void update_observables();
};
