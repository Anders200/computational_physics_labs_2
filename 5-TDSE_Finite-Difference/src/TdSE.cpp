#include "TdSE.hpp"   
#include <iostream>

tdse::tdse(size_t N, double dt, double omega_x, double omega_y, double m)
    : N(N), dt(dt), omega_x(omega_x), omega_y(omega_y), m(m)
{
    if (N < 3)
        throw std::invalid_argument("N must be at least 3");

    dx = 4.0 / static_cast<double>(N);
    dy = dx;
    spdlog::info("Grid: {}x{}, dx={:.4f}, dt={:.4f}", N, N, dx, dt);
    V.resize(N * N);
    psi.resize(N * N, std::complex<double>(0.0, 0.0));
    psi_new.resize(N * N, std::complex<double>(0.0, 0.0));
    init_potential();
}

void tdse::init_potential()
{
    spdlog::info("Initializing potential...");
    for (size_t i = 0; i < N; ++i)
    {
        double x = (static_cast<double>(i) - N / 2.0) * dx;
        for (size_t j = 0; j < N; ++j)
        {
            double y = (static_cast<double>(j) - N / 2.0) * dy;
            V[idx(i, j)] = 0.5 * m * (omega_x * omega_x * x * x
                                     + omega_y * omega_y * y * y);
        }
    }
}

void tdse::set_initial_state(const std::vector<double>& psi0, 
                             const std::vector<double>& psi1, 
                             const std::vector<double>& psi2)
{
    if (psi0.size() != N*N || psi1.size() != N*N || psi2.size() != N*N) {
        spdlog::error("Initial state vectors must have size N*N");
        exit(1);
    }

    for (size_t k = 0; k < N*N; ++k)
    {
        psi[k] = (psi0[k] + psi1[k] - std::complex<double>(0.0, 1.0) * psi2[k]) / std::sqrt(3.0);
    }

    // for (size_t k = 0; k < N*N; ++k)
    // {
    //     psi[k] = (psi0[k] + std::complex<double>(1.0, 0.0) * psi1[k] + psi2[k]) / std::sqrt(3.0);
    // }
}

void tdse::resize_history_vectors(size_t total_steps)
{
    norm_history.reserve(total_steps);
    energy_history.reserve(total_steps);
    x_avg_history.reserve(total_steps);
    y_avg_history.reserve(total_steps);
}

void tdse::evolve(size_t total_steps)
{
    resize_history_vectors(total_steps);

    for (size_t step = 0; step < total_steps; ++step)
    {
        step_crank_nicolson();
        update_observables();
    }
}

void tdse::update_observables()
{
    double norm = 0.0;
    double energy = 0.0;
    double x_avg = 0.0;
    double y_avg = 0.0;
    
    for (size_t k = 0; k < N*N; ++k)
    {
        double prob = std::norm(psi[k]); 
        
        norm += prob;
        energy += V[k] * prob;
        
        x_avg += (static_cast<double>(k / N) - N / 2.0) * dx * prob;
        y_avg += (static_cast<double>(k % N) - N / 2.0) * dy * prob;
    }
    
    norm_history.push_back(norm);
    energy_history.push_back(energy);
    x_avg_history.push_back(x_avg);
    y_avg_history.push_back(y_avg);
}

void tdse::step_crank_nicolson() {
    std::vector<std::complex<double>> H_psi_t(N*N, 0.0);
    apply_hamiltonian(psi, H_psi_t); 

    std::vector<std::complex<double>> psi_next = psi; 
    std::vector<std::complex<double>> H_psi_next(N*N, 0.0);
    
    std::complex<double> factor(0.0, -dt / (2.0 * H_BAR));

    for (size_t iter = 0; iter < 5; ++iter) { 
        std::fill(H_psi_next.begin(), H_psi_next.end(), std::complex<double>(0.0, 0.0));
        
        apply_hamiltonian(psi_next, H_psi_next);
        for (size_t k = 0; k < N*N; ++k) {
            psi_next[k] = psi[k] + factor * (H_psi_t[k] + H_psi_next[k]);
        }
    }
    psi = psi_next; 
}

void tdse::apply_hamiltonian(const std::vector<std::complex<double>>& in, 
                             std::vector<std::complex<double>>& out)
{
    for (size_t i = 1; i < N - 1; ++i)
    {
        for (size_t j = 1; j < N - 1; ++j)
        {
            std::complex<double> laplacian =
                (in[idx(i+1,j)] + in[idx(i-1,j)] - 2.0*in[idx(i,j)]) / (dx*dx)
              + (in[idx(i,j+1)] + in[idx(i,j-1)] - 2.0*in[idx(i,j)]) / (dy*dy);

            out[idx(i,j)] = -(H_BAR*H_BAR / (2.0*m)) * laplacian
                             + V[idx(i,j)] * in[idx(i,j)];
        }
    }
}
