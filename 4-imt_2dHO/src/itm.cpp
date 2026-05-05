#include "itm.hpp"
#include <cstdlib>
#include <cmath>
#include <stdexcept>




Solver::Solver(size_t N,
               double alpha_factor,
               double omega_x,
               double omega_y,
               size_t max_iter,
               double tol,
               double m,
               int seed
               )
    : N(N), alpha_factor(alpha_factor),
      omega_x(omega_x), omega_y(omega_y),
      max_iter(max_iter), tol(tol), m(m), seed(seed)
{
    if (N < 3)
        throw std::invalid_argument("N must be at least 3");

    // Grid spacing: N * dx = 4  (as specified)
    dx = 4.0 / static_cast<double>(N);
    dy = dx;

    double alpha_c = (m * dx * dx) / (2.0 * H_BAR * H_BAR);
    alpha = alpha_factor * alpha_c;

    spdlog::info("Grid: {}x{}, dx={:.4f}, alpha_c={:.6f}, alpha={:.6f}",
                 N, N, dx, alpha_c, alpha);

    V.resize(N * N);
    psi.resize(N * N, 0.0);
    Hpsi.resize(N * N, 0.0);

    init_potential();
}

void Solver::init_potential()
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
void Solver::init_wavefunction()
{
    spdlog::info("Initializing wavefunction...");
    std::srand(seed);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            bool boundary = (i == 0 || i == N-1 || j == 0 || j == N-1);
            psi[idx(i, j)] = boundary ? 0.0
                                      : 2.0 * (static_cast<double>(std::rand()) / RAND_MAX) - 1.0;
        }
}

void Solver::apply_hamiltonian()
{
    for (size_t i = 1; i < N - 1; ++i)
        for (size_t j = 1; j < N - 1; ++j)
        {
            double laplacian =
                (psi[idx(i+1,j)] + psi[idx(i-1,j)] - 2.0*psi[idx(i,j)]) / (dx*dx)
              + (psi[idx(i,j+1)] + psi[idx(i,j-1)] - 2.0*psi[idx(i,j)]) / (dy*dy);

            Hpsi[idx(i,j)] = -(H_BAR*H_BAR / (2.0*m)) * laplacian
                             + V[idx(i,j)] * psi[idx(i,j)];
        }
}

// <E> = Σ_{i,j} ψ(i,j) * Hψ(i,j) * dx * dy
double Solver::compute_energy() const
{
    double E = 0.0;
    for (size_t k = 0; k < N*N; ++k)
        E += psi[k] * Hpsi[k];
    return E * dx * dy;
}

// Normalize:  ψ /= sqrt( Σ |ψ|² dx dy )
void Solver::normalize()
{
    double norm = 0.0;
    for (size_t k = 0; k < N*N; ++k)
        norm += psi[k] * psi[k];
    norm = std::sqrt(norm * dx * dy);
    if (norm < 1e-300)
        throw std::runtime_error("Wavefunction collapsed to zero during normalization");
    for (size_t k = 0; k < N*N; ++k)
        psi[k] /= norm;
}

void Solver::orthogonalize()
{
    for (const auto& prev : found_states)
    {
        double ck = 0.0;
        for (size_t k = 0; k < N*N; ++k)
            ck += prev[k] * psi[k];
        ck *= dx * dy;

        for (size_t k = 0; k < N*N; ++k)
            psi[k] -= ck * prev[k];
    }
}

void Solver::iterate_one_step()
{
    apply_hamiltonian();
    for (size_t k = 0; k < N*N; ++k)
        psi[k] -= alpha * Hpsi[k];
    
    // boundary 
    for (size_t i = 0; i < N; ++i)
    {
        psi[idx(0,   i)] = 0.0;
        psi[idx(N-1, i)] = 0.0;
        psi[idx(i,   0)] = 0.0;
        psi[idx(i, N-1)] = 0.0;
    }
}

// Main driver
std::vector<double> Solver::solve(size_t state_idx)
{
    std::vector<double> energy_history;

    for (size_t state = 0; state <= state_idx; ++state)
    {
        spdlog::info("Solving for state {}...", state);
        energy_history.clear();

        init_wavefunction();
        if (state > 0)
            orthogonalize();   // project out already-found states
        normalize();

        double E_prev = 1e30;
        for (size_t iter = 0; iter < max_iter; ++iter)
        {
            iterate_one_step();
            if (state > 0)
                orthogonalize();
            normalize();

            apply_hamiltonian();          
            double E = compute_energy();
            energy_history.push_back(E);

            double dE = std::abs(E - E_prev);
            if (iter % 200 == 0)
                spdlog::info("  state {} iter {:5d}  E = {:.8f}  |ΔE| = {:.2e}",
                             state, iter, E, dE);

            if (dE < tol && iter > 10)
            {
                spdlog::info("  state {} converged at iter {}  E = {:.8f}", state, iter, E);
                break;
            }
            E_prev = E;
        }

        // Store converged state
        found_states.push_back(psi);
        state_energies.push_back(energy_history.back());
    }

    return energy_history;   
}
