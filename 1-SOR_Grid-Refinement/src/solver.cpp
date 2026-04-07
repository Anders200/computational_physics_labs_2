#include "solver.hpp"



PoissonSolver2D::PoissonSolver2D(int n_power, int log_every, bool track_history, int history_stride)
    : N(1 << n_power),                    
      dx(3.0 / N),
      log_every(log_every),
      n_power(n_power),
      track_solution_history(track_history)
{
    PoissonSolver2D::initialize();
}

void PoissonSolver2D::initialize() 
{
    u.assign((N+1)*(N+1), 0.0);
    f.assign((N+1)*(N+1), 0.0);
    compute_rhs();
    // diagnostic: print max |f|
    double maxf = 0.0;
    for (double v : f) maxf = std::max(maxf, std::abs(v));
    std::cout << "[init] N=" << N << ", dx=" << dx << ", max|f|=" << maxf << "\n";
}

void PoissonSolver2D::compute_rhs() 
{
    for (int i = 0; i <= N; ++i) 
    {
        for (int j = 0; j <= N; ++j) 
        {
            double x = i * dx;
            double y = j * dx;
            f[idx(i, j)] = 2 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
}

void PoissonSolver2D::solve_gauss_seidel(int max_iters, double tol) {
    solve_sor(max_iters, tol, 1.0);
}

void PoissonSolver2D::solve_sor(int max_iters, double tol, double omega) {
    energy_history.clear();
    residual_history.clear();
    std::string method = (omega == 1.0) ? "Gauss-Seidel" : "SOR w = " +  std::to_string(omega);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iters; iter++) {
        sor_step(omega);

        double res = compute_max_residual();
        double S = compute_energy();

        if (iter % log_every == 0) 
        {
            double maxu = 0.0;
            for (double val : u) maxu = std::max(maxu, std::abs(val));

            std::cout << std::fixed << std::setprecision(5);
            std::cout   << "\r"
                        << "2 ^ "
                        << n_power
                        << ": "
                        << method
                        << " Iteration " << std::setw(5) << iter
                        << ": Residual = " << res
                        << ", Energy = " << S
                        << ", max|u|=" << maxu
                        << std::flush;

            residual_history.push_back(res);
            energy_history.push_back(S);
        }

        if (res < tol) break;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << " (converged in " << duration.count() << " ms)\n";
}

void PoissonSolver2D::sor_step(double omega) {
    for (int j = 1; j < N; ++j) 
    {
        for (int i = 1; i < N; ++i) 
        {
            int ip = i + 1;
            int im = i - 1;
            int jp = j + 1;
            int jm = j - 1;

            double u_new = 0.25 * (
                u[idx(im,j)] + u[idx(ip,j)] +
                u[idx(i,jm)] + u[idx(i,jp)] +
                dx*dx * f[idx(i,j)]
            );

            u[idx(i,j)] = omega * u_new + (1.0 - omega) * u[idx(i,j)];
        }
    }
}


double PoissonSolver2D::compute_energy() const {
    double S = 0.0;

    for (int j = 1; j < N; j++) {
        for (int i = 1; i < N; i++) {

            double ux = (u[idx(i+1,j)] - u[idx(i-1,j)]) / (2*dx);
            double uy = (u[idx(i,j+1)] - u[idx(i,j-1)]) / (2*dx);

            double grad2 = ux*ux + uy*uy;

            S += 0.5 * grad2 - f[idx(i,j)] * u[idx(i,j)];
        }
    }

    return S * dx * dx;
}

double PoissonSolver2D::compute_residual_at(int i, int j) const 
{
    return (
        u[idx(i+1,j)] + u[idx(i-1,j)] +
        u[idx(i,j+1)] + u[idx(i,j-1)] -
        4*u[idx(i,j)]
    ) / (dx*dx) + f[idx(i,j)];
}

double PoissonSolver2D::compute_max_residual() const 
{
    double max_r = 0.0;
    for (int j = 1; j < N; j++) 
    {
        for (int i = 1; i < N; i++) 
        {
            max_r = std::max(max_r, std::abs(compute_residual_at(i,j)));
        }
    }
    return max_r;
}

std::vector<double> PoissonSolver2D::get_energy_history() const 
{
    return energy_history;
}

std::vector<double> PoissonSolver2D::get_residual_history() const 
{
    return residual_history;
}

std::vector<std::vector<double>> PoissonSolver2D::get_solution() const 
{
    std::vector<std::vector<double>> grid(N+1, std::vector<double>(N+1));
    for (int j = 0; j <= N; ++j) {
        for (int i = 0; i <= N; ++i) {
            grid[j][i] = u[idx(i,j)];
        }
    }
    return grid;
}

void PoissonSolver2D::solve_with_refinement(int lower_n, int max_iters, double tol, double omega) 
{
    if (lower_n >= this->n_power)
    {
        std::cout << "Error: lower_n must be < final n_power\n";
        return;
    }
    
    int final_n = this->n_power;  // Save the final target level
    
    std::vector<std::vector<double>> coarse_solution;
    std::vector<double> accumulated_energy;
    std::vector<double> accumulated_residual;
    refinement_timing_history.clear();
    
    // Phase 1: Solve on coarsest level (lower_n)
    {
        int original_N = this->N;
        double original_dx = this->dx;
        
        // Temporarily set to coarse grid
        this->n_power = lower_n;
        this->N = 1 << lower_n;
        this->dx = 3.0 / this->N;
        
        // Reinitialize u, f for coarse grid
        initialize();
        
        // Solve on coarse grid and record time for this level
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            solve_sor(max_iters, tol, omega);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            refinement_timing_history.push_back(static_cast<double>(dur));
        }
        coarse_solution = get_solution();

        // Accumulate histories from coarse level
        accumulated_energy = energy_history;
        accumulated_residual = residual_history;
        solution_history_refinement.push_back(coarse_solution);
    }
    
    // Phase 2: Refinement loop
    int current_n = lower_n;
    while (current_n < final_n)  // Compare against FIXED final_n, not this->n_power
    {
        current_n++;
        
        // Reinitialize u, f for this level
        this->n_power = current_n;
        this->N = 1 << current_n;
        this->dx = 3.0 / this->N;
        
        // Reinitialize grids
        u.assign((N+1)*(N+1), 0.0);
        f.assign((N+1)*(N+1), 0.0);
        compute_rhs();
        
        // Interpolate coarse solution to fine grid
        set_solution_from_coarse(coarse_solution, current_n - 1);
        
        // Solve on fine grid and record time for this level
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            solve_sor(max_iters, tol, omega);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            refinement_timing_history.push_back(static_cast<double>(dur));
        }

        // Accumulate histories from this level
        accumulated_energy.insert(accumulated_energy.end(), energy_history.begin(), energy_history.end());
        accumulated_residual.insert(accumulated_residual.end(), residual_history.begin(), residual_history.end());

        coarse_solution = get_solution();
        solution_history_refinement.push_back(coarse_solution);
    }
    
    // Set final state (n_power and N should already be at final_n from last iteration)
    this->n_power = final_n;
    this->N = 1 << final_n;
    this->dx = 3.0 / this->N;
    
    // Set final accumulated histories
    energy_history = accumulated_energy;
    residual_history = accumulated_residual;
}


std::vector<std::vector<std::vector<double>>> PoissonSolver2D::get_solution_history_refinement() const 
{
    return solution_history_refinement;
}

std::vector<double> PoissonSolver2D::get_refinement_timing_history() const
{
    return refinement_timing_history;
}

void PoissonSolver2D::set_solution_from_coarse(const std::vector<std::vector<double>>& coarse_sol, int coarse_n_power)
{
    // Interpolate solution from coarser grid to current grid
    int N_coarse = (1 << coarse_n_power);
    double dx_coarse = 3.0 / N_coarse;
    
    // Bilinear interpolation from coarse grid to fine grid
    for (int j = 0; j <= N; ++j)
    {
        for (int i = 0; i <= N; ++i)
        {
            double x = i * dx;
            double y = j * dx;
            
            // Find position in coarse grid
            double i_c = x / dx_coarse;
            double j_c = y / dx_coarse;
            
            int i0 = (int)std::floor(i_c);
            int i1 = std::min(i0 + 1, N_coarse);
            int j0 = (int)std::floor(j_c);
            int j1 = std::min(j0 + 1, N_coarse);
            
            // Interpolation weights
            double wi = i_c - i0;
            double wj = j_c - j0;
            
            // Bilinear interpolation
            double u_interp = 
                (1.0 - wi) * (1.0 - wj) * coarse_sol[j0][i0] +
                wi * (1.0 - wj) * coarse_sol[j0][i1] +
                (1.0 - wi) * wj * coarse_sol[j1][i0] +
                wi * wj * coarse_sol[j1][i1];
            
            u[idx(i, j)] = u_interp;
        }
    }
}
