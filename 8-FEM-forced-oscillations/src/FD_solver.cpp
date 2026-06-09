#include "FD_solver.hpp"
#include <cmath>
#include <algorithm>

// Constructor
FD_solver::FD_solver(size_t N, double L, double v, double damping, double omega)
    : m_N(N), m_L(L), m_v(v), m_damping(damping), m_omega(omega), m_t(0.0)
{
    m_dx = m_L / static_cast<double>(m_N);
    m_dt = m_dx / (2.0 * m_v);
    m_spatial_coeff = (m_dt * m_dt * m_v * m_v) / (m_dx * m_dx);

    size_t grid_elements = m_N * m_N;
    
    m_u_p.resize(grid_elements, 0.0);
    m_u_c.resize(grid_elements, 0.0);
    m_u_n.resize(grid_elements, 0.0);

    initialize();
}

FD_solver::~FD_solver() {}

void FD_solver::initialize()
{
    std::fill(m_u_p.begin(), m_u_p.end(), 0.0);
    std::fill(m_u_c.begin(), m_u_c.end(), 0.0);
    m_t = 0.0;
    
    m_energy_history.clear();
    m_center_disp_history.clear();
}

// Enforces Dirichlet boundary conditions (outermost edges fixed to 0)
void FD_solver::apply_BC()
{
    for (size_t k = 0; k < m_N; ++k) {
        m_u_n[idx(k, 0)] = 0.0;
        m_u_n[idx(k, m_N - 1)] = 0.0;
        
        m_u_n[idx(0, k)] = 0.0;
        m_u_n[idx(m_N - 1, k)] = 0.0;
    }
}

// Single finite difference time step
void FD_solver::step()
{
    for (size_t i = 1; i < m_N - 1; ++i) {
        for (size_t j = 1; j < m_N - 1; ++j) {
            
            double laplacian = m_u_c[idx(i + 1, j)] + m_u_c[idx(i - 1, j)] + 
                               m_u_c[idx(i, j + 1)] + m_u_c[idx(i, j - 1)] - 
                               4.0 * m_u_c[idx(i, j)];

            m_u_n[idx(i, j)] = m_u_c[idx(i, j)] * (2.0 - m_damping * m_dt) + 
                               m_u_p[idx(i, j)] * (m_damping * m_dt - 1.0) + 
                               m_spatial_coeff * laplacian;

            size_t c = m_N / 2;
            if (i == c && j == c) {
                m_u_n[idx(i, j)] += (m_dt * m_dt) * std::sin(m_omega * m_t);
            }
        }
    }

    apply_BC();

    size_t c = m_N / 2;
    m_center_disp_history.push_back(m_u_c[idx(c, c)]);
    m_energy_history.push_back(calculate_total_energy());

    m_t += m_dt;
    m_u_p = m_u_c;
    m_u_c = m_u_n;
}

void FD_solver::evolve(double t_final)
{
    while (m_t < t_final) {
        step();
    }
}

// Total energy calculation E(t)
double FD_solver::calculate_total_energy()
{
    double energy_sum = 0.0;

    for (size_t i = 1; i < m_N - 1; ++i) {
        for (size_t j = 1; j < m_N - 1; ++j) {
            
            double du_dt = (m_u_c[idx(i, j)] - m_u_p[idx(i, j)]) / m_dt;
            double du_dx = (m_u_c[idx(i + 1, j)] - m_u_c[idx(i, j)]) / m_dx;
            double du_dy = (m_u_c[idx(i, j + 1)] - m_u_c[idx(i, j)]) / m_dx;

            double energy_density = (du_dt * du_dt) + (du_dx * du_dx) + (du_dy * du_dy);
            energy_sum += energy_density;
        }
    }

    return 0.5 * energy_sum * (m_dx * m_dx);
}

std::vector<double> FD_solver::get_energy_history()
{
    return m_energy_history;
}

std::vector<double> FD_solver::get_displacement_history()
{
    return m_center_disp_history;
}
