#pragma once

#include <vector>
#include <cstddef>

class FD_solver
{
public:
    FD_solver(size_t N, double L, double v,double damping, double omega);
    ~FD_solver();

    void step();
    void evolve(double t_final);

    std::vector<double> get_energy_history();
    std::vector<double> get_displacement_history();

    

private:
    
    size_t m_N; // N+1
    double m_L;
    double m_v;
    double m_damping;
    double m_omega;
    double m_dx; // L / N
    double m_dt; // dx / (2 * v)
    double m_t;  // current time
    
    double m_spatial_coeff;

    std::vector<double> m_u_p;
    std::vector<double> m_u_c;
    std::vector<double> m_u_n;


    std::vector<double> m_energy_history;
    std::vector<double> m_center_disp_history;

    void initialize();
    void apply_BC();
    double calculate_total_energy();

    inline size_t idx(size_t i, size_t j){return i*m_N + j;}
};
