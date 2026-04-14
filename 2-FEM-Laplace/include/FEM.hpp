#include <iostream>
#include <vector>
#include "spdlog/spdlog.h"


enum class BasisType { Bilinear, Quadratic };


class FEM 
{
public:
    // getters:
    double get_a() const { return a; }
    size_t get_N() const { return N; }
    double get_L() const { return L; }
    std::vector<double> get_coefficients() const { return coefficients; }
    std::vector<double> get_RHS() const { return RHS; }
    std::vector<std::vector<double>> get_stiffness_matrix() const { return stiffness_matrix; }
    std::vector<double> get_u() const { return u; }
    std::vector<std::vector<double>> get_ngl() const { return ngl; }

    FEM(double L, size_t N, double rc_x, double rc_y, BasisType basis = BasisType::Bilinear);
    void assemble_stiffness_matrix();
    void solve();
    void build_mesh();
    void apply_boundary_conditions();




private:
    double a; // size of the square L / N
    size_t N; // number of elements per side
    double L; // box length
    double rc_x, rc_y; // source point coordinates
    int nodes_per_element; // 4 / 9

    BasisType basis_; // type of basis functions
    

    size_t num_nodes; // number of nodes
    std::vector<size_t> nlg; // global local mapping i * n + j 
    std::vector<double> nodeX_;
    std::vector<double> nodeY_;

    std::vector<double> coefficients;
    std::vector<double> rhs; // load vector
    std::vector<std::vector<double>> stiffness_matrix; // stiffness matrix



    int  findNode(double x, double y) const;

    int  addNode(double x, double y);
    bool isBoundaryNode(int i) const;

    std::vector<std::vector<double>> localStiffness_;
    void computeLocalStiffness();

    void initialize; // initializes the mesh and local stiffness matrix



};


