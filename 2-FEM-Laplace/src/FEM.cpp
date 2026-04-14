#include "FEM.hpp"

FEM::FEM(double L, size_t N, double rc_x, double rc_y, BasisType basis)
    : L(L), N(N), rc_x(rc_x), rc_y(rc_y), basis_(basis)
{
    if (basis_ == BasisType::Bilinear) {
        nodes_per_element = 4;
        this->num_nodes = (N + 1) * (N + 1);
    } else if (basis_ == BasisType::Quadratic) {
        nodes_per_element = 9;
        this->num_nodes = (2 * N + 1) * (2 * N + 1);
    } else {
        throw std::invalid_argument("Unsupported basis type");
    }
    this->a = L / N;

    initialize();
}

void FEM::build_mesh()
{
    
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int k = i * N + j; // element index
            double y = i * a;
            double x = j * a;
            int gn = addNode(x,y); // always new at this stage
            nlg[k * nodes_per_element + 0] = gn; // anchor node
        }
    }


    const double offsets[9][2] = 
    {
            {0,    0   },   // 0: anchor (already done)
            {a_,   0   },   // 1: bottom-right
            {0,    a_  },   // 2: top-left
            {a_,   a_  },   // 3: top-right
            {a_/2, 0   },   // 4: bottom edge midpoint
            {a_,   a_/2},   // 5: right edge midpoint
            {0,    a_/2},   // 6: left edge midpoint
            {a_/2, a_  },   // 7: top edge midpoint
            {a_/2, a_/2}    // 8: centre
    };




}

void FEM::initialize()
{
    nlg.resize(num_nodes);
    nodeX_.resize(num_nodes);
    nodeY_.resize(num_nodes);
    coefficients.resize(num_nodes, 0.0);
    rhs.resize(num_nodes, 0.0);
    stiffness_matrix.resize(num_nodes, std::vector<double>(num_nodes, 0.0));
    
}


