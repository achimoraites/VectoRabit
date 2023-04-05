// VectoRabit.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <Eigen/Dense>

#include "VectoRabit.h"

using namespace std;


int main() {
    Eigen::Matrix2d A;
    A << 1, 2,
        3, 4;
    Eigen::Vector2d b(5, 6);
    Eigen::Vector2d x = A.colPivHouseholderQr().solve(b);

    std::cout << "Solution: " << x.transpose() << std::endl;

    return 0;
}
