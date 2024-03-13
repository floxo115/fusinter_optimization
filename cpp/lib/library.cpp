#include "library.h"

#include <iostream>
#include"Eigen/Dense"

void hello() {
    std::cout << "Hello, World!" << std::endl;
    Eigen::MatrixXd m(2,2);
    std::cout << m << std::endl;
}
