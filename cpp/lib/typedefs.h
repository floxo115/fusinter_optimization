#ifndef FUSINTER_V3_TYPEDEFS_H
#define FUSINTER_V3_TYPEDEFS_H

#include<Eigen/Dense>
namespace lib {
    typedef Eigen::Matrix<float, Eigen::Dynamic, 1> data_vec;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> label_vec;
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> table;
}
#endif //FUSINTER_V3_TYPEDEFS_H
