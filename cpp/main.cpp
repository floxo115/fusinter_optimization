//
// Created by floxo on 12/28/23.
//

#include "paper_data.h"
#include <iostream>

#include "lib/library.h"
int main() {
//    auto splitter = lib::Splitter(paper_data_x, paper_data_y);
//    auto splits = splitter.apply();
//
////    for (auto el: splits){
////        std::cout << el << ", ";
////    }
//
////    std::cout << std::endl;
//
//
//    auto tm = lib::TableManager(paper_data_x, paper_data_y);
//    auto table = tm.create_table(splits);
//    std::cout << table << std::endl;
//
//    auto comp_table = tm.compress_table(table, 1);
//    std::cout << comp_table << std::endl;

//    Eigen::Matrix<int, -1, -1> table(3,3);
//    table << 1,0,3,1,5,2,7,3,9;
//    std::cout << table << std::endl;
//    auto mvc = lib::MergeValueComputer(table, 0.2, 0.7);

    auto discretizer = lib::FUSINTERDiscretizer(0.95, 0.99);

    auto data_x = paper_data_x;
    auto data_y = paper_data_y;

    auto splits = discretizer.fit(data_x, data_y);
    for (auto el: splits) {
        std::cout << el << std::endl;
    }
}