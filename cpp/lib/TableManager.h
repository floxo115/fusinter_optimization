#ifndef FUSINTER_V3_TABLEMANAGER_H
#define FUSINTER_V3_TABLEMANAGER_H

#include<vector>
#include<set>

#include <Eigen/Dense>

#include "typedefs.h"
#include "errors.h"

namespace lib {
    class TableManager {
    private:
        lib::data_vec data_x;
        lib::label_vec data_y;

    public:
        TableManager(const lib::data_vec &data_x, const lib::label_vec &data_y) : data_x(data_x), data_y(data_y) {
            if (!std::is_sorted(data_x.begin(), data_x.end())) {
                throw lib::NOT_SORTED_ERROR();
            }
            if (this->data_x.size() != this->data_y.size()) {
                throw lib::NOT_MATCHING_DATA_SIZES();
            }
        }

        lib::table create_table(const std::vector<float> &init_splits) {
            auto n_labels = std::set<int>{this->data_y.begin(), this->data_y.end()}.size();
            auto n_splits = init_splits.size() + 1;

            Eigen::Matrix<int, -1, -1> table(n_labels, n_splits);
            table.setZero();

            Eigen::VectorXi n_labels_in_interval(n_labels);
            n_labels_in_interval.setZero();

            auto i = 0;
            for (auto split_idx = 0; split_idx < init_splits.size(); split_idx++) {
                auto split_val = init_splits[split_idx];
                while (this->data_x[i] < split_val) {
                    n_labels_in_interval[this->data_y[i]] += 1;
                    i += 1;
                }

                table.col(split_idx) = n_labels_in_interval;
                n_labels_in_interval.setZero();
            }

            while (i < this->data_x.size()) {
                n_labels_in_interval[this->data_y[i]] += 1;
                i += 1;
            }
            table.col(init_splits.size()) = n_labels_in_interval;

            return table;
        }

        static lib::table compress_table(const table &input_table, const int i) {
            int n = input_table.rows();
            int m = input_table.cols();

            if (i < 0 || (m - 1) < i) {
                // TODO create Exception class
                throw "the parameter i has to have values between 0 and (len of columns -1), but is {i}";
            }

            Eigen::Matrix<int, -1, -1> new_table(n, m - 1);

            int init_table_index = 0;
            int new_table_index = 0;
            //TODO refactor this code
            while (init_table_index < m) {
                if (init_table_index == i) {
                    auto new_col = new_table.col(new_table_index);
                    auto old_left_col = input_table.col(init_table_index);
                    auto old_right_col = input_table.col(init_table_index + 1);
                    new_col = old_left_col + old_right_col;
                    new_table_index += 1;
                } else if (init_table_index != i + 1) {
                    auto new_col = new_table.col(new_table_index);
                    new_col = input_table.col(init_table_index);
                    new_table_index += 1;
                }

                init_table_index += 1;

            }
            return new_table;
        }
    };
}
#endif //FUSINTER_V3_TABLEMANAGER_H
