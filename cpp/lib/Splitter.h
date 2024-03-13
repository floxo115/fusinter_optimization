#ifndef FUSINTER_V3_SPLITTER_H
#define FUSINTER_V3_SPLITTER_H

#include<vector>
#include<iostream>
#include<tuple>

#include "typedefs.h"
#include "errors.h"

namespace lib {

    class Splitter {
    private:
        data_vec data_x;
        label_vec data_y;

    public:
        Splitter(const data_vec &data_x, const label_vec &data_y) : data_x(data_x), data_y(data_y) {
            if (!std::is_sorted(data_x.begin(), data_x.end()))
                throw NOT_SORTED_ERROR();
            if (data_y.size() != data_x.size())
                throw NOT_MATCHING_DATA_SIZES();
        };


        std::vector<float> apply() {
            std::vector<float> splits;
            std::vector<int> labels;
            int index = 0;
            int label;

            std::tie(label, index) = this->get_label_of_next_value(index);
            labels.push_back(label);
//            auto tuple = this->get_label_of_next_value(index);
//            auto label = std::get<0>(tuple);
//            index = std::get<1>(tuple);

            while (index < this->data_x.size()) {
                std::tie(label, index) = this->get_label_of_next_value(index);

                if (label != labels.back() || labels.back() == -1) {
                    splits.push_back(this->data_x[index - 1]);
                    labels.push_back(label);
                }
            }
            return splits;
        }

    protected:
        std::tuple<int, int> get_label_of_next_value(int index) {
            float value = this->data_x[index];
            int label = this->data_y[index];
            index++;

            while (index < this->data_x.size()) {
                if (value != this->data_x[index]) {
                    break;
                }

                if (label != this->data_y[index]) {
                    label = -1;
                }

                index++;
            }

            return std::make_tuple(label, index);
        }
    };
}

#endif //FUSINTER_V3_SPLITTER_H
