#ifndef FUSINTER_V3_FUSINTER_H
#define FUSINTER_V3_FUSINTER_H

#include "typedefs.h"
#include <cassert>
#include<map>
#include<set>
#include<algorithm>
#include <Eigen/Dense>

namespace lib {
    class FUSINTERDiscretizer {
    private:
        float alpha;
        float lam;
        Eigen::VectorXf data_x;
        Eigen::VectorXi data_y;

    public:
        FUSINTERDiscretizer(float alpha, float lam)
        :alpha(alpha), lam(lam){
            assert(alpha > 0);
            assert(lam > 0);
        };

        std::vector<float> fit(lib::data_vec data_x,lib::label_vec data_y){

            auto label_set = std::set<int> {data_y.begin(), data_y.end()};
            std::map<int, int> label_map;

            int i = 0;
            for(auto el:label_set){
                label_map[el] = i;
                i++;
            }
            for(int i = 0; i < data_y.size(); i++){
                data_y[i] = label_map.at(data_y[i]);
            }

            Eigen::VectorXi argsorts(data_x.size());
            for (int i = 0; i < data_x.size(); i++)
                argsorts[i] = i;
            std::sort(argsorts.begin(), argsorts.end(), [&data_x](int i1, int i2){
                return data_x[i1] < data_x[i2];
            });

            this->data_x = Eigen::VectorXf(data_x.size());
            this->data_y = Eigen::VectorXi(data_x.size());
            for(int i = 0; i < data_x.size(); i++){
                this->data_x[i] = data_x[argsorts[i]];
                this->data_y[i] = data_y[argsorts[i]];
            }

            auto splitter = lib::Splitter(this->data_x, this->data_y);
            auto splits = splitter.apply();
            auto tablemanager = lib::TableManager(this->data_x, this->data_y);
            auto table = tablemanager.create_table(splits);
            auto mvc = lib::MergeValueComputer(table, this->alpha, this->lam);

            int n_run = 0;
            int max_runs = splits.size();
            while(n_run < max_runs ){
                auto split_values = mvc.get_all_deltas();

                int max_ind = 0;
                float max_el = split_values[0];
                for (int i = 1; i < split_values.size(); i++){
                    if(split_values[i] > max_el){
                        max_el = split_values[i];
                        max_ind = i;
                    }
                }

                if(split_values[max_ind] <= 0)
                    break;

                splits.erase(splits.begin()+max_ind);
                table = tablemanager.compress_table(table, max_ind);
                mvc.update(table, max_ind);
                n_run++;
            }

            return splits;
        }

    };
}

#endif