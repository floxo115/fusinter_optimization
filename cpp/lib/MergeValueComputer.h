#ifndef FUSINTER_V3_MERGEVALUECOMPUTER_H
#define FUSINTER_V3_MERGEVALUECOMPUTER_H

#include<Eigen/Dense>

#include<cmath>

#include "typedefs.h"

namespace lib {
// TODO implement this
    void removeColumn(lib::table &matrix, unsigned int colToRemove) {
        unsigned int numRows = matrix.rows();
        unsigned int numCols = matrix.cols() - 1;

        if (colToRemove < numCols)
            matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows,
                                                                                        numCols - colToRemove);

        matrix.conservativeResize(numRows, numCols);
    }

    float shannon_entropy(
            Eigen::VectorXi input_column,
            float alpha,
            float lam,
            int m,
            int n
    ) {

        float col_sum = 0;
        int n_j = input_column.sum();
        float col_fac = alpha * n_j / n;

        for (int i = 0; i < m; i++) {
            float p = (input_column[i] + lam) / (n_j + m * lam);
            col_sum += -(p * std::log2(p));
        }
        return col_fac * col_sum + (1 - alpha) * m * lam / n_j;
    }

    class MergeValueComputer {
    private:
        lib::table table;
        float alpha;
        float lam;
        int m;
        int n;
        std::vector<float> cols_entropy;
        std::vector<float> deltas;

    public:
        MergeValueComputer(const lib::table &table, float alpha, float lam)
                : table(table),
                  alpha(alpha),
                  lam(lam) {
            this->m = table.rows();
            this->n = table.sum();

            for (int i = 0; i < this->table.cols(); i++) {
                auto col = this->table.col(i);
                auto entropy = shannon_entropy(col, this->alpha, this->lam, this->m, this->n);
                this->cols_entropy.push_back(entropy);
            }

            for (int i = 0; i < this->table.cols() - 1; i++) {
                auto delta = compute_delta(i);
                this->deltas.push_back(delta);
            }
        };

        void update(const lib::table &table, int max_ind) {
            this->table = table;

            this->cols_entropy[max_ind] = shannon_entropy(this->table.col(max_ind), this->alpha, this->lam, this->m,
                                                          this->n);
            this->cols_entropy.erase(this->cols_entropy.begin() + max_ind + 1);

            if (max_ind != 0)
                this->deltas[max_ind - 1] = this->compute_delta(max_ind, true);
            if (max_ind != this->deltas.size() - 1)
                this->deltas[max_ind + 1] = this->compute_delta(max_ind);

            this->deltas.erase(this->deltas.begin() + max_ind);
        }

        float get_table_entropy() {
            float result = 0;
            for (auto el: this->cols_entropy)
                result += el;
            return result;
        };

        std::vector<float> get_all_deltas() {
            return this->deltas;
        }

    private:
        float compute_merge_entropy(int col_idx) {
            auto col = this->table.block(0, col_idx, this->m, 2)
                    .rowwise()
                    .sum();

            return shannon_entropy(col, this->alpha, this->lam, this->m, this->n);
        };

        float compute_delta(int col_idx, bool left = false) {
            if (left)
                col_idx = col_idx - 1;

            float entropy = 0;
            entropy += this->cols_entropy[col_idx];
            entropy += this->cols_entropy[col_idx + 1];
            entropy -= this->compute_merge_entropy(col_idx);
            return entropy;
        };
    };
}

#endif
