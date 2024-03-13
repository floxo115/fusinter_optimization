#define CATCH_CONFIG_MAIN

#include<tuple>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../paper_data.h"
#include "errors.h"
#include "Splitter.h"
#include "TableManager.h"
#include "typedefs.h"
#include "MergeValueComputer.h"

/////////////////////////////////////////////////////// SPLITTER TESTS /////////////////////////////////////////////////

TEST_CASE("Splitter with unsorted input") {
    auto data_y = paper_data_y;
    auto data_x = paper_data_x;

    data_x[0] = data_x[70];
    SECTION("should throw NOT_SORTED_ERROR") {
        REQUIRE_THROWS_AS(lib::Splitter(data_x, data_y), lib::NOT_SORTED_ERROR);
    }
}

TEST_CASE("Splitter with non matching labels and data vectors") {
    auto data_y = paper_data_y(Eigen::seq(0, 80));
    auto data_x = paper_data_x;
    SECTION("should throw NOT_MATCHING_DATA_SIZES") {
        REQUIRE_THROWS_AS(lib::Splitter(data_x, data_y), lib::NOT_MATCHING_DATA_SIZES);
    }
}

struct splitter_test_data {
    lib::data_vec data_x;
    lib::label_vec data_y;
    std::vector<float> expected;
};

TEST_CASE("Splitter with different valid inputs") {
    auto data = GENERATE(
            splitter_test_data(
                    paper_data_x,
                    paper_data_y,
                    {2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40}
            ),
            splitter_test_data(
                    lib::data_vec{{-10, -10, -10., -9., -9., -8., -8., -8., -8., 2., 2., 3., 3.}},
                    lib::label_vec{{0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2}},
                    {-8., 2.}
            )
    );
    auto splitter = lib::Splitter(data.data_x, data.data_y);
    REQUIRE(splitter.apply() == data.expected);
}

/////////////////////////////////////////////////// TABLE MANAGER TESTS ///////////////////////////////////////////////

TEST_CASE("TableManager with unsorted input") {
    auto data_y = paper_data_y;
    auto data_x = paper_data_x;

    data_x[0] = data_x[70];
    SECTION("should throw NOT_SORTED_ERROR") {
        REQUIRE_THROWS_AS(lib::TableManager(data_x, data_y), lib::NOT_SORTED_ERROR);
    }
}

TEST_CASE("TableManager with unequal sized input") {
    auto data_y = lib::label_vec{{0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}};
    auto data_x = paper_data_x;

    SECTION("should throw NOT_SORTED_ERROR") {
        REQUIRE_THROWS_AS(lib::TableManager(data_x, data_y), lib::NOT_MATCHING_DATA_SIZES);
    }
}

struct table_manager_test_input {
    lib::data_vec data_x;
    lib::label_vec data_y;
    std::vector<float> input_splits;
    lib::table expected;
};

TEST_CASE("TableManager create_table with valid inputs") {
    auto data = GENERATE(
            table_manager_test_input(
                    paper_data_x,
                    paper_data_y,
                    {2, 3, 13, 14, 15, 16, 17, 18, 19, 20, 23, 37, 38, 39, 40},
                    lib::table{{1, 0, 0, 1, 26, 0, 2, 1, 3, 0, 0, 3, 3, 0, 0, 3, 2, 1, 0, 2, 5, 0, 0, 27, 1,
                                2, 2, 1, 0, 2, 2, 0}}.reshaped(2, 16)
            ),
            table_manager_test_input(
                    lib::data_vec{{-10, -10, -10, -9, -9, -8, -8, -8, -8, 2, 2, 3, 3}},
                    lib::label_vec{{0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2}},
                    {-8., 2.},
                    lib::table{{5, 0, 0, 2, 2, 0, 0, 0, 4}}.reshaped(3, 3)
            )
    );

    auto tm = lib::TableManager(data.data_x, data.data_y);
    REQUIRE(tm.create_table(data.input_splits) == data.expected);
}

TEST_CASE("TableManager compress_table") {
    auto input_table = lib::table(3, 3);
    input_table << 2, 1, 0, 0, 2, 4, 0, 3, 0;
    auto expected_tables = std::vector<lib::table>{
            lib::table{{3, 2, 3, 0, 4, 0}}.reshaped(3, 2),
            lib::table{{2, 0, 0, 1, 6, 3}}.reshaped(3, 2)
    };

    auto table = lib::TableManager::compress_table(input_table, 0);
    REQUIRE(table == expected_tables[0]);
    table = lib::TableManager::compress_table(input_table, 1);
    REQUIRE(table == expected_tables[1]);
}

/////////////////////////////////////////////MERGE_VALUE_COMPUTER //////////////////////////////

struct ShannonEntropyTestInput {
    Eigen::VectorXi column;
    float alpha;
    float lam;
    int m;
    int n;
    float expected;
};

TEST_CASE("Shannon Entauto data") {
    auto data = GENERATE(
            ShannonEntropyTestInput(Eigen::VectorXi{{1, 4, 5, 3, 6, 7}}, 0.5, 0.5, 6, 20, 1.649793),
            ShannonEntropyTestInput(Eigen::VectorXi{{6,4,7,2,1,6,7,8,5,4}}, 0.2, 0.7, 10,50, 0.753156)
    );

    auto result = lib::shannon_entropy(data.column, data.alpha, data.lam, data.m, data.n);
    result *= std::pow(10, 6);
    result = std::ceil(result);

    auto expected = data.expected;
    expected *= std::pow(10, 6);
    expected = std::ceil(expected);

    REQUIRE(result == expected);
}


TEST_CASE("MergeValueComputer") {
    Eigen::MatrixXi table(3,5);
    table << 1,0,3,1,5,2,7,3,9,3,2,6,7,8,4;

    SECTION("alpha = 0.5, lam = 0.5"){
        auto mvc = lib::MergeValueComputer(table, 0.5, 0.5);
        std::vector<float> expected {0.161442  , 0.06892525, 0.06215969, 0.05056034};
        auto results = mvc.get_all_deltas();
        REQUIRE(results.size() == expected.size());
        for(int i = 0; i < results.size(); i++){
            auto r = results[i] * 100000;
            r = std::ceil(r);
            auto e = expected[i] * 100000;
            e = std::ceil(e);
            REQUIRE(r == e);
        }

        Eigen::MatrixXi table1(3,4);
        table1 << 1, 0, 3, 6, 2, 7, 3, 12, 2, 6, 7, 12;
        mvc.update(table1, 3);
        results = mvc.get_all_deltas();
        expected = {0.161442  , 0.06892525, 0.06065164};
        REQUIRE(results.size() == expected.size());
        for(int i = 0; i < results.size(); i++){
            auto r = results[i] * 100000;
            r = std::ceil(r);
            auto e = expected[i] * 100000;
            e = std::ceil(e);
            REQUIRE(r == e);
        }

        Eigen::MatrixXi table2(3,3);
        table2 << 1,3,6,9,3,12,8,7,12;
        mvc.update(table2, 0);
        results = mvc.get_all_deltas();
        expected = {0.06215969, 0.06065164};
        REQUIRE(results.size() == expected.size());
        for(int i = 0; i < results.size(); i++){
            auto r = results[i] * 100000;
            r = std::ceil(r);
            auto e = expected[i] * 100000;
            e = std::ceil(e);
            REQUIRE(r == e);
        }
    }
}
