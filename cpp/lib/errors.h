#ifndef FUSINTER_V3_ERRORS_H
#define FUSINTER_V3_ERRORS_H
#include<exception>
namespace lib {
    struct NOT_SORTED_ERROR : public std::exception {};
    struct NOT_MATCHING_DATA_SIZES : public std::exception {};
}
#endif //FUSINTER_V3_ERRORS_H
