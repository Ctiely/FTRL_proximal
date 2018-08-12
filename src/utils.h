//
// Created by Clytie on 2018/8/12.
//

#ifndef FTRL_UTILS_H
#define FTRL_UTILS_H

#include "constants.h"
#include "dataset.h"

#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
using namespace std;


template <typename T>
void saveAslibsvm(vector<T> & arr, FILE * fout) {
    for (unsigned int i = 0; i < arr.size(); ++i) {
        if (abs(arr[i]) > TOLERANCE) {
            fprintf(fout, "%u:%g", i + 1, arr[i]);
            if (i != arr.size() - 1) {
                fprintf(fout, " ");
            }
        }
    }
    fprintf(fout, "\n");
}

template <typename T>
ftrl_data<T> readFromlibsvm(const char * cdata) {
    unsigned int begin = 0;
    ftrl_data<T> data;
    char prefix = ' ';
    while (cdata[begin] != '\0') {
        if (prefix == ' ') {
            if (cdata[begin + 1] == '\n') { //if there is a \n after a space, then we go to next line
                break;
            }
            data.indexs.push_back((unsigned int)atoi(cdata + begin) - 1);
            while (cdata[begin] != '\0' && cdata[begin] != ':') {
                ++begin;
            }
        } else {
            data.data.push_back((float)atof(cdata + begin + 1));
            while (cdata[begin] != '\0' && cdata[begin] != ' ') {
                ++begin;
            }
        }
        prefix = cdata[begin];
    }
    return data;
}

template <typename T>
vector<T> readFromSep(const char * cdata, char sep=' ') {
    vector<T> arr;
    unsigned int begin = 0;
    while (cdata[begin] != '\0') {
        arr.push_back((float)atof(cdata + begin));
        ++begin;
        while (cdata[begin] != '\0' && cdata[begin] != sep) {
            ++begin;
        }
    }
    return arr;
}

template <typename T>
vector<T> libsvm2sparse(const char * cdata, unsigned long num_features) {
    ftrl_data<T> data = readFromlibsvm<float>(cdata);
    vector<T> sparse(num_features);
    for (int i = 0; i < data.indexs.size(); ++i) {
        if (data.indexs[i] < num_features) {
            sparse[data.indexs[i]] = data.data[i];
        }
    }
    return sparse;
}
#endif //FTRL_UTILS_H
