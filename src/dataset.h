//
// Created by Clytie on 2018/8/12.
//

#ifndef FTRL_DATASET_H
#define FTRL_DATASET_H

#include <vector>
#include <string>
using namespace std;

template <typename T>
struct ftrl_data {
    ftrl_data() = default;
    vector<unsigned int> indexs;
    vector<T> data;
};

class dataset {
public:
    explicit dataset(string & dataLine);
    ~dataset() = default;

    unsigned int length;
    ftrl_data<float> data;
    int label;

private:
    string dataLine;
    void readLine();
};


#endif //FTRL_DATASET_H
