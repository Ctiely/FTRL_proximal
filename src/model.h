//
// Created by Clytie on 2018/8/12.
//

#ifndef FTRL_MODEL_H
#define FTRL_MODEL_H

#include "dataset.h"

#include <vector>
using namespace std;


class model {
public:
    explicit model(string file_path);
    explicit model(unsigned long num_features, float alpha=1, float beta=1, float lambda1=0, float lambda2=0);
    ~model() = default;

    unsigned long num_features;
    float alpha, beta, lambda1, lambda2;
    float w_intercept;
    vector<float> w;
    void fit(dataset * pxt);
    float predict_proba(dataset * px);
    int predict(dataset * px, float threshold=0.5);
    int load_model(string & file_path);
    int save_model(string file_path);
    int fit(string data_file);

private:
    float z_intercept, n_intercept;
    vector<float> z, n;

    void init_model();
    float sigmoid(float x);
    float sign(float z);
    float computeSigma(float ni, float gi);
    void undateIntercept();
    void updateW(unsigned int i);
};


#endif //FTRL_MODEL_H
