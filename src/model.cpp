//
// Created by Clytie on 2018/8/12.
//

#include "model.h"
#include "utils.h"

#include <cassert>
using namespace std;


model::model(string file_path) {load_model(file_path);}

model::model(unsigned long num_features, float alpha, float beta, float lambda1, float lambda2)
        : num_features(num_features), alpha(alpha), beta(beta), lambda1(lambda1), lambda2(lambda2) {init_model();}

void model::init_model() {
    assert(alpha > 0), assert(beta >= 0), assert(lambda1 >= 0), assert(lambda2 >= 0);
    z = vector<float>(num_features);
    n = vector<float>(num_features);
    w = vector<float>(num_features);
    z_intercept = 0;
    n_intercept = 0;
    w_intercept = 0;
}

float model::sigmoid(float x) {
    if (x <= -35.0f) {
        return 0.000000000000001f;
    } else if (x >= 35.0f) {
        return 0.999999999999999f;
    }

    return 1.0f / (1.0f + exp(-x));
}

float model::sign(float z) {
    if (z < 0) {
        return -1.0f;
    } else {
        return 1.0f;
    }
}

float model::computeSigma(float ni, float gi) {
    return (sqrt(ni + gi * gi) - sqrt(ni)) / alpha;
}

void model::undateIntercept() {
    if (abs(z_intercept) <= lambda1) {
        w_intercept = 0;
    } else {
        w_intercept = -(z_intercept - sign(z_intercept) * lambda1) / ((beta + sqrt(n_intercept)) / alpha + lambda2);
    }
}

void model::updateW(unsigned int i) {
    if (abs(z[i]) <= lambda1) {
        w[i] = 0;
    } else {
        w[i] = -(z[i] - sign(z[i]) * lambda1) / ((beta + sqrt(n[i])) / alpha + lambda2);
    }
}

void model::fit(dataset * pxt) {
    float pt = predict_proba(pxt);
    float grad = pt - pxt->label;

    //update interpect
    float sigma_intercept = computeSigma(n_intercept, grad);
    z_intercept += grad - sigma_intercept * w_intercept;
    n_intercept += grad * grad;

    //update coefficient
    for (int i = 0; i < pxt->length; ++i) {
        float gi = grad * pxt->data.data[i];
        unsigned int idx = pxt->data.indexs[i];
        if (idx < num_features) {
            float sigma = computeSigma(n[idx], gi);
            z[idx] += gi - sigma * w[idx];
            n[idx] += gi * gi;
        }
    }

    undateIntercept();
    for (auto idx : pxt->data.indexs) {
        if (idx < num_features) {
            updateW(idx);
        }
    }
}

int model::fit(string data_file) {
    FILE * fin = fopen(data_file.c_str(), "r");
    if (!fin) {
        printf("Cannot open file %s to write!\n", data_file.c_str());
        return 1;
    }

    char buff1[BUFF_SIZE_SHORT];
    char buff2[BUFF_SIZE_LONG];
    string line;

    fgets(buff1, BUFF_SIZE_SHORT - 1, fin);
    auto num_samples = atol(buff1);

    for (int i = 0; i < num_samples; ++i) {
        fgets(buff2, BUFF_SIZE_LONG - 1, fin);
        line = buff2;
        auto * pxt = new dataset(line);
        fit(pxt);
    }
    fclose(fin);
    return 0;
}

float model::predict_proba(dataset * px) {
    float dot = 0;
    for (int i = 0; i < px->length; ++i) {
        if (px->data.indexs[i] < num_features) {
            dot += px->data.data[i] * w[px->data.indexs[i]];
        }
    }
    return sigmoid(dot + w_intercept);
}

int model::predict(dataset * px, float threshold) {
    float prob = predict_proba(px);
    if (prob < threshold) {
        return 0;
    } else {
        return 1;
    }
}

int model::save_model(string file_path) {
    FILE * fout = fopen(file_path.c_str(), "w");
    if (!fout) {
        printf("Cannot open file %s to write!\n", file_path.c_str());
        return 1;
    }
    fprintf(fout, "%zu\n", num_features);
    fprintf(fout, "%g %g %g %g\n", alpha, beta, lambda1, lambda2);
    fprintf(fout, "%g %g %g\n", z_intercept, n_intercept, w_intercept);
    saveAslibsvm(z, fout);
    saveAslibsvm(n, fout);
    saveAslibsvm(w, fout);

    fclose(fout);
    return 0;
}

int model::load_model(string & file_path) {
    FILE * fin = fopen(file_path.c_str(), "r");
    if (!fin) {
        printf("Cannot open file %s to read!\n", file_path.c_str());
        return 1;
    }
    printf("Load model from %s\n", file_path.c_str());

    char buff1[BUFF_SIZE_SHORT];

    //load num_features
    fgets(buff1, BUFF_SIZE_SHORT - 1, fin);
    num_features = (unsigned long)atol(buff1);

    //load alpha, beta, lambda1, lambda2
    fgets(buff1, BUFF_SIZE_SHORT - 1, fin);
    vector<float> params = readFromSep<float>(buff1, ' ');
    alpha = params[0], beta = params[1], lambda1 = params[2], lambda2 = params[3];

    init_model();

    //loda z_intercept, n_intercept, w_intercept
    fgets(buff1, BUFF_SIZE_SHORT - 1, fin);
    vector<float> intercepts = readFromSep<float>(buff1, ' ');
    z_intercept = intercepts[0], n_intercept = intercepts[1], w_intercept = intercepts[2];

    char buff2[BUFF_SIZE_LONG];

    //load z
    fgets(buff2, BUFF_SIZE_LONG - 1, fin);
    z = libsvm2sparse<float>(buff2, num_features);

    //load n
    fgets(buff2, BUFF_SIZE_LONG - 1, fin);
    n = libsvm2sparse<float>(buff2, num_features);

    //load w
    fgets(buff2, BUFF_SIZE_LONG - 1, fin);
    w = libsvm2sparse<float>(buff2, num_features);

    fclose(fin);
    return 0;
}