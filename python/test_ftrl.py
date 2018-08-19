#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:10:10 2018

@author: clytie
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
os.chdir("../")
import FTRL
os.chdir("python")



model_fit = FTRL.ftrl(123)
with open("../data/Traindata.ftrl", "r") as f:
    line = f.readline()
    while 1:
        line = f.readline()
        if line:
            model_fit.fit(line)
        else:
            print("model fit finished.")
            break
    
model_fit_batch = FTRL.ftrl(123)
model_fit_batch.fit_batch("../data/Traindata.ftrl")

assert(model_fit.coeffs["intercept"] == model_fit.coeffs["intercept"])
assert(np.all(model_fit.coeffs["coef"] == model_fit.coeffs["coef"]))

preds_ftrl = []
labels = []
with open("../data/Testdata.ftrl", "r") as f:
    while 1:
        line = f.readline()
        if line:
            labels.append(int(line[0]))
            pred = model_fit.predict(line)
            pred2 = model_fit_batch.predict(line)
            assert(pred == pred2)
            preds_ftrl.append(pred)
        else:
            break
        
error_rate_ftrl = np.sum(np.array(preds_ftrl) != np.array(labels)) / float(len(preds_ftrl))
print("ftrl model error rate is {:.5f}".format(error_rate_ftrl))


x_train, y_train = load_svmlight_file("../data/data.ftrl", n_features=123)
x_test, y_test = load_svmlight_file("../data/Testdata.ftrl", n_features=123)

lr = LogisticRegression()
lr.fit(x_train, y_train)
preds_sklearn = lr.predict(x_test)

error_rate_sklearn = np.sum(np.array(preds_sklearn) != np.array(labels)) / float(len(preds_sklearn))
print("sklearn Logistic Regression error rate is {:.5f}".format(error_rate_sklearn))

