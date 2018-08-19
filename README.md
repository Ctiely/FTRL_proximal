# FTRL_proximal
FTRL-Proximal is an algorithm for online learning which is quite successful in solving sparse problems. 

## introduction
The implementation is based on the algorithm from the ["Ad Click Prediction: a View from the Trenches"](https://www.eecs.tufts.edu/%7Edsculley/papers/ad-click-prediction.pdf) paper.

## install:
```
git clone https://github.com/Ctiely/FTRL_proximal
cd FTRL_proximal/python
make clean
make
```
In Mac OSX, FTRL.so will appear in the directory FTRL_proximal.
Copy this .so file into your work path, and just import FTRL.

## usage:
```model = FTRL.ftrl(num_features, alpha, beta, lambda1, lambda2)```

or

```model = FTRL.ftrl(file_path="path/to/model.ftrl")```

Please refer to 
[```python/test_ftrl.ipynb```](https://github.com/Ctiely/FTRL_proximal/blob/master/python/test_ftrl.ipynb)
or 
[```python/test_ftrl.py```](https://github.com/Ctiely/FTRL_proximal/blob/master/python/test_ftrl.py) for specific usage.
