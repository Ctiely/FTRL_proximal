from distutils.core import setup, Extension
from Cython.Build import cythonize
#import numpy as np

ext = Extension("FTRL",
                sources=["FTRL.pyx", "../src/dataset.cpp", "../src/model.cpp"],
                language="c++")
                #include_dirs=['.', np.get_include()])

setup(name="FTRL",
      ext_modules=cythonize(ext))
