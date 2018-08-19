from libcpp.vector cimport vector
from libcpp.string cimport string
import logging
import numpy as np


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

cdef extern from "../src/dataset.h":
    cdef cppclass ftrl_data[T]:
        ftrl_data();
        vector[unsigned int] indexs;
        vector[T] data;
        
    cdef cppclass dataset:
        dataset(string & dataLine);
        
        unsigned int length;
        ftrl_data[float] data;
        int label;
        
cdef extern from "../src/model.h":
    cdef cppclass model:
        model(string file_path);
        model(unsigned long num_features, float alpha, float beta, float lambda1, float lambda2);
    
        unsigned long num_features;
        float alpha, beta, lambda1, lambda2;
        float w_intercept;
        vector[float] w;
        
        void fit(dataset * pxt);
        float predict_proba(dataset * px);
        int predict(dataset * px, float threshold);
        int load_model(string & file_path);
        int save_model(string file_path);
        int fit(string data_file);
        

cdef class _ftrl_dataset:
    cdef dataset * _thisptr
    
    def __cinit__(self, string dataLine):
        self._thisptr = new dataset(dataLine)
        
        if self._thisptr == NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

cdef class ftrl:
    """
    This class wraps a class implemented by c++.
    FTRL-Proximal is an algorithm for online learning which is quite successful 
    in solving sparse problems. 
    @params:
        - num_features:         unsigned long. number of features.
        - alpha:                float. parameter alpha.
        - beta:                 float. parameter beta.
        - lambda1:              float. parameter lambda1.
        - lambda2:              float. parameter lambda2.
        - file_path:            string. model path.
    
    Please specify at least one of two parameters: num_features or file_path.
    The model first loads from the specified file.
    """
    cdef model * _thisptr
    
    def __cinit__(self, 
                  unsigned long num_features=0, 
                  float alpha=1, 
                  float beta=1, 
                  float lambda1=0, 
                  float lambda2=0,
                  string file_path=""):
        assert((file_path != "" or num_features != 0), "must specify file_path or num_features!")
        if file_path != "":
            self._thisptr = new model(file_path)
            print("load model from file {}".format(file_path))
        else:
            self._thisptr = new model(num_features, alpha, beta, lambda1, lambda2)
        
        if self._thisptr == NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr
    
    @property
    def num_features(self):
        return self._thisptr.num_features
    
    @property
    def alpha(self):
        return self._thisptr.alpha
    
    @property
    def beta(self):
        return self._thisptr.beta
    
    @property
    def lambda1(self):
        return self._thisptr.lambda1
    
    @property
    def lambda2(self):
        return self._thisptr.lambda2
    
    @property
    def coeffs(self):
        coef = {"coef": np.array(self._thisptr.w), "intercept": self._thisptr.w_intercept}
        return coef
    
    
    cpdef void fit(self, string dataLine):
        """
        fit model with one train data.
        @params:
            - dataLine:        string. LIBSVM format data.
        """
        train_data = _ftrl_dataset(dataLine)
        self._thisptr.fit(train_data._thisptr)
        
    cpdef float predict_proba(self, string dataLine):
        """
        predict probability using model.
        @params:
            - dataLine:        string. LIBSVM format data.  Use - to specify unknown label.
        """
        test_data = _ftrl_dataset(dataLine)
        return self._thisptr.predict_proba(test_data._thisptr)
        
    cpdef int predict(self, string dataLine, float threshold=0.5):
        """
        predict label using model.
        @params:
            - dataLine:        string. LIBSVM format data. Use - to specify unknown label.
            - threshold:       float. threshold of probability.
        """
        test_data = _ftrl_dataset(dataLine)
        return self._thisptr.predict(test_data._thisptr, threshold)
    
    cpdef void load_model(self, string file_path) except *:
        """
        load model from specified model file.
        @params:
            - file_path.        string. model path.
        """
        callbacks = self._thisptr.load_model(file_path)
        if callbacks:
            raise IOError("Cannot open file {} to read!".format(file_path))
        else:
            #logging.info("load model from file {}".format(file_path))
            print("load model from file {}.".format(file_path))
        
    cpdef void save_model(self, string file_path) except *:
        """
        save model into specified model file.
        @params:
            - file_path.        string. model path.
        """
        callbacks = self._thisptr.save_model(file_path)
        if callbacks:
            raise IOError("Cannot open file {} to write!".format(file_path))
        else:
            print("save model into file {}.".format(file_path))
            
            
    cpdef void fit_batch(self, string data_file) except *:
        """
        fit model with a batch of train data.
        @params:
            - data_file:        string. LIBSVM format data file. And the first line
                                must be the number of train data.
        """
        callbacks = self._thisptr.fit(data_file)
        if callbacks:
            raise IOError("Cannot open file {} to read!".format(data_file))
        else:
            #print("fit model using data from file {}.".format(data_file))
            pass
    
    
    
    
    
    
    
    
    
    
    
    
    