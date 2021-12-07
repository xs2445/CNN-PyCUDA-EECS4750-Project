import numpy as np

class Layer(object):
    def __init__(self, size_in, dtype):
        """
        This is the class from which every layer inherit.
        This layer cache the input matrix

        Args:
        - size_in: the size of input data
        - dtype: the data type of the data
        """
        self.size_in = size_in
        self.dtype = dtype
        self.cache_in = np.zeros((self.size_in), dtype=self.dtype)
