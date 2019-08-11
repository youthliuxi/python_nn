import numpy as np
np.random.seed(123)
print(np.random.rand())
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
b = a.reshape([2,6])
print(a)
print(b)
c = b.reshape([3,4])
print(c)