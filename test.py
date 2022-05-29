import numpy as np

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
v4 = np.array([7,8,9])
v5 = np.array([10,11,12])

w = np.array([1/4,1/4,1/4,1/4])
m1 = np.c_[v1,v2,v4,v5]

print(m1)

vres = m1.dot(w)

print(vres)