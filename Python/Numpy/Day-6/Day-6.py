# -*- coding: utf-8 -*-
"""

I didnt use print statment because I wanted to execute line by line to understand 
how numpy is working

@author: muthukumar_ram
"""
import numpy as np
from matplotlib import pyplot as plt
# Data Processing Using Arrays
points = np.arange(-5, 5, 0.01) # Creating 1000 values

"""
np.meshgrid function takes two 1D arrays and produces
two 2D matrices corresponding to all pairs of (x, y) in the two arrays
"""
xs,ys = np.meshgrid(points,points)
ys.shape # (1000,1000)
z = np.sqrt(xs**2 + ys**2)
z.shape #(1000,1000)
plt.title("Plot of $\sqrt{x^2 + y^2}$ for grid values")
plt.imshow(z)
plt.colorbar()

"""
np.where is like where condn in sql
a if a else y like that
"""
# without np.where
a1 = [1,2,3,4,5]
a2 = [6,7,8,9,0]
cond = [True,False,False,True,False]
result = [(aa1 if cond1 else aa2) for aa1,aa2,cond1 in zip(a1,a2,cond)]
print(result)
# [1, 7, 8, 4, 0]

# using np.where

result = np.where(cond,a1,a2)
print(result)
# [1 7 8 4 0]

a3 = np.random.rand(4,4)
res = np.where(a3 > 0.5)
"""
a3:
array([[ 0.53798596,  0.33860222,  0.11750492,  0.93202907],
       [ 0.70524456,  0.30438786,  0.64601622,  0.72010387],
       [ 0.32738324,  0.6202863 ,  0.4916738 ,  0.76060135],
       [ 0.34414069,  0.3231245 ,  0.79714482,  0.46385536]])
res:
(array([0, 0, 1, 1, 1, 2, 2, 3], dtype=int64),
 array([0, 3, 0, 2, 3, 1, 3, 2], dtype=int64))
"""
res = np.where(a3 > 0.5,2, -2) # broadcasting the values applying 2 for True -2 for False
"""
array([[ 2, -2, -2,  2],
       [ 2, -2,  2,  2],
       [-2,  2, -2,  2],
       [-2, -2,  2, -2]])
"""
res = np.where(a3 > 0.5,2,a3) # broadcasting only positive values

# np.sum, cumsum, cumprod

a = np.array([[1,2,3], [4,5,6]])
"""
You can pass index as argument for the below functions
"""
a.sum() #21

a.cumsum() #array([ 1,  3,  6, 10, 15, 21], dtype=int32)
#produces intermediate values [1+2=3 3+3=6]
a.cumprod() # Out[28]: array([  1,   2,   6,  24, 120, 720], dtype=int32)

# sorting
a.sort()

#Unique
names = np.array([1,2,3,4,1,3,5])
np.unique(names)
# array([1, 2, 3, 4, 5])

"""
unique(x)           Compute the sorted, unique elements in x
intersect1d(x, y)   Compute the sorted, common elements in x and y
union1d(x, y)       Compute the sorted union of elements
in1d(x, y)          Compute a boolean array indicating whether each element of x is contained in y
setdiff1d(x, y)     Set difference, elements in x that are not in y
setxor1d(x, y)      Set symmetric differences; elements that are in either of the arrays, but not both
"""

