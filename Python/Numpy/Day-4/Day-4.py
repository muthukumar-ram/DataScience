# -*- coding: utf-8 -*-
"""


@author: muthukumar-ram

I didnt use print statment because I wanted to execute line by line to understand 
how numpy is working
"""

"""
Multi dimensional Indexing
"""
import numpy as np
a1 = np.arange(1,10)
a1= a1.reshape(3,3)
a1[:2,1]
"""
array([[1, 2, 3],
       [4, 5, 6],   ==> array([2,5]), :2 represents 0,1 rows and 1 represents col 1 alone, so we got [2,5]
       [7, 8, 9]])
Stride:
if we have more elements, then we can use skip parameter like a1[::2,:10:3] all rows skip 2 and till 10th column and skip 3

"""
# Boolean Indexing
age = np.array([15,18,28,45,10,22,32,35,24])
randArr = np.arange(1,10)
age > 25 # this will give boolean array

# array([False, False,  True,  True, False, False,  True,  True, False], dtype=bool)

randArr [ age > 25] # we can apply column,row index as well randArr[age > 25,::2]
# array([3, 4, 7, 8])
 
ageFilter = (age > 25) & (age < 35)
randArr[ageFilter]
# array([3, 7])
# we can assign values as well
randArr[ageFilter] = 99

"""
randArr
Out[112]: array([ 1,  2, 99,  4,  5,  6, 99,  8,  9])
"""

arr1 = np.empty((4, 4))
for i in range(4):
    arr1[i]= i # bulk assigning
arr1
"""
array([[ 0.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  1.],
       [ 2.,  2.,  2.,  2.],
       [ 3.,  3.,  3.,  3.]])
"""
# Matrix Computation
a = np.arange(9)
a = a.reshape(3,3)
b = np.arange(9,18)
b = b.reshape(3,3)
a+b # we can apply -,*,/
a*b

# For dot product
np.dot(a,b) # a.dot(b)

"""
array([[ 42,  45,  48],
       [150, 162, 174],
       [258, 279, 300]])
"""
"""
Some math functions
"""
np.sqrt(b)
np.exp(b)
np.mean(b)
np.std(b,ddof=0) # degree of freedom
np.min(b) 
# 9 which is min value in the array
np.min(b,axis=1) 
# array([ 9, 12, 15]) min value in horizontally
np.min(b,axis=0)
# array([ 9, 10, 11]) min value in each vertically
np.max(b) # similar to min
np.maximum(a,b) # maximum by each element
"""
fmin,fmax will ignore the NaN
"""
