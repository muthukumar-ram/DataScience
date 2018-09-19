# -*- coding: utf-8 -*-
"""


@author: muthukumar-ram

I didnt use print statment because I wanted to execute line by line to understand 
how numpy is working
"""
"""
Day 1 and Day 2

1. Array creation, ndarry creation
2. datatypes and array types
3. type conversion
"""

import numpy as np

array1D = np.arange(3) # range from 0 to 2
array1D1 = np.array([1.5,2,3,4,5]) # we can call it Vector
print(array1D)
print(array1D1)
print(array1D.dtype) #returns the type of an array int32 for me
print(array1D1.dtype) # returns float64 it will store homogeneous data
array2D = np.array([[1,2],[3,4]])
array2D1 = np.array([np.arange(2),np.arange(2)])
print(array2D)
print(array2D1)
array2D.shape #row col (2,2)

array2D[0,1] #returns 2 0th row and 1st col
array1D2 = np.array([0,1,2,3,4,5,6,7,8,9], dtype='f')
# array([ 1.,  2.,  3.,  4.,  5.,  6.], dtype=float32)

# type conversion, from float to int
array1D2.astype('i')
# array([1, 2, 3, 4, 5, 6], dtype=int32)

"""
4. slicing and indexing
"""

# 1D
array1D2[3:6] # starting position 3 and ends before 6-1 position
#array([ 3.,  4.,  5.], dtype=float32)

array1D2[1:9:2] # 2 is the skip steps
#array([ 1.,  3.,  5.,  7.], dtype=float32)
array1D2[::-1] # reverse the array
#array([ 9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  0.], dtype=float32)

"""
5. Manipulating Array shapes
"""
array1D3 = np.array([0,1,2,3,4,5,6,7,8], dtype='f')
arr1 = array1D3.reshape(3,3) # 3 by 3 matrix
arr1.ravel() # changing to 1 d array as view, it wont chnage the original array
print(arr1.shape)
arr1.flatten() # similar to ravel
arr2 = np.arange(10)
print(arr2.shape)
arr2.resize(2,5)
print(arr2.shape)
print(arr2.transpose()) # in linear algebra transpose will be used often, (5,2)

"""
6. Stacking array
"""

a1 = np.array([1,1,1,1,1,1]).reshape(2,3)
a2 = np.array([0,0,0,0,0,0]).reshape(2,3)
# horizantal stack
np.hstack((a1,a2))  

# or 
np.concatenate((a1,a2), axis = 1)  # axis = 0 means row and 1 means column

# or 
np.column_stack((a1,a2)) # explanation given below

"""
array([[1, 1, 1, 0, 0, 0],
       [1, 1, 1, 0, 0, 0]])
"""
# Vertical Stack
np.vstack((a1,a2))

# or 

np.concatenate((a1,a2),axis = 0) # # axis = 0 means row and 1 means column

# or
np.row_stack((a1,a2))
"""
Out[63]: 
array([[1, 1, 1],
       [1, 1, 1],
       [0, 0, 0],
       [0, 0, 0]])
"""

# Depth Stack
depth = np.dstack((a1,a2))
depth[0]

"""
Out[72]: 
array([[[1, 0],
        [1, 0],
        [1, 0]],

       [[1, 0],
        [1, 0],
        [1, 0]]])
depth[0]
Out[73]: 
array([[1, 0],
       [1, 0],
       [1, 0]])
"""

# Column stacking: The column_stack() function stacks one-dimensional arrays column-wise.
ones = np.arange(2) # [0,1]
twos = 2 * ones # [0,2]
np.column_stack((ones,twos))

"""
array([[0, 0],
       [1, 2]])
"""

# Row stack is just opposite to the column stack and alternative to vstack()

np.row_stack((ones,twos))
"""
Out[77]: 
array([[0, 1],
       [0, 2]])
"""
"""
7. Array Splitting

Arrays can be split vertically, horizontally, or depth-wise. The functions involved are
hsplit(), vsplit(), dsplit(), and split(). We can either split arrays into arrays
of the same shape or indicate the position after which the split should occur.
"""

# Horizontal Splitting

b1 = np.arange(9).reshape(3,3)
b1

sp1 = np.hsplit(b1,3)
sp1[1]

# or

np.split(b1,3,axis = 1)

"""
Out[85]: 
[array([[0],
        [3],
        [6]]), array([[1],
        [4],
        [7]]), array([[2],
        [5],
        [8]])]
sp1[1]
array([[1],
       [4],
       [7]])
"""

# vertical split
sp2 = np.vsplit(b1,3)
# or
np.split(b1,3,axis=0)
# [array([[0, 1, 2]]), array([[3, 4, 5]]), array([[6, 7, 8]])]
sp2[1]
# array([[3, 4, 5]])
"""
Depth-wise splitting: The dsplit() function, splits an array
depth-wise. We will need an array of rank three first:
"""
c1 = np.arange(27).reshape(3, 3, 3)
"""
Out:
array([[[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8]],
[[ 9, 10, 11],
[12, 13, 14],
[15, 16, 17]],
[[18, 19, 20],
[21, 22, 23],
[24, 25, 26]]])
"""
np.dsplit(c1, 3)

"""
8. Some Attributes of Numpy Arrays
"""

b1.size     # no of elements in array
b1.itemsize # size of element
b1.ndim  # no of dimension
b1.nbytes # whole size of array
b1.nbytes == (b1.size * b1.itemsize) # both are same you will get true
b1.T  # Tranpose
#b1.resize(3,2) 

# complex numbers in numpy array, j is the complex number

complexArr = np.array([1.j + 1, 2.j + 3])
# array([ 1.+1.j,  3.+2.j])

complexArr.real
# Out[105]: array([ 1.,  3.])
complexArr.imag
#array([ 1.,  2.])
complexArr.dtype
# Out[107]: dtype('complex128')
"""
flat: This attribute returns a numpy.flatiter object. This is the only way
to acquire a flatiter—we do not have access to a flatiter constructor.
The flat iterator enables us to loop through an array as if it is a flat array.
"""

someArr = np.arange(4).reshape(2,2) # im running out of names
flatObj = someArr.flat
flatObj
for i in flatObj:
    print(i)
"""
0
1
2
3
"""
# we can directly access the np element directly by access because flat make numpy array
# as 1d array
someArr.flat[2] # will give single item
someArr.flat[2] = 10 # we can assign the values like list in python

# converting numpy array to list
complexArr.tolist() # [(1+1j), (3+2j)]

# we cant convert complex to int