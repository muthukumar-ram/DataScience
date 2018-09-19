# -*- coding: utf-8 -*-
"""


@author: muthukumar-ram

I didnt use print statment because I wanted to execute line by line to understand 
how numpy is working

Day 3
9. Copies and Views
"""
import numpy as np

c1 = np.arange(27).reshape(3, 3, 3)
cCopy = c1.copy()
cView = c1.view()
cCopy[1][1][1] # which gives 13 single no
cCopy[1][1][1] = 99
cView[1][1][1] 

c2 = c1
c2[1][1][1] = 99 # this will change the value both in C2 and C1. So be careful when using assignment

cCopy[0,1,0] #this will give 3  which is 0th array 1st row 0th column, coz this is 3D
"""
Some missed np array operations
"""
# array with zeros
zeros = np.zeros((3,3)) # creates 3 by 3 matrix with zeros
zeros1 = np.zeros_like(c1) # creates c1 like matrix with zeros

# creating Identical matrix
eye = np.eye(3)

# matrix inverse
a = np.array([[1,2,3],[0,2,4],[2,5,3]]) 
np.linalg.inv(a)

"""
Out[57]: 
array([[ 1.4, -0.9, -0.2],
       [-0.8,  0.3,  0.4],
       [ 0.4,  0.1, -0.2]])
"""
# creating random array
rand = np.random.random(size=2)
rand1 = np.random.randint(low=1,high=5,size=4) # we can set boundry explore few other random functions as well


