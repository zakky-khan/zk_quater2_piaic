#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[7]:


import numpy as np
oneD_arr=np.arange(0,10)
twoD_aar=np.reshape(oneD_arr,(2,5))
twoD_aar


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[20]:


arr1=np.arange(10).reshape(2,5)
arr2=np.ones(10).reshape(2,5)
stack_vertically=np.vstack((arr1,arr2))
stack_vertically



# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[23]:


arr1=np.arange(10).reshape(2,5)
arr2=np.ones(10, dtype=int).reshape(2,5)
stack_horizontal=np.hstack((arr1,arr2))
stack_horizontal


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[28]:


arrays=np.array([[0, 1, 2, 3, 4],[5,6, 7, 8, 9]])
flat_arr=arrays.flatten()
flat_arr


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[8]:


import numpy as np
higherDimension=np.arange(15).reshape(1,3,5)
higherDimension
higherDimension=higherDimension.flatten()
higherDimension


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[14]:


oneDimension=np.arange(15)
oneDimension=np.arange(15).reshape(-1,3)
oneDimension


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[17]:


arr=np.arange(25).reshape(5,5)
arr
squareOfArray=np.square(arr)
squareOfArray


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[25]:


arr=np.random.randint(30, size=(5,6))
meanOfArray=arr.mean()
meanOfArray


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[27]:


arr=np.random.randint(30, size=(5,6))
standardDeviation=np.std(arr)
standardDeviation


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[28]:


arr=np.random.randint(30, size=(5,6))
Median=np.median(arr)
Median


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[31]:


arr=np.random.randint(30, size=(5,6))
transpose=arr.T
transpose


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[37]:


arr=np.arange(16).reshape(4,4)
SumOfDigonal = np.trace(arr)
SumOfDigonal


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[3]:


import numpy as np
arr=np.arange(16).reshape(4,4)
determinant=np.linalg.det(arr)
determinant


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[22]:


arr=np.arange(6)
print(arr)
print(np.percentile(arr,5))
print(np.percentile(arr,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[33]:


arr =np.arange(10).reshape(2,5)
arr
DoContain_nullValue=np.isnan(arr)
DoContain_nullValue


# In[ ]:




