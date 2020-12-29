#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[4]:


import numpy as np


# 2. Create a null vector of size 10 

# In[10]:


null_vector=np.zeros(10)
null_vector


# 3. Create a vector with values ranging from 10 to 49

# In[8]:


vector_row=np.arange(10,50)
vector_row


# 4. Find the shape of previous array in question 3

# In[13]:


vector_row.shape


# 5. Print the type of the previous array in question 3

# In[14]:


vector_row.dtype


# 6. Print the numpy version and the configuration
# 

# In[16]:


import numpy as np
print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[17]:


vector_row.ndim


# 8. Create a boolean array with all the True values

# In[22]:


boolean_array=np.ones(5, dtype=bool)
boolean_array


# 9. Create a two dimensional array
# 
# 
# 

# In[23]:


array_row_column = np.arange(10).reshape(2,5)
array_row_column


# 10. Create a three dimensional array
# 
# 

# In[25]:


array_x_y_z = np.arange(20).reshape(2,5,2)
array_x_y_z


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[34]:


forward_vector=np.arange(5)
reverse_vector=forward_vector[::-1]
reverse_vector


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[38]:


null_vector2=np.zeros(10)
null_vector2[5]=1
null_vector2


# 13. Create a 3x3 identity matrix

# In[41]:


array_2D=np.identity(3)
array_2D


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[44]:


arr = np.array([1, 2, 3, 4, 5])
array_float=(arr.astype(float))
array_float


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[50]:


arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr3 = arr1*arr2
arr3


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[5]:


import numpy as np
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
comaring_arr=arr1<arr2
comaring_arr


# 17. Extract all odd numbers from arr with values(0-9)

# In[19]:


arr_OddZeroToNine=np.arange(1,11,2)
arr_OddZeroToNine


# 18. Replace all odd numbers to -1 from previous array

# In[29]:


arry_odd=np.array([1,3,5,7,9])
arry_odd[arry_odd<11]=-1
arry_odd


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[15]:


import numpy as np
arr = np.arange(10)
np.where((arr>4)&(arr<9),12,arr)


# 20. Create a 2d array with 1 on the border and 0 inside

# In[18]:


TwoD_arr = np.array([[1, 0, 0], [0, 0, 1]], np.int32)
TwoD_arr


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[21]:


arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
arr2d[arr2d ==5] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[24]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[arr3d<4] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[12]:


import numpy as np
TwoD_arr=np.arange(10).reshape(2,5)
TwoD_arr
print(TwoD_arr[0:1])


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[29]:


TwoD_arr=np.arange(10).reshape(5,2)
TwoD_arr
print(TwoD_arr[1,1:3])


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[32]:


TwoD_arr=np.arange(10).reshape(2,5)
TwoD_arr
print(TwoD_arr[0:,3:4])


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[8]:


import numpy as np
Arr = np.random.randint(5, size=(10,10))
Arr
max_element = np.max(Arr) 
min_element = np.min(Arr) 
max_element
min_element 


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[12]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
c=np.intersect1d(a,b)
c


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[16]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
c = np.in1d(a,b)
c


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[18]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names != "Will"]


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[24]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(data[names != "Will"])
print(data[names != "Joe"])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[3]:


import numpy as np
TwoD_arr=np.arange(1,16).reshape(5,3)
TwoD_arr


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[5]:


ThreeD_arr=np.arange(1,17).reshape(2,2,4)
ThreeD_arr


# 33. Swap axes of the array you created in Question 32

# In[7]:


ThreeD_arr
np.swapaxes(ThreeD_arr,0,1)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[14]:


arr = np.array([0,1, 2, 3, 4, 5,6,7,8,9])
sq_arr=np.sqrt([0,1, 2, 3, 4, 5,6,7,8,9])
sq_arr
sq_arr[sq_arr < 0.5] = 0
print(sq_arr)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[26]:


random_arr1=np.random.random(12)
random_arr1
random_arr2=np.random.random(12)
random_arr2
maxVal=np.maximum(random_arr1, random_arr2)
maxVal


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[28]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names= set(names)
names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[33]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print(np.intersect1d(a, b))


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[39]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray[:, 1] = [10,10,10]
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[45]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
multpli=np.matmul(x, y)
multpli


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[46]:


rdm_matrix=np.random.rand(3,2)
rdm_matrix


# In[49]:


cum_sum=np.cumsum(rdm_matrix)
cum_sum


# In[ ]:




