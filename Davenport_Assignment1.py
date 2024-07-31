#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#Import the data
housing_data_df = pd.read_csv(r"C:\Users\Tahes\OneDrive\Desktop\DSC 320 - Math for Data Science\housing_data.csv")


# In[8]:


housing_data_df #displays the data


# In[4]:


#define a function to calculate root mean square error 1.(a)
def rmse(act, pred):
    error = (1/len(act)) * np.sum(np.square(pred - act))
    root_square_error = np.sqrt(error)
    return root_square_error


# In[5]:


rmse(housing_data_df["sale_price"], housing_data_df["sale_price_pred"]) #1.(b)


# In[6]:


#define a function to calculate mean absolute error 2.(a)
def mae(act, pred):
    mean_absolute_error = (1/len(act)) * np.sum(np.abs(pred - act))
    return mean_absolute_error


# In[7]:


mae(housing_data_df['sale_price'], housing_data_df['sale_price_pred']) #2.(b)


# In[8]:


#import the data
mushroom_data_df = pd.read_csv(r"C:\Users\Tahes\OneDrive\Desktop\DSC 320 - Math for Data Science\mushroom_data.csv")


# In[9]:


mushroom_data_df #displays the data


# In[10]:


#define a function to calculate accuracy 3.(a)
def accuracy(act, pred):
    size_of_act = len(act)
    count = 0
    for i in range(0, size_of_act):
        if act[i] == pred[i]:
            count = count + 1
    percentage = (count/size_of_act)*100
    return percentage


# In[11]:


accuracy(mushroom_data_df['actual'], mushroom_data_df['predicted']) #3.(b)


# In[12]:


#define a function based on the given f(P) in 4.
def f_function(p):
    value = 2028
    value = value - (1499.7*p)
    value = value + (449.14*pow(p,2))
    value = value - (69.919*pow(p,3))
    value = value + (5.998*pow(p,4))
    value = value - (0.27*pow(p,5))
    value = value + 0.005*pow(p,6)
    return value    


# In[13]:


#create a list (array) of x-values
x_points = np.array([4, 5, 6, 7, 8, 9, 10, 11])


# In[28]:


#plot the function f(p)
y_points = []
n = len(x_points)
for i in range(0,n):
    value = f_function(x_points[i])
    y_points.append(value)
    
plt.plot(x_points, y_points)
plt.xlabel("p") #creates the x-axis label
plt.ylabel("y-values") #creates the y-axis label
plt.title("Plot of function f of p") #creates a plot title
plt.show() #display the graph of f(p)


# In[15]:


print(y_points)


# (b) p = 5
# (c) minimum error = 1.25

# In[16]:


#import scikit-learn
from sklearn.linear_model import LinearRegression


# In[18]:


#import data
car_data_df = pd.read_csv(r"C:\Users\Tahes\OneDrive\Desktop\DSC 320 - Math for Data Science\car_data.csv")
car_data_df #displays the data


# In[29]:


#make a scatterplot
plt.scatter(car_data_df["weight"], car_data_df["hwy_mpg"])
plt.xlabel("weight") #creates the x-axis label
plt.ylabel("hwy mpg") #creates the y-axis label
plt.title("Scatterplot of highway mpg vs. weight") #creates a plot title
plt.show() #displays the graph


# In[21]:


#create linear regression object
lregression = LinearRegression()
lregression.fit(car_data_df[["weight"]],car_data_df["hwy_mpg"])


# In[32]:


#create a plot
x_points = np.array(range(1500, 5000, 500))
y_points = lregression.predict(x_points.reshape(-1,1))


# In[33]:


plt.scatter(car_data_df["weight"], car_data_df["hwy_mpg"])
plt.plot(x_points, y_points, c="red")

plt.xlabel("weight") #creates the x-axis label
plt.ylabel("hwy mpg") #creates the y-axis label
plt.show()#display the graph with the best fit line


# 2. Based on the plot, the trend shows that the high miles per gallon decreases as the weight increases.
# 3. For a decreasing function, the slope is negative since the output values decrease as the input values increase.
# 4. Because the slope, m, is negative and -0.05 < 0 it still means the function is decreasing.

# In[ ]:


#define a function to calculate root mean square error 7.
def rmse(act, pred):
    error = (1/len(act)) * np.sum(np.square(pred - act))
    root_square_error = np.sqrt(error)
    return root_square_error

