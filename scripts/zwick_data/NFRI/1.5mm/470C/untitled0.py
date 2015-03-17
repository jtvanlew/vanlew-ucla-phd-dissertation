# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 23:02:02 2014

@author: jon
"""

# list of numbers
j=[4,5,6,7,1,3,7,5]
#list comprehension of values of j > 5
x = [i for i in j if i>5]
#value of x
print len(x)

#or function version
def length_of_list(list_of_numbers, number):
     x = [i for i in list_of_numbers if j > number]
     return len(x)
length_of_list(j, 5)