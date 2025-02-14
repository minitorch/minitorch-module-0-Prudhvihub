"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from functools import reduce as reduce_

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(x,y):
    return x*y  

# - id
def id(x):
    return x

# - add
def add(x,y):
    return x+y
# - neg
def neg(x):
    return -x
# - lt
def lt(x,y):
    return float(x<y)
# - eq
def eq(x,y):
    return float(x==y)
# - max
def max(x,y):
    return x if x> y else y
# - is_close
def is_close(x, y):
    """
    Check if two values are close within epsilon.
    """
    return float(abs(x - y) < EPS)
# - sigmoid
def sigmoid(x):
    """
    Numerically stable sigmoid function.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        # For negative values, use the identity sigmoid(x) = 1 - sigmoid(-x)
        # This helps avoid numerical overflow
        z = math.exp(x)
        return z / (1.0 + z)
# - relu
def relu(x):
    return x if x>0 else 0
EPS = 1e-2
# - log
def log(x):
    return math.log(x + EPS)
# - exp
def exp(x):
    return math.exp(x)
# - log_back
def log_back(x, d):
    """
    Derivative of log(x) times d.
    """
    return d / (x + EPS)
# - inv
def inv(x):
    return 1.0/x
# - inv_back
def inv_back(x, d):
    """
    Derivative of 1/x times d.
    """
    return -(d / (x * x))
# - relu_back
def relu_back(x,d):
    if x>0:
        return d
    else:
        return 0
# - sigmoid_back
def sigmoid_back(x,d):
    return sigmoid(x)*(1-sigmoid(x))*d
# - exp_back
def exp_back(x,d):
    return math.exp(x)*d
# - max_back
def max_back(x,y,d):
    if x>y:
        return d
    else:
        return 0
# - is_close_back
def is_close_back(x,y,d):
    if is_close(x,y):
        return d
    else:
        return 0
# - log_back
def log_back(x,d):
    return d/x              
# For sigmoid calculate as: 

# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
def map(fn):
    def map_fn(input_list):
        return [fn(x) for x in input_list]
    return map_fn

def zipWith(fn):
    def zipWith_fn(input_list1,input_list2):
        return [fn(x,y) for x,y in zip(input_list1,input_list2)]
    return zipWith_fn

def reduce(fn, initial):
    def reduce_fn(input_list):
        return reduce_(fn, input_list, initial)
    return reduce_fn

def negList(input_list):
    return map(neg)(input_list)

def addLists(input_list1,input_list2):
    return zipWith(add)(input_list1,input_list2)    

def sum(input_list):
    return reduce(add,0)(input_list)

def prod(input_list):
    return reduce(mul,1)(input_list)    

def count(input_list):
    return reduce(add,0)(input_list)

def mean(input_list):
    return sum(input_list)/count(input_list)

def maxList(input_list):
    return reduce(max,float('-inf'))(input_list)

def minList(input_list):
    return reduce(min,float('inf'))(input_list) 

def sort(input_list):
    return sorted(input_list)

def reverse(input_list):
    return input_list[::-1] 

def any(input_list):
    return reduce(max,0)(input_list)

def all(input_list):
    return reduce(mul,1)(input_list)



# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
