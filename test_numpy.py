import numpy as np

def foo(arr):
    arr[0] = 33

a = np.array([1,2,3])
foo(a)
print(a[0])