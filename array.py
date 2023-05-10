"""
Python implementation of a dynamic array. 

Functionality is intended to be similar to arraylist.cpp in the C++ folder
"""
import ctypes
from time import time
import sys

class DynamicArrays:
    """A dynamic array akin to a simplified python list"""
    def __init__(self):
        """Initializes a dynamic array"""
        self._n = 0 # number of elements in the list
        self._capacity = 1 # number of elements that can actually be stored
        self._A = self._make_array(self._capacity) # the storage

    def __len__(self):
        """Returns the number of elements in the array"""
        return self._n

    def is_empty(self):
        """Returns True if array is empty"""
        return self._n == 0

    def is_full(self):
        """Returns True if array is full"""
        return self._n == self._capacity

    def __getitem__(self, index):
        """Returns the element at index k"""
        if not 0 <= index < self._n:
            raise IndexError("Index is out of range")
        return self._A[index]
    
    def append(self, obj):
        """Adds an obj to the back of an array"""
        if self._n == self._capacity: # array is full
            self._resize(2*self._capacity)
        self._A[self._n] = obj
        self._n += 1

    def insert(self, index, obj):
        """Adds an obj to list at a particular index provided"""
        if self._n == self._capacity:
            self._resize(2 * self._capacity)

        for i in range(self._n, index, -1): # shift to the right
            self._A[i] = self._A[i - 1]
        self._A[index] = obj
        self._n += 1
    
    def _resize(self, c):
        """Non-public utility to resize the array"""
        B = self._make_array(c) # make a bigger array
        for i in range(self._n):
            B[i] = self._A[i]
        self._A = B
        self._capacity = c

    def count(self, obj):
        """Return the number of occurences of obj in array"""
        count = 0

        for i in range(self._n):
            if self._A[i] == obj:
                count += 1
        return count

    def __contains__(self, obj):
        """Loops through the array to find if obj in the array"""
        for i in range(self._n):
            if self._A[i] == obj:
                return True
        return False

    def __setitem__(self, index, obj):
        """Replace obj at a particular position in the array with obj"""
        if not 0 <= index < self._n:
            raise IndexError("Index is out of range.")
        self._A[index] = obj

    def remove(self, obj):
        """Removes the first occurence of obj in the array.
        
        Raise ValueError if obj is not in the array
        """
        for i in range(self._n):
            if self._A[i] == obj: # match found
                for j in range(i, self._n - 1):
                    self._A[j] = self._A[j + 1] # shift others to fill gap
                self._A[self._n - 1] = None # convention to deprecate
                self._n -= 1
                return
        raise ValueError("Value not found in array") # only reached if match is not found

    def pop(self, index=None):
        """Removes and returns an element in the array.
        
        If the user provides no argument - the default state, the function removes and returns the last
        element in the array. 
        """
        if index is None:
            old_val = self._A[self._n - 1]
            self._A[self._n - 1] = None # deprecate
            self._n -= 1
            return old_val
        else:
            if not 0 <= index < self._n:
                raise IndexError("Index out of range")
            old_val = self._A[index]
            for i in range(index, self._n - 1):
                self._A[i] = self._A[i + 1]
            self._A[self._n - 1] = None
            self._n -= 1
            return old_val

    def insertVal(self, obj):
        """Adds obj to the array if it doesn't exist already"""
        for i in range(self._n):
            if self._A[i] == obj:
                return
        if self.is_full():
            self._resize(2 * self._capacity)
        self._A[self._n] = obj
        self._n += 1
    
    def _make_array(self, c):
        """Return new array with capacity c"""
        return (c * ctypes.py_object)()


def compute_average(n):
    """Perform n number of appends to a python list so we can experimentally
    compute the average time elapsed for the operation"""
    data = []
    start = time()

    for _ in range(n):
        data.append(None) # we don't care about the content

    end = time()

    return (end-start)/n

if __name__ == '__main__':
    a = DynamicArrays()
    print(len(a))
    a.append(2)
    a.append(3)
    print(len(a))
    a.append(4)
    a.insert(2, 2)
    a[0] = 17
    a.append(5)
    print(a[2])
    print(a[0])
    print(a.count(2))
    print(a.pop())
    print(len(a))
    a.insertVal(3)
    print(len(a))
    a.insertVal(3)
    print(len(a))

    time = compute_average(1000)
    print(time)
