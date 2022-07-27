'''
Last amended: 27th July, 2022
		--Ramnavmi--
Myfolder: C:\Users\ashok\OneDrive\Documents\python
	  /home/ashok/Documents/1.basic_lessons

Examples of python usage:
    	(https://www.python.org/about/apps/)
    	i)   Web and Internet Development
    	ii)  Scientific, Engineering and Numeric
    	iii) Education (for teaching programming)
    	iv)  Desktop GUIs (widgets, gtk+)
    	v)   Software Development (database related applications)
    	vi)  Business Applications (ERP/e-commerce systems)


Objectives:
	i)   Learn python data structures
	ii)  Opening and reading file
	iii) Iterating over sequences with for-loop
	iv)  List/dictionary comprehension
	v)   Writing functions in python
	vi)  Writing classes in python



References:
	https://docs.python.org/2/tutorial/introduction.html#unicode-strings
       	http://www.datacarpentry.org/python-ecology-lesson/
       	http://www.datacarpentry.org/python-ecology-lesson/00-short-introduction-to-Python
       	http://www.datacarpentry.org/python-ecology-lesson/00-short-introduction-to-Python
'''

# 1.0 Variable types in python need not be
#     declared beforehand just as in R
#     python is dynamically-typed language

x=5                   # Type of x is integer
x="abc"               # type of x changed but python makes no complaints

# A Python Integer Is More Than Just an Integer
#  A Python integer is a pointer to a position in
#    memory containing all the Python object information,
#     including the bytes that contain the integer value.
#  https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html
x = 10
x.<PressTab>               # This shows that 'x' is not merely
x.numerator                #  a raw-integer but stores much more information
x.denominator  		       #   like an object


type(x)               	   # What is the type of x


# 1.1 Operators
#  +, -, /, *, %

"abc" + "cde"    # Concatenation of strings
2 + "abc"        # Error
41 % 5           # Modulo: Gives remainder
2 ** 5           # 32; Power operator

# 1.2 Comparison operators
#     <, >, ==, !=, <=, >=

5 > 3
5 == 5
5 == "abc"
"abc" == "abc"
"abc" > "cde"
"abc" >= "abc"

# 1.3 Logic Operators
#     and , or, not

not(True)              # False
True and False         # False
False or not(False)    # True


########### Sequence Types #####################
##### 2. Lists, Tuples & range ############
#  sequence is the generic term for an ordered set.
# Some sequence classes in python are: lists, tuples, strings
#   and range objects

 '''
Lists
=====
Python knows a number of compound data types, used to group together
other values. The most versatile is the list, which can be written
as a list of comma-separated values (items) between square brackets.
Lists might contain items of different types, but usually the items
all have the same type.

https://docs.python.org/3/tutorial/datastructures.html

'''


# L1.0 Lists
numbers=[1,2,5]
numbers[2]               # Indexing starts from 0
numbers[2] = 89          # List value is reassigned


# L1.1 Useless list
ul = [ 1, "abc", 23.34, 2, "abc"]
type(ul)



# L1.2 List of methods that apply to list
dir(numbers)	           # numbers.__sizeof__()

numbers.__add__(numbers)   # Used by extend() method
numbers.extend(numbers)    #  internally


# L1.3 Append a number or list to list
#      Append acts to create 'stacks' of lists
#      Append can append an integer or a list
ex=[2,9]
numbers.append(ex)         # Append a list
numbers.append(10)        # Append an integer
numbers

# L1.3.1 Append will append any object
#		to list

# L1.3.1.1 Append a dictionary
sk = {"k" : 4}
numbers.append(sk)
numbers

# L1.3.1.2 Append a function:

def abc(d):
	return d * d

numbers.append(abc)
numbers

# L1.4 Note the difference between
#         append() and extend():

ex = [2,9]
numbers = [6,7,8]
numbers.append(ex)
numbers


# L1.4.1  'extend' merges a list with a list
#        extend() cannot merge an integer with a list
numbers= [6,7,8]
ex = [2,9]
numbers = [6,7,8]
numbers.extend(ex)
numbers

# L1.4.2 This fails
numbers.extend(10)

# L1.4.3 This succeeds
numbers.extend([10])

# L1.4.4 This also succeeds:
#       As number appends any object
#       and 10 is an object:

numbers.append(10)
numbers


# L2.0 Popping out from list. Last in first out

numbers.pop()
numbers


# L2.1 Deleting any list item in between:

gef = list(range(3,21,2))
gef

# L2.2 What is the index of value 13
gef.index(13)
gef.pop(5)
gef

# L2.2.1  del vs pop

del(gef[5])     # This also works
gef.pop(2:4)    # This does not work
del(gef[2:4])   # This works
gef.pop()       # This works but
                #   there is no equivalence with del

# L2.2.2  Using .remove() without getting its index:

gef = [2,13,2,3,13]
gef
gef.remove(13)
gef


# L3.0 Iterate over contents of list
#      For iterator vs iterable
#      Ref: https://stackoverflow.com/questions/9884132/what-exactly-are-iterator-iterable-and-iteration

for num in numbers:             # s =numbers__iter__() ; list(s)
	print(num)


# L3.1 Or as here
result = 0
for i in range(100):           # range(5) is an iterable just as list is
    result += i

result


# L3.2 Single line for loop
#      Enclosed in square brackets
#       chr(97) is 'a'
#     List comprehension

squares = []
for x in range(10, 20):
    squares.append(x**2)    # squares.extend(x **2) will not work
	                        #  as extend can only extend a list with a list
							#   x**2 is an integer-object
							#     This will work: squares.extend([x**2])

squares

# L3.3  List comprehension
squares = [x * x for x in range(10, 20)]
squares


# L3.4
k = [1,2,3,4,4]
squares = [(i,x * x) for i,x in enumerate(k)]
squares


squares = [{i,x * x} for i,x in enumerate(k)]
squares

# L3.5
x=[chr(i) for i in range(97,100) ]
x


# L3.6  Heterogeneous lists
#        But this flexibility comes at a cost: to allow
#         these flexible types, each item in the list must
#          contain its own type info, reference count, and other
#            information–that is, each item is a complete Python object.
 #            It can be much more efficient to store data in a fixed-type array.
 #             For example np.array()
 # https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html

L3=[True, "abc", 1, 2.9]
type(L3)
for item in L3:
    print(type(item))


# L3.7 Delete a list
n = [1,2,3]
del n[:]  	   # Delete only list-elements
n              # n is an empty list
del n          # n is deleted
n


# L4. Mutating lists
#    What works and what fails
l = list(range(10))
l
l[3] = 10000       # This works
l[2:4] = 10000     # This fails
l[2:4] = [10000] * len(l[2:4])    # This is OK


####
# L5. Functions operating on lists
####
# enumerate():   returns an enumerate object for the list.
#                It gives the list items with their indices.
                 list(enumerate([2,10,300]))
# len()          returns the length of the list.
# filter()      filter() takes a function and filters
#               a sequence by checking each item.
    list(filter(lambda val:val%2==0, [1,2,3,4,5,6,7,8,9]))
# all()         Returns True if all elements in the list are True
#                or if the list is empty
                all([1,2,3,False])
# any()         returns True if at least one element in the list is True
                any([1,2,3,False])
# max(), min(), sorted()                




'''
Tuples
======
A tuple is similar to a list in that it's an ordered
sequence of elements. However, tuples can not be
changed once created (they are "immutable").
Tuples are created by placing comma-separated
values inside parentheses ().
'''


# t1
t=(3,4,6,8)
t[1]
t[1] = 6	# Gives error


# t1.1
a = 1,7     # This is also a tuple
type(a)



# t1.2 zipping tuples

t1 = (1,4,5,6)
t2=("a","b","c","d")
zip(t1,t2)
list(zip(t1,t2))
tuple(zip(t1,t2))


for idx, x in zip(t1,t2):
    print(idx)
    print(x)


# t1.3 enumerate returns position of an element and also element
for idx, x in enumerate(zip(t1,t2)):
	print(idx)
	print(x)



####
# t2. Functions operating on tuples
####
# enumerate():   returns an enumerate object for the tuple.
#                It gives the list items with their indices.
                 tuple(enumerate((2,10,300)))
# len()          returns the length of the tuple.
# filter()      filter() takes a function and filters
#               a sequence by checking each item.
    list(filter(lambda val:val%2==0, (1,2,3,4,5,6,7,8,9)))
# all()         Returns True if all elements in the tuple are True
#                or if the tuple is empty
                all((1,2,3,False))
# any()         returns True if at least one element in the tuple is True
                any((1,2,3,False))
# max(), min(), sorted()



'''
A dictionary is a container that holds unordered pairs
of objects - keys and values.
'''

# 6
d = { 'a' : 9, 'b' : 6}
d['a']
d['a'] = 999
d['b']
d['c'] = 9	# Add a key to d


# 6.1
len(d)			# Function on dict
type(d)
d.keys()		# Methods
d.values()


# 6.2
d.items()
[(i,j) for i , j in d.items()]


del d['a']		# Delete ke:value pair of 'a'


# 6.3 In dictionary while value can be changed,
#      keys are immutable objects:
#      Stackoverflow:  Hashable, Immutable
# 			https://stackoverflow.com/questions/2671376/hashable-immutable
#      Stackoverflow: What does hashable mean in python?
#			https://stackoverflow.com/questions/14535730/what-does-hashable-mean-in-python

d = {(1,2,3) : 56, 'ab' : 78 }	# Correct. Both keys are immutable
d[(1,2,3)]   # gives 56
d = {[1,2,3] : 56, 'ab' : 78 }	# Incorrect. One key is mutable

# 6.4 Why should keys be mutable?
#      Because internally, dict, stores values against
#       hash of keys. All immutable objects in python
#        are hashable. That is they have __hash__()
#         method.
(1,2,3).__hash__()    # exists
[1,2,3].__hash__()    # does not


# 6.5 Tuples are faster than lists:
#     https://stackoverflow.com/questions/3340539/why-is-tuple-faster-than-list-in-python

%timeit a = (10,20,30) ; a[1]	# Create and print a tuple multiple times
%timeit a = [10,20,30] ; a[1]	# Create and print a list multiple times


"""
Range
=====
Generate a sequence of numbers
range(start,stop,step)
  step can be negative
"""

## 7. Range types: Occupies much less memory than lists
 #     range type represents an immutable sequence of numbers
import sys

a = range(10)
b = range(10,15)
c = range(10,100000)

sys.getsizeof(a)      # Check memory occupied
a.__sizeof__()		  # 48

sys.getsizeof(c)
c.__sizeof__()        # 48


sys.getsizeof(list(c))	# Includes size of garbage collector
list(c).__sizeof__()



# 7.1
5 in a
20 in a
len(a)

# 7.2 Slicing sequences

d = range(200)
d[9:103:8]            # Output starts from 9th index upto 103 in steps of 8


# 7.3
d = "This is a good sentence. I like it."
d[2:15:3]            # Start from 2nd index

# 7.4
x = list((3,8,10,2,34,100,67,23))
x[1:4:2]         # Note only one number is output
               # The slice from i to j is: i <= k < j.

x[-3]          # Index from end start from 1
               # In R it means except 3rd
               # Here it means 3rd from end



"""
strings
=======
String module provides a number of methods to manipulate
strings.
https://www.tutorialspoint.com/python/python_strings.htm
https://docs.python.org/3/library/stdtypes.html#typesseq
"""


"""
Operations on string:
	Is one string in another string
	Concatenate two strings
	Multiply a string by 2
	Is there an alphanumeric chanracter in string?
	Capitalize a string
	Split a string
	Replace an element in string with another
	Is there a digit in string?
	Access string in steps of 2

"""

## 8. String sequence type is immutable
s = 'xyabcdefxy'
x = 'abcdef'

# 8.1 Operations on strings
'x' in s
'xy' in s
'xa' in s
'fx' in s
'xya' in s
'xyz' not in s
s + x
s * 2
s +=x
s

# 8.2 Methods on strings
#      upper, lower, split, replace, isdigit, isalnum
#      capitalize

s.isalnum()
s.capitalize()

s.split("b")  # Split separator is 'b'
s.replace('xy', 'zz' )
s.replace('xy', 'zz', 1 )  # At most one place
s.upper()
s.upper().lower()
s.isdigit()

h = '123'
h.isdigit()
s.isalnum()
s.capitalize()


# 8.3 Accessing strings
s
s[0]
s[1:3]
s[1:6:2]   # access index 1, 3 and 5
len(s)
s


# 8.4 Strings are immutable
s[0] = '9'    # Returns error


# 8.4 for loop
for i in s:
    print(i)




"""
sets
====
https://docs.python.org/2/library/sets.html
# Sets module provides classes for manipulation of
#  unordered sequence of unique elements
# Common uses include: removing duplicates,
# and set operations such as:
#    intersection, union, difference, and symmetric difference.
"""

# 9.1
s = {"Delhi", "Delhi", "Kanpur", "Kanpur", "ooty", "ooty"}
s
type(s)
len(s)

# 9.2 Set elements are immutable but a set
#     itself can be extended.
s = {1,2,3}
g = {3,4,5}

s |= g   # s = s.union(g)

# 9.3
t = {"Delhi", "Jaipur", "Ahmedabad", "Lucknow", "Chandigarh", "Hissar"}
len(t)
s.union(t)                # All elements
s.intersection(t)         # Common element


# 9.4
s.difference(t)           # Remaining back in s (s-t)
t.difference(s)           # Remaining back in t (t-s)


# 9.5
"Delhi"  in s              # membership test
"Delhi" not in s                          #  Also:   s is t


# 9.6 Subset
{"Delhi", "Jaipur"}.issubset(s)
{"Delhi", "Kanpur"}.issubset(t)


# 9.7 Superset operation
s.issuperset({"Delhi", "Kanpur"})


"""
Functions
=========

Functions vs Methods
A method refers to a function which is part of a class.
You access it with an instance or object of the class.
A function doesn’t have this restriction: it just refers
to a standalone function. This means that all methods
are functions but not all functions are methods.

"""

# 10 Defining function

## 10.1 A python function
#   'a' and 'b' can be any objects
def xx(a,b):
	return a + b

# 10.1.1 Try
xx(3,4)

xx([1,4], [4,6])


# 10.2
def squareof(x):
    return x * x

# 10.2.1 Multi return function
#        % is modulus
def squareof(x,y):
    return x * x, x%y

squareof(2,4)
squareof(y = 4,x = 2)       # Call with keyword arguments in any order



## 10.3 An R function
g = function(x) {
        return (x * x)
        }
`
# 10.4 Square contents of a list
hj=[]
 def sq(xx):
    for i in xx:
        re=i * i
        hj.append(re)

sq(numbers)
hj

# 10.5 Function returns three values
#     Input to function can be list or Series
def myfunction (c,d):
	l=np.mean(c)
	g=np.median(d)
	h=[l,g]
	return h,l,g


c=[1,2,3]
d=[4,5,6]
x,y,x= myfunction(c,d)
x
y
z



## 10.6 A python function
def squareof(x):
    return x * x

def squareof(x,y):
    return x * x, x%y

# 10.6.1 A function returns another function
#         Technique of returning a function from
#         another function is known as currying.
# Ref: https://stackoverflow.com/questions/14261474/how-do-i-write-a-function-that-returns-another-function

#  Calculate volume of cylinder: pi * r^2 * h

def vol_cyl(r):
      def volume(h):
          return 3.14 * r**2 * h
      return volume

# 10.6.2 Use it as:
a = vol_cyl(3)   # r = 3
a(4)		 # h = 2


# 10.7 Variable number of arguments
def var(*myargs):
    for i in myargs:
        print(i)
    return (np.sum(myargs))

var(2,7,8)

# 10.8 Anonymous functions are also called
#      lambda functions in Python because
#      instead of declaring them with the
#      standard def keyword, you use the lambda keyword.
# Ref: https://realpython.com/python-lambda/
# A lambda function can’t contain any python statements (return x)
# A Python lambda function is a single expression

# 10.8.1
myfun = lambda x: x*8  # x is called 'bound variable'
myfun(9)
# OR
(lambda x: x*8)(9)

# 10.8.2
full_name = lambda fName,lName: f'{fName.title()} {lName.title()}'
full_name('ashok','harnal')

# 10.8.3
def sqr(x):
	return x * x

myfunc = lambda x, sqr: sqr(x)
myfunc(20,sqr)

# 10.8.4
(lambda x, y, z: x + y + z)(1, 2, 3)
(lambda x, y, z=3: x + y + z)(1, 2)
(lambda x, y, z=3: x + y + z)(1, y=2)
(lambda *args: sum(args))(1,2,3)
(lambda **kwargs: sum(kwargs.values()))(one=1, two=2, three=3)



########################
# 11. Some common operations on sequence types

x in s             # True if an item of s is equal to x, else False 	(1)
x not in s 	     # False if an item of s is equal to x, else True 	(1)
s + t 	            # the concatenation of s and t 	(6)(7)
s * n or n * s 	  # equivalent to adding s to itself n times 	(2)(7)
s[i] 	            # ith item of s, origin 0 	(3)
s[i:j] 	         # slice of s from i to j 	(3)(4)
s[i:j:k] 	         # slice of s from i to j with step k 	(3)(5)
len(s) 	         # length of s
min(s) 	         # smallest item of s
max(s) 	         # largest item of s
s.index(x[, i[, j]]) 	# index of the first occurrence of x in s (at or after index i and before index j) 	(8)
s.count(x) 	          # total number of occurrences of x in s



#12 Classes in python

class abc:
    a = 3
    b= 5
    def omg(self):
        return(self.a * self.b)


k = abc()
k.a = 5
k.b = k.a
k.omg()


class yo1:
    value = 5
    def __init__(self, value):
        self.value = value

    def fog(self, nv):
        self.value = nv


ok = yo1(200)
ok.value

ok.fog(300)
ok.value


# 1.0
%reset -f       # ipython magic command to clear memory


# 13. OS related operations
import os
os.getcwd()
#os.chdir("c:\\users\\ashok\\Documents")
#os.chdir("/home/ashok/Documents")
os.listdir()  # Directory contents


# 13.1 Joining path and filenames correctly
path = "/home/ashok/Documents"
fname = "readme.txt"
os.path.join("/home/ashok/Documents", fname)


# 13.2 How to read a file
file = os.path.join("/home/ashok/Documents", fname)
s = open(file)
data = s.read()
data
s.close()      # File must be closed to release resources back to system


# 13.3 Look at the file contents
data

########################
# Use of map
# Ref: https://realpython.com/python-map-function/
"""
map
Sometimes you might face situations in which you need
to perform the same operation on all the items of an
input iterable to build a new iterable. The quickest
and most common approach to this problem is to use a
Python for loop. However, you can also tackle this
problem without an explicit loop by using map().
map() takes a function object and an iterable
(or multiple iterables) as arguments and returns an
iterator that yields transformed items on demand.
The function’s signature is defined as follows:

map(function, iterable[, iterable1, iterable2,..., iterableN])

function can be any Python function that takes a
number of arguments equal to the number of iterables
you pass to map().
Note: The first argument to map() is a function object,
      which means that you need to pass a function without
	  calling it. That is, without using a pair of parentheses.

"""
# 14
# 14.1
def square(number):
    return number ** 2
#14.2
numbers = [1, 2, 3, 4, 5]
#14.3
squared = map(square, numbers)
#14.4
list(squared)
#15
list(map(lambda x, y: x - y, [2, 4, 6], [1, 3, 5]))






########## FINISH ################
"""
About python's usage:
    Ref: https://www.python.org/about/

    1. Web and Internet Development

        Frameworks such as Django and Pyramid.
        Micro-frameworks such as Flask and Bottle.
        Advanced content management systems such as Plone and django CMS.

    2. Database Access
    3. Desktop GUIs
    4. Scientific & Numeric

        SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of
        open-source software for mathematics, science, and engineering.
        In particular, its core packages are:
            numpy: An N-dimensional array package provides sophisticated functions
            pandas: high-performance, easy-to-use data structures and data analysis tools
            ipython:
            matplotlib

        scikit-learn Machine Learning in Python
           http://scikit-learn.org/stable/index.html

    5. Education
    6. Network Programming
    7. Software & Game Development
"""
