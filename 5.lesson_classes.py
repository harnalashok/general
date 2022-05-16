# -*- coding: utf-8 -*-
"""
Last amended: 4th Feb, 2019
My folder: /home/ashok/Documents/1.basic_lessons
Reference: Introduction to Computer Science using Python by Charles Dierbrach
           (Chapter 10)

"""

####################
## A. ENCAPSULATION
####################


# 1.0 Example1
# 1.1 Define a simple class having just one member-Method
#     __init__() constructor is implied
class firstClass():
    """This is very simple class"""
    def lonemember(self):
        x = 45
        return 2 * x


# 1.2 All memebers of a class whether data
#     or functions are public unless hidden
#     Instantiate the class and create object

fo = firstClass()
type(fo)         # firstClass
fo.__doc__
fo.lonemember()    # 90



# 2. Example2:
# Methods with two underscores in the beginning and end
#  are called automatically when a class is instantiated
#   __init__() is a method that  is called when
#    an object of class is created.
#  A member is hidden if its name starts with two-underscores

# 2.1
class SomeClass():
    # 2.2
    def __init__(self):         # Initialize state of object
        self.__n = 0            # hidden member __n
        self.n2 = 0             # public member n2
        g = 30                  # This variable will not be visible outside __init__()
    # 2.3
    def lonemember(self):
        #h = g * 30                  # This statement gives error as 'g' is not known
        #h = n2 * 30                 # this statement also gives error as 'n2' is unknown
        self.__n +=  4               # self._n is visible inside class but not outside it
        return self.n2+self.__n      # this statement is correct as self.n2 exists


# 2.4
obj = SomeClass()
obj.__n                        # Gives error
obj.n2                         # 0
n2                             # gives error
obj.lonemember()               # 4


# 3. Example3:
#  A member is hidden if its name starts with two-underscores

class exampleClass():
    # 3.1
    def __init__(self):       # Initialize state of object
        self.e = 3            # public member 'e'
        self.__f = 4          # hidden member __f
    # 3.2
    def __lonemember(self):       # this method is not visible outside class
        return 4 * self.__f       # this statement is correct as self.__f exists
    # 3.3
    def new(self):                # Only this method is visible outside of class
        d =  5 * self.__lonemember()  # Hidden method visible only inside class
        return  d                     # 4 * 4 * 5


# 3.4
obj = exampleClass()
obj.__lonemember()      # Gives error
obj.new()               # 80
obj.e                   # 1
obj.__f                 # Gives error




# 4. Example4:
# Class members are visible and can be changed
class MyClass():
    # 4.1
    def __init__(self,x,y):
        self.u = x           # u can be changed
        self.t= y
    # 4.2
    def mult(self):
        x1 = self.u
        x2 = self.t
        self.r = self.u
        return x1 * x2


# 4.3
mult()                  # Not visible outside class
obj = MyClass(3,4)
obj.mult()
obj.r
obj.x1

# 4.4
obj1 = MyClass(8,9)
obj1.mult()
obj.u = 80           # u can be changed
obj.mult()


####################
## B. INHERITANCE
# Ref: https://www.python-course.eu/python3_inheritance.php
####################

# 5. Base class or superclass, parent
class person():
    def __init__(self, age, gender, country):
        self.__age = age
        self.__gender = gender
        self.country = country

    # 5.1
    def agegroup(self):
        if self.__age <= 30: age_group="young"
        if (self.__age > 30) & (self.__age < 50): age_group = "middle"
        if (self.__age >= 50): age_group = "senior"
        return age_group

    # 5.2 set and get functions
    def setage(self, age):
        self.__age = age

    # 5.3 Get hidden age
    def getage(self):          # Otherwise, we cannot extract hidden '__age'
        return str(self.__age)

    # 5.4 Get hidden gender
    def getgender(self):       # Otherwise, we cannot extract hidden '__gender'
        return (self.__gender)


# 5.5 Start using 'person' class
p=person(50,"m", "India")
p.agegroup()
p.setage(35)
p.getage()
p.agegroup()
p.getgender()


# 6. Create a child class of 'person'
# Single class inheritance
# student is  a child class/subclass
class student(person):

    # 6.1 Note the variables being passed on to __init__
    def __init__(self, age, gender, country, college):
        person.__init__(self,age,gender,country)       # Call __init__ of parent and pass variables to it
        self.__college = college

    # 6.2 This method supercedes parent's method
    def agegroup(self):
        return (self.getage() + " " + self.getgender())

    # 6.3 This is a new method in this class
    def describe_student(self):
        #d = self.getage() + " " + self.__gender + self.country  # Wrong hidden __age cannot be accessed
        d = self.getage() + " " + self.getgender() + " " + self.country  # Correct
        return d




# 6.4 Use student class
f = student(35,"m", "india", "FORE")
f.getage()            # Invoke method of parent class
f.describe_student()  # Invokde method of student class
f.agegroup()          # Invoke method of child that supercedes parent method

#############################################################
