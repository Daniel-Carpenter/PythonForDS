<a href="https://colab.research.google.com/github/BoomerPython/Week_1/blob/main/DSA_Week1_Variables.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This file provides some practical experience with variables, operators, data types, and making programs interactive

## Variables

Variables are names given to data that we need to store and manipulate in our programs.


```python
# We can declare each variable individually

coachSalary = 1200
coachName = "Bob"
print("$", coachSalary, " ", coachName)
```

    $ 1200   Bob
    


```python
# We can declare variables collectively

coachSalary, coachName = 1400, "Lincoln"
print("$", coachSalary, " ", coachName)
```

    $ 1400   Lincoln
    


```python
# Naming a variable
# Best practices:  letters, numbers, underscores
# No reserved words (like print, input, etc)
# Names are case sensitive
# Our convention for variables - camelCase
# Our convention for files - under_score
```

## Assignment Operators


```python
# Basic assignment operations

x = 5
y = 10
x = y
# y = x   # what is the difference here?
print ("x = ", x)
print ("y = ", y)
```

    x =  5
    y =  5
    

# Data Types

# Lists


```python
# Declare the list
# List elements can be different data types

myList = [1, 2, 3, 4, 5, "Lincoln"]
print(myList)

# print the third item
# you should get a 3
print(myList[2])

#print the last item
# you should get "Lincoln"
print(myList[-1])
```

    [1, 2, 3, 4, 5, 'Lincoln']
    3
    Lincoln
    

# Tuple


```python
# Tuples are like lists but we cannot modify their
# values.
# Good for things that will not change - like monthss
# of year

monthsOfYear = ("Jan", "Feb", "Mar", "Apr",
                "May", "Jun", "Jul", "Aug",
                "Sep", "Oct", "Nov", "Dec")
```


```python
# You can access just like a list - with the index

monthsOfYear[5]

```




    'Jun'




```python
monthsOfYear[-1]
```




    'Dec'



# Dictionary

Dictionary is a collection of related data PAIRS

To declare a dictionary, your write:


```python
myCoaches = {"Barry":600, "Gary":700, "Howard":800}
```


```python
print(myCoaches)
```

    {'Barry': 600, 'Gary': 700, 'Howard': 800, 'John': 'Do we acknowledge him'}
    


```python
myCoaches["John"] = "Do we acknowledge him"
```


```python
del myCoaches["John"]
```


```python
print(myCoaches)
```

    {'Barry': 600, 'Gary': 700, 'Howard': 800}
    
