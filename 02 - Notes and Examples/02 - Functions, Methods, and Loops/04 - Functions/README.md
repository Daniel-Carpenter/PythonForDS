## Functions

*Recommended reading:  PDA - Chapter 3*  

### Topics:
* def
* return
* Namespaces
* Scope:  area in which a given name refers to a given value
* Local Functions
* Errors and Exception Handling


---

<br>

# Example Functions
* This script provides two simple functions
* The first function accepts a number and prints
* out the OU cheer that number of times

## Basic Function (No Return)
* Notice there is not a return on this function
* Every function does not need a return function
* But a function without a return does return None


```python
def functionBoomer(n):
    for i in range(n):
        print("Boomer")
        print("Sooner")
    
# Call it and get your input
cheer = int(input("How many times do you want to see the cheer? "))

functionBoomer(cheer)
```

    How many times do you want to see the cheer? 3
    Boomer
    Sooner
    Boomer
    Sooner
    Boomer
    Sooner
    

## Function that returns result


```python
def functionMath(x, y):
    rtnProd = x * y
    rtnSum  = x + y
    return(rtnProd, rtnSum)

print(functionMath(2, 5))
```

    (10, 7)
    

