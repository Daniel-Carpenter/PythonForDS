## Reading and Writing Files

*Recommended Reading:  PDA – Chapter 3*  

### Topics:
* Python file modes
* Python attributes
* `txt` files
* `Excel` files

---

<br>

# Example of Reading and writing to files
> This script provides two simple examples of writing and reading files

## Open a file that already exists and write 500 random numbers


```python
file_loc = "C:/Users/daniel.carpenter/OneDrive - the Chickasaw Nation/Documents/GitHub/OU-DSA/Python for DSA/02 - Notes and Assignments/02 - Functions, Methods, and Loops/03 - Read and Write Files/theOutputFile.txt"

import random

# Open the files
f = open(file_loc, 'w')

# 500 lines to write to file
for count in range(500):
    
    # Make a random number and generate random number
    number = random.randint(1, 500)
    f.write(str(number) + '\n')

f.close() # Close the output
```

## Read the numbers in the created file, then sum up the numbers and print


```python
f = open(file_loc, 'r')
theSum = 0

# For each line in the file
for line in f:
    line = line.strip() # trim the whitespace of each line
    number = int(line)  # Read as coerced int
    theSum += number    # Cumulative sum

f.close() # close the file

print("The sum of numbers from the created file is", theSum)
```

    The sum of numbers from the created file is 120943
    

## Read the entire file as string `f.read()`


```python
f = open(file_loc, 'r')
contentsOfFile = f.read()
f.close() # close the file

# Print it
print(contentsOfFile)
```
    487
    433
    9
    332
    415
    109
    9
    2
    ...
    
    