# Week 2 - Files
# This script provides two simple examples of writing and reading files

"""
This block will open a file that already exists and write 500
random numbers
"""

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


"""
Read the numbers in the created file
Sum up the numbers and print
"""
f = open(file_loc, 'r')
theSum = 0

# For each line in the file
for line in f:
    line = line.strip() # trim the whitespace of each line
    number = int(line)  # Read as coerced int
    theSum += number    # Cumulative sum

f.close() # close the file

print("The sum of numbers from the created file is", theSum)
