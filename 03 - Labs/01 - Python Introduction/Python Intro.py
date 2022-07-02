# -*- coding: utf-8 -*-
"""
Homework 1
Daniel Carpenter

Purpose of File - Function that:
    1. Calculates the cost/square footage of a house given user input
    2. File will also compare to the standard cost/sq. ft. of a house
"""

def sqFtComparisonTool():

    print("--- This tool will compare your home's cost/sq. ft. to the standard cost ---")
    
    # Require user input of price
    housePrice = float(input("Please enter the price of your home"))
    
    # Require user input of square footage
    houseSqFt = float(input("Please enter the total square footage of your home"))
    
    # Create a function to calculate price per square foot of a house:
    def calcPricePerSqFt(housePrice, houseSqFt):
        
        if houseSqFt > 0: # div/0 check
            return housePrice / houseSqFt
        else:
            print('Please enter a value greatr than 0 for house sq. ft.')
    
    # Print the cost per square foot
    pricePerSqFt = calcPricePerSqFt(housePrice, houseSqFt)
    print('Your house is $', pricePerSqFt, 'per sq. ft.')
    
    # Create a variable called Standard and set to 125.0
    Standard = 125
    print('\nThe standard for a house is $', Standard, 'per sq. ft.')
    
    # Compare the result with a Standard
    ## if the cost is less, print a positive response
    if Standard > pricePerSqFt:
        print('Look at you go! You got a deal for your house.')
    
    ## if the cost is more, print a negative response
    elif Standard < pricePerSqFt:
        print('Unfortunately the cost per square footage of your home is greater than the standard cost.')
    
    ## Else the cost is the same
    else:
        print('Cost per Square Foot is the same as the standard.')


# Example of Less than
sqFtComparisonTool()

# Example of greater than
sqFtComparisonTool()

# Example of the same
sqFtComparisonTool()