# -*- coding: utf-8 -*-
"""
Homework 2
Modeling a Salary
Daniel Carpenter

Purpose of File:
    1. Import Football datasets to predict:
            - recommended salary for the OU football coach
            - appropriate salary be if OU moved to the SEC
            - single biggest impact on salary size
"""

# Packages
import pandas  as pd
import numpy   as np
import seaborn as sns # Plots


# =============================================================================
# Import data from https://github.com/BoomerPython/data
# Data automatically in form of data frame
# =============================================================================

# Data for the coaches dataset
Coaches = pd.read_csv('https://raw.githubusercontent.com/BoomerPython/data/main/coaches_bp_dsa.csv')

# Data for the Stadiums dataset
Stadiums = pd.read_csv('https://raw.githubusercontent.com/BoomerPython/data/main/stadium_bp_dsa.csv')

# Data for the Team Statistics dataset
TeamStats = pd.read_csv('https://raw.githubusercontent.com/BoomerPython/data/main/stats_bp_dsa.csv')


# =============================================================================
# Build a data frame for your analysis incuding data pulled in above
# Join the three above datasets together on the identifiers `School` and `Conf`
# =============================================================================

# Left Join dataframs: Coaches to Stadiums 
df = pd.merge(Coaches, Stadiums, 
              on = ['School', 'Conf'], # Keys to join to
              how = 'left')            # Left join

# Left Join above df to TeamStats
df = pd.merge(df, TeamStats, 
              on = ['School', 'Conf'], # Keys to join to
              how = 'left')            # Left join

# Drop null values from dataset since will not work with model ----------------

## Create a copy of the schools before removing the NAs
allSchools = df[:]

## Drop the schools with NAs
colsDropped = df.isna().sum()
print('\nSummary of NA values present by column in data frame:', colsDropped)
df = df.dropna()

## Capture a copy of a list of the removed schools
removedSchools = allSchools[~allSchools['School'].isin(df['School'])]['School'].unique()



# Trim the column names and rename some for coding ease -----------------------
## Note there was whitespace in the column names
df = df.rename(columns=lambda x: x.strip())
df = df.rename(columns = {'Graduation Rate (GSR)': 'GradRate',
                          'Defense Score':         'DefenseScore',
                          'OffenceScore':          'OffenseScore',
                          'W':                     'WinRecord',
                          'L':                     'LossRecord',
                          'Ratio':                 'WinLossRatio',
                          'Conf':                  'Conference'})

## View the changes
print('\n', df.columns)


# Convert Non-Numeric Columns to Numeric that should be -----------------------

## Trim out whitespace
df['TotalPay'] = df['TotalPay'].str.strip()
df['StadSize'] = df['StadSize'].str.strip()

## Stadium size: Convert to type float
df['StadSize'] = df['StadSize'].str.replace(',', '').astype(float)

## Total Pay of coach: Convert to type float
### First remove commas 
df['TotalPay'] = df['TotalPay'].str.replace(',', '')

### Remove $ signs and convert to number
df['TotalPay'] =  pd.to_numeric(df['TotalPay'].str.replace('$', '').astype(float),
                                errors = 'coerce')

### View changes
print('\n', df.head())


# =============================================================================
# Conduct an initial data analysis - develop appropriate visualizations
# =============================================================================

# Pairs plots - https://seaborn.pydata.org/generated/seaborn.pairplot.html

## Initial Look to see how variables correlate with pay -----------------------
allConfs = sns.pairplot(df, y_vars=['TotalPay'], 
                        kind="reg", # add Regression line
                        plot_kws={'line_kws':{'color':'orange'}} 
                        )

allConfs.fig.suptitle('Pair Plot of all Conferences. Note Stadium Size and Score')

## Does the conference have anything to do? -----------------------------------

### Get Unique list of conferences to iterate over
uniqueConferences = df['Conference'].unique()

### Plot each conference to see if each conference holds to overall trends
for conference in uniqueConferences:
    
    # Filter dataframe to conference
    df_tempConf = df[df['Conference'] == conference]
    
    # Pair plot of JUST the conference
    thePlot = sns.pairplot(df_tempConf, 
                           kind="reg", # add Regression line
                           plot_kws={'line_kws':{'color':'gray'}},
                           y_vars=['TotalPay']
                           )

    # Add some titles
    thePlot.fig.suptitle("Conference Name: " + conference + " | Schools in Conf.: " + str(len(df_tempConf['School'])))

# Boxplots/distributional

# =============================================================================
# Fit a regression model with the salary as the response variable
# =============================================================================


# =============================================================================
# What is the recommended salary for the OU football coach?
# =============================================================================

# Create a function with the paramters and fit with OU data

# =============================================================================
# What would the appropriate salary be if OU moved to the SEC?
# =============================================================================

# Change the parameter to the SEC

# =============================================================================
# What schools did we drop from our data and why?
# =============================================================================

print('Below shows the schools dropped from the sample since there were null values:\n',
      removedSchools)

# =============================================================================
# What is the single biggest impact on salary size?
# =============================================================================

# Largest coefficient

# =============================================================================
# Bonus points for adding additional (relevant) data.
# =============================================================================


