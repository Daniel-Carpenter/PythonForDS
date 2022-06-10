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
    

# Distribution of salary  ---------------------------------------------------------

import matplotlib.pyplot as plt

## Create the figure
fig, axs = plt.subplots(2, 1, constrained_layout=False)

## The title of the figure
fig.suptitle("Distribution of Head Coach Salaries")

## A histogram of salaries
axs[0].hist(df['TotalPay'], density = False, stacked = False, rwidth = .8)
axs[0].set_ylabel('Number of Schools')

## A boxplot of salaries
axs[1].boxplot(df['TotalPay'], vert=False)
axs[1].set_xlabel('Salary (Millions of Dollars)')
axs[1].set_ylabel('Number of Schools')

plt.show()


# Boxplot by conference -------------------------------------------------------
bplots = sns.boxplot(y="Conference", x="TotalPay", color = "steelblue1",
                     data=df.sort_values(by="TotalPay"))
bplots.set_title('Distribution of Head Coach Salaries')
bplots.set_xlabel('Salary (Millions of Dollars)')


# =============================================================================
# Fit a regression model with the salary as the response variable
# =============================================================================

import statsmodels.formula.api as smf # Linear regression package

# Create the model
# Note Did not include LossRecord or WinLossRatio since multicolinearity
# Note Did not include Score since potential colinearity with Points per game
model = smf.ols(formula='TotalPay ~ Conference + GradRate + StadSize + WinRecord + OffenseScore + DefenseScore + PointsPerGame',
                data=df)

# The estimation using OLS
est = model.fit()

# Show the fitted model
print(est.summary())

# Get the coefficients
coefsRaw = est.params      # As series
coefs = np.array(coefsRaw) # AS np array


# =============================================================================
# What is the recommended salary for the OU football coach?
# =============================================================================

# Create a function to compare salary of a given school to its estimated salary based on data
# Also can perform what if analysis with the conference. E.g. move a coach to another conf.
def estimateCoachSalary(chosenSchool,    # The school to see check estiamted salary of
                        whatIfConference # Can change the conference to see if salary changes
                        ):

    # Filter to the chosen school
    school_Stats = df.query("School == @chosenSchool")
    
    # Input paramaters of the school
    school_Conference    = 'Conference[T.' + whatIfConference +']'
    school_Coach         = str(np.array(school_Stats['Coach'])[0])
    school_TotalPay      = float(school_Stats['TotalPay'])
    school_GradRate      = float(school_Stats['GradRate'])
    school_StadSize      = float(school_Stats['StadSize'])
    school_WinRecord     = float(school_Stats['WinRecord'])
    school_OffenseScore  = float(school_Stats['OffenseScore'])
    school_DefenseScore  = float(school_Stats['DefenseScore'])
    school_PointsPerGame = float(school_Stats['PointsPerGame'])
    
    # Input paramaters of the school in list
    school_Params = [1, 1, # To indicate intercept and conference
                     school_GradRate, school_StadSize,  school_WinRecord, 
                     school_OffenseScore,  school_DefenseScore, school_PointsPerGame]
    
    
    tailoredCoefs = [] # To hold a list of tailored coefficients
    
    for variable in range(len(coefsRaw)):
    
        indexName     = coefsRaw.index[variable]    
        variableToAdd = coefs[variable]
    
        # Only add if in the right conference
        if not (indexName.startswith("Conference") and not indexName == school_Conference):
            tailoredCoefs.append(variableToAdd)
    
    
    # Now calculate the expected salary by doing matrix multipication
    estSalary = np.dot(np.array(tailoredCoefs), np.array(school_Params))
    
    # Summary of if overpaid
    isOverpaid = estSalary < school_TotalPay
    if isOverpaid: paidOverUnder = 'overpaid' 
    else: paidOverUnder = 'underpaid'
    
    # Summary of salary
    print('\nThe estimated (modeled) salary of', chosenSchool, 'in the', whatIfConference,
          'is $', '{:,.2f}'.format(estSalary),
          '\nCurrently,', chosenSchool, 'pays $', '{:,.0f}'.format(school_TotalPay),
          '\n Therfore,', school_Coach, 'is', paidOverUnder)
    
    # Return the estimated salary
    return estSalary
    

# Check the estimated salary of Oklahoma coach 
estimateCoachSalary(chosenSchool = 'Oklahoma',
                    whatIfConference = 'Big 12')


# =============================================================================
# What would the appropriate salary be if OU moved to the SEC?
# =============================================================================

# Change the conference of OU to the SEC
estimateCoachSalary(chosenSchool = 'Oklahoma',
                    whatIfConference = 'SEC')


# =============================================================================
# What schools did we drop from our data and why?
# =============================================================================

print('\nBelow shows the schools dropped from the sample since there were null values:\n',
      removedSchools)


# =============================================================================
# What is the single biggest impact on salary size?
# =============================================================================

# Get the most impactful parameter (max) - note not best fit, highest value
maxValue         = max(coefs)                  # The coefficient of the most impactful
idxOfMaxValue    = np.where(coefs == maxValue) # The idx of the max value
mostImpactfulVar = coefsRaw.index[idxOfMaxValue]     # The most impactful variable and value

print('\nThe most impactful variable is:', mostImpactfulVar[0], 
      '\nI.e., being in the', mostImpactfulVar[0], 
      'correlates with an increase in salary of $', '{:,.2f}'.format(maxValue),
      '\nPlease not that this does not consider the statistical significance.')

print('\nNote that big 12 coefficient has a statistically signifcant t-value')
print('Model also has very high R squared, indicating that most of the variation',
      'in the independant variables explain the dependant variable.')


