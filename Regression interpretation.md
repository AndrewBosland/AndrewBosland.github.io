---
layout: wide_default
---    
# THIS FILE IS IN THE HANDOUTS FOLDER. COPY IT INTO YOUR CLASS NOTES

- [**Read the chapter on the website!**](https://ledatascifi.github.io/ledatascifi-2023/content/05/02_reg.html) It contains a lot of extra information we won't cover in class extensively.
- After reading that, I recommend [this webpage as a complimentary place to get additional intuition.](https://aeturrell.github.io/coding-for-economists/econmt-regression.html)

## Today

[Finish picking teams and declare initial project interests in the project sheet](https://docs.google.com/spreadsheets/d/1kRbuRKfKh9lCdoVBGLxSbDTIRBEfnV7Y8AcP-hZbmTw/edit#gid=1508330834)


# Today is mostly about INTERPRETING COEFFICIENTS (6.4 in the book)

1. 25 min reading groups: Talk/read through two regression pages (6.3 and 6.4) 
    - Assemble your own notes. Perhaps in the "Module 4 notes" file, but you can do this in any file you want.
    - After class, each group will email their notes to the TA/me for participation. (Effort grading.)
1. 10 min: class builds joint "big takeaways and nuanced observations" 
1. 5 min: Interpret models 1-2 as class as practice. 
1. 20 min reading groups: Work through remaining problems below.
1. 10 min: wrap up  

---


```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col # nicer tables

```


```python
url = 'https://github.com/LeDataSciFi/ledatascifi-2023/blob/main/data/Fannie_Mae_Plus_Data.gzip?raw=true'
fannie_mae = pd.read_csv(url,compression='gzip') 
```

## Clean the data and create variables you want


```python
fannie_mae = (fannie_mae
                  # create variables
                  .assign(l_credscore = np.log(fannie_mae['Borrower_Credit_Score_at_Origination']),
                          l_LTV = np.log(fannie_mae['Original_LTV_(OLTV)']),
                          l_int = np.log(fannie_mae['Original_Interest_Rate']),
                          Origination_Date = lambda x: pd.to_datetime(x['Origination_Date']),
                          Origination_Year = lambda x: x['Origination_Date'].dt.year,
                          const = 1
                         )
                  .rename(columns={'Original_Interest_Rate':'int'}) # shorter name will help the table formatting
             )

# create a categorical credit bin var with "pd.cut()"
fannie_mae['creditbins']= pd.cut(fannie_mae['Co-borrower_credit_score_at_origination'],
                                 [0,579,669,739,799,850],
                                 labels=['Very Poor','Fair','Good','Very Good','Exceptional'])

```


```python
fannie_mae['l_LTV'].mean()
```




    4.207507631461931




```python
ltv = 3.2-2.2*(4.2)
print(ltv)
```

    -6.040000000000002


## Statsmodels

As before, the psuedocode:
```python
model = sm_ols(<formula>, data=<dataframe>)
result=model.fit()

# you use result to print summary, get predicted values (.predict) or residuals (.resid)
```

Now, let's save each regression's result with a different name, and below this, output them all in one nice table:


```python
# one var: 'y ~ x' means fit y = a + b*X

reg1 = sm_ols('int ~  Borrower_Credit_Score_at_Origination ', data=fannie_mae).fit()

reg1b= sm_ols('int ~  l_credscore  ',  data=fannie_mae).fit()

reg1c= sm_ols('l_int ~  Borrower_Credit_Score_at_Origination  ',  data=fannie_mae).fit()

reg1d= sm_ols('l_int ~  l_credscore  ',  data=fannie_mae).fit()

# multiple variables: just add them to the formula
# 'y ~ x1 + x2' means fit y = a + b*x1 + c*x2
reg2 = sm_ols('int ~  l_credscore + l_LTV ',  data=fannie_mae).fit()

# interaction terms: Just use *
# Note: always include each variable separately too! (not just x1*x2, but x1+x2+x1*x2)
reg3 = sm_ols('int ~  l_credscore + l_LTV + l_credscore*l_LTV',  data=fannie_mae).fit()
      
# categorical dummies: C() 
reg4 = sm_ols('int ~  C(creditbins)  ',  data=fannie_mae).fit()

reg5 = sm_ols('int ~  C(creditbins)  -1', data=fannie_mae).fit()

```

Ok, time to output them:


```python
# now I'll format an output table
# I'd like to include extra info in the table (not just coefficients)
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

# q4b1 and q4b2 name the dummies differently in the table, so this is a silly fix
reg4.model.exog_names[1:] = reg5.model.exog_names[1:]  

# This summary col function combines a bunch of regressions into one nice table
print('='*108)
print('                  y = interest rate if not specified, log(interest rate else)')
print(summary_col(results=[reg1,reg1b,reg1c,reg1d,reg2,reg3,reg4,reg5], # list the result obj here
                  float_format='%0.2f',
                  stars = True, # stars are easy way to see if anything is statistically significant
                  model_names=['1','2',' 3 (log)','4 (log)','5','6','7','8'], # these are bad names, lol. Usually, just use the y variable name
                  info_dict=info_dict,
                  regressor_order=[ 'Intercept','Borrower_Credit_Score_at_Origination','l_credscore','l_LTV','l_credscore:l_LTV',
                                  'C(creditbins)[Very Poor]','C(creditbins)[Fair]','C(creditbins)[Good]','C(creditbins)[Vrey Good]','C(creditbins)[Exceptional]']
                  )
     )
```

    ============================================================================================================
                      y = interest rate if not specified, log(interest rate else)
    
    ============================================================================================================
                                            1        2      3 (log) 4 (log)     5         6        7        8   
    ------------------------------------------------------------------------------------------------------------
    Intercept                            11.58*** 45.37*** 2.87***  9.50***  44.13*** -16.81*** 6.65***         
                                         (0.05)   (0.29)   (0.01)   (0.06)   (0.30)   (4.11)    (0.08)          
    Borrower_Credit_Score_at_Origination -0.01***          -0.00***                                             
                                         (0.00)            (0.00)                                               
    l_credscore                                   -6.07***          -1.19*** -5.99*** 3.22***                   
                                                  (0.04)            (0.01)   (0.04)   (0.62)                    
    l_LTV                                                                    0.15***  14.61***                  
                                                                             (0.01)   (0.97)                    
    l_credscore:l_LTV                                                                 -2.18***                  
                                                                                      (0.15)                    
    C(creditbins)[Very Poor]                                                                             6.65***
                                                                                                         (0.08) 
    C(creditbins)[Fair]                                                                         -0.63*** 6.02***
                                                                                                (0.08)   (0.02) 
    C(creditbins)[Good]                                                                         -1.17*** 5.48***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Exceptional]                                                                  -2.25*** 4.40***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Very Good]                                                                    -1.65*** 5.00***
                                                                                                (0.08)   (0.01) 
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared Adj.                       0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    Adj R-squared                        0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    No. observations                     134481   134481   134481   134481   134481   134481    67366    67366  
    ============================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


# Today. Work in groups. Refer to the lectures. 

You might need to print out a few individual regressions with more decimals.

1. Interpret coefs in model 1-4
1. Interpret coefs in model 5
1. Interpret coefs in model 6 (and visually?)
1. Interpret coefs in model 7 (and visually? + comp to table)
1. Interpret coefs in model 8 (and visually? + comp to table)
1. Add l_LTV  to Model 8 and interpret (and visually?)






```python
reg1.params[0] + reg1.params[1]*700
```




    5.595708581716892



- Model 1
    - "A 1 unit increase in credit score is associated with a decrease of 0.86 b.p. in interest rates."
    - @x=700, E(y) is 5.5957
    - 700 to 700 --> int rate falls ~6 b.p.
- Model 2
    - "A 1% increase in credit score is associated  with a decrease in 6.07 b.p. in interest rates."
    - @x=700 E(y) is 5.5957
    - 700 to 707 --> inr rate falls ~6 b.p.
- Model 3
    - "A 1 unit increase in credit score is associated with a proportional decrease of 0.17% in interest rate"
    - @x=700 E(y) is 5.5957
    - 700 to 707 --> int rate falls ~6 b.p.
- Model 4
    - "A 1 unit increase in credit score is associated with a proportional decrease of 1.19% in interest rate"
    - @x=700 E(y) is 5.5957
    - 700 to 707 --> int rate falls ~6 b.p.
- Model 5
    - "A 1% increase in credit score is associated  with a decrease in 5.59 b.p. in interest rates HOLDING LOG(LTV) CONSTANT."
- Model 6

# Class Notes

## Regression Model
𝑦=𝑎+𝛽1𝑋1+𝛽2𝑋2

# Interperating coefs on X / Beta
X is Binary:
"A 1 unit increase in X is associated with a B_i change in x, holding all other X constant"
Fill in the blanks, but depends on what X-I is and f(y)
Beta compares jump from False to True

X is categorical:
X = short medium tall, then regression includes
    x = medium
    x = tall
Beta compares jump from "omitted level" to the given level

## Model 1
𝑦=𝑎+𝛽1𝑋1+𝛽2𝑋2

the average value of 𝑦 is 𝑎 for group 0 (because 𝑋1=𝑋2=0 if 𝑋=0)

𝑦 is 𝛽1 units higher on average for cases when 𝑋=1 than when 𝑋=0


𝑦 is 𝛽2 units higher on average for cases when 𝑋=2 than when 𝑋=0

## Model 2
log𝑦=𝑎+𝛽1𝑋1+𝛽2𝑋2

the average value of log𝑦 is 𝑎 for group 0 (because 𝑋1=𝑋2=0if 𝑋=0)

𝑦 is about 100∗𝛽1 % higher on average for cases when 𝑋=1 than when 𝑋=0


𝑦 is about 100∗𝛽2 % higher on average for cases when 𝑋=2 than when 𝑋=0


## X is Interaction Term
log(price)=8.2+1.53∗log(carat)+0.33∗Ideal+0.18∗log(carat)⋅Ideal

Relationship of size on price: 1.53+0.18∗𝐼𝑑𝑒𝑎𝑙

A 1% increase in size is associated with a 1.53% higher price for non ideal diamonds

A 1% increase in size is associated with a 1.71% higher price for ideal diamonds

Relationship of cut on price: 0.33+0.18∗𝑙𝑜𝑔(𝑐𝑎𝑟𝑎𝑡)

For 1 carat diamonds (𝑙𝑜𝑔(1)=0), ideal diamonds are 33% more expensive than non-ideal diamonds

For 2 carat diamonds (𝑙𝑜𝑔(2)=0.693), ideal diamonds are 45% more expensive than non-ideal diamonds

## 𝛽3≠0
 implies that the relationship between carat size and price is different for ideal and non-ideal diamonds.

Mathematically: 1%↑
 in carat →
 price increases by 1.53% for non-ideal but 1.71% for ideal
 
## 𝛽3≠0
 implies that the relationship between cut quality and price is different for diamonds of different sizes.

Mathematically: 1 carat diamonds that are ideal are 33% more expensive than non-ideal diamonds, but 2 carat ideal diamonds are 45% more expensive than non-ideal diamonds

## N controls
𝑦=𝑎+𝛽0𝑋0+𝛽1𝑋1+...+𝛽𝑁𝑋𝑁+𝑢
