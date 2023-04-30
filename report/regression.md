## Part 1: EDA

_Insert cells as needed below to write a short EDA/data section that summarizes the data for someone who has never opened it before._ 
- Answer essential questions about the dataset (observation units, time period, sample size, many of the questions above) 
- Note any issues you have with the data (variable X has problem Y that needs to get addressed before using it in regressions or a prediction model because Z)
- Present any visual results you think are interesting or important

# EDA
This dataset is a dataset pertaining to residential homes and the unit of observation is the parcel which each respresent an individual residential home. The time period is from 2006 - 2008 in terms of year sold. It has 81 variables that include 1,941 observations. I noticed that there are a large number of missing values for v_Lot_Ftontage in particular about 321 missing observations. When visually exploring the data I had some interesting findings, some that supported my initial hypotheses and some that didnt. For example, I initially thought the year in which the house was built would have a somewhat linear effect on the sale price of the house and it does. On the other hand, I would have thought lot size would have had a more linear relationship with sale price and was surprised to find out the opposite.


```python
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols as sm_ols
import matplotlib.pyplot as plt

file_path = "input_data2/housing_train.csv"

housing_df = pd.read_csv(file_path)

print(housing_df.head())

print(housing_df.describe())

print(housing_df.columns)

print(housing_df.isna().sum())

plt.figure(figsize=(8,6))
sns.scatterplot(x='v_Lot_Area', y='v_SalePrice', data=housing_df)
plt.title('v_SalePrice vs. v_Lot_Area')
plt.xlabel('v_Lot_Area')
plt.ylabel('v_SalePrice')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='v_Neighborhood', y='v_SalePrice', data=housing_df)
plt.title('v_SalePrice by v_Neighborhood')
plt.xlabel('v_Neighborhood')
plt.ylabel('v_SalePrice')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='v_Year_Built', y='v_SalePrice', data=housing_df)
plt.title('v_SalePrice vs. v_Year_Built')
plt.xlabel('v_Year_Built')
plt.ylabel('v_SalePrice')
plt.show()
```

               parcel  v_MS_SubClass v_MS_Zoning  v_Lot_Frontage  v_Lot_Area  \
    0  1056_528110080             20          RL           107.0       13891   
    1  1055_528108150             20          RL            98.0       12704   
    2  1053_528104050             20          RL           114.0       14803   
    3  2213_909275160             20          RL           126.0       13108   
    4  1051_528102030             20          RL            96.0       12444   
    
      v_Street v_Alley v_Lot_Shape v_Land_Contour v_Utilities  ... v_Pool_Area  \
    0     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    1     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    2     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    3     Pave     NaN         IR2            HLS      AllPub  ...           0   
    4     Pave     NaN         Reg            Lvl      AllPub  ...           0   
    
      v_Pool_QC v_Fence v_Misc_Feature v_Misc_Val v_Mo_Sold v_Yr_Sold  \
    0       NaN     NaN            NaN          0         1      2008   
    1       NaN     NaN            NaN          0         1      2008   
    2       NaN     NaN            NaN          0         6      2008   
    3       NaN     NaN            NaN          0         6      2007   
    4       NaN     NaN            NaN          0        11      2008   
    
       v_Sale_Type  v_Sale_Condition  v_SalePrice  
    0          New           Partial       372402  
    1          New           Partial       317500  
    2          New           Partial       385000  
    3          WD             Normal       153500  
    4          New           Partial       394617  
    
    [5 rows x 81 columns]
           v_MS_SubClass  v_Lot_Frontage     v_Lot_Area  v_Overall_Qual  \
    count    1941.000000     1620.000000    1941.000000     1941.000000   
    mean       58.088614       69.301235   10284.770222        6.113344   
    std        42.946015       23.978101    7832.295527        1.401594   
    min        20.000000       21.000000    1470.000000        1.000000   
    25%        20.000000       58.000000    7420.000000        5.000000   
    50%        50.000000       68.000000    9450.000000        6.000000   
    75%        70.000000       80.000000   11631.000000        7.000000   
    max       190.000000      313.000000  164660.000000       10.000000   
    
           v_Overall_Cond  v_Year_Built  v_Year_Remod/Add  v_Mas_Vnr_Area  \
    count     1941.000000   1941.000000       1941.000000     1923.000000   
    mean         5.568264   1971.321999       1984.073158      104.846074   
    std          1.087465     30.209933         20.837338      184.982611   
    min          1.000000   1872.000000       1950.000000        0.000000   
    25%          5.000000   1953.000000       1965.000000        0.000000   
    50%          5.000000   1973.000000       1993.000000        0.000000   
    75%          6.000000   2001.000000       2004.000000      168.000000   
    max          9.000000   2008.000000       2009.000000     1600.000000   
    
           v_BsmtFin_SF_1  v_BsmtFin_SF_2  ...  v_Wood_Deck_SF  v_Open_Porch_SF  \
    count     1940.000000     1940.000000  ...     1941.000000      1941.000000   
    mean       436.986598       49.247938  ...       92.458011        49.157135   
    std        457.815715      169.555232  ...      127.020523        70.296277   
    min          0.000000        0.000000  ...        0.000000         0.000000   
    25%          0.000000        0.000000  ...        0.000000         0.000000   
    50%        361.500000        0.000000  ...        0.000000        28.000000   
    75%        735.250000        0.000000  ...      168.000000        72.000000   
    max       5644.000000     1474.000000  ...     1424.000000       742.000000   
    
           v_Enclosed_Porch  v_3Ssn_Porch  v_Screen_Porch  v_Pool_Area  \
    count       1941.000000   1941.000000     1941.000000  1941.000000   
    mean          22.947965      2.249871       16.249871     3.386399   
    std           65.249307     22.416832       56.748086    43.695267   
    min            0.000000      0.000000        0.000000     0.000000   
    25%            0.000000      0.000000        0.000000     0.000000   
    50%            0.000000      0.000000        0.000000     0.000000   
    75%            0.000000      0.000000        0.000000     0.000000   
    max         1012.000000    407.000000      576.000000   800.000000   
    
             v_Misc_Val    v_Mo_Sold    v_Yr_Sold    v_SalePrice  
    count   1941.000000  1941.000000  1941.000000    1941.000000  
    mean      52.553838     6.431221  2006.998454  182033.238022  
    std      616.064459     2.745199     0.801736   80407.100395  
    min        0.000000     1.000000  2006.000000   13100.000000  
    25%        0.000000     5.000000  2006.000000  130000.000000  
    50%        0.000000     6.000000  2007.000000  161900.000000  
    75%        0.000000     8.000000  2008.000000  215000.000000  
    max    17000.000000    12.000000  2008.000000  755000.000000  
    
    [8 rows x 37 columns]
    Index(['parcel', 'v_MS_SubClass', 'v_MS_Zoning', 'v_Lot_Frontage',
           'v_Lot_Area', 'v_Street', 'v_Alley', 'v_Lot_Shape', 'v_Land_Contour',
           'v_Utilities', 'v_Lot_Config', 'v_Land_Slope', 'v_Neighborhood',
           'v_Condition_1', 'v_Condition_2', 'v_Bldg_Type', 'v_House_Style',
           'v_Overall_Qual', 'v_Overall_Cond', 'v_Year_Built', 'v_Year_Remod/Add',
           'v_Roof_Style', 'v_Roof_Matl', 'v_Exterior_1st', 'v_Exterior_2nd',
           'v_Mas_Vnr_Type', 'v_Mas_Vnr_Area', 'v_Exter_Qual', 'v_Exter_Cond',
           'v_Foundation', 'v_Bsmt_Qual', 'v_Bsmt_Cond', 'v_Bsmt_Exposure',
           'v_BsmtFin_Type_1', 'v_BsmtFin_SF_1', 'v_BsmtFin_Type_2',
           'v_BsmtFin_SF_2', 'v_Bsmt_Unf_SF', 'v_Total_Bsmt_SF', 'v_Heating',
           'v_Heating_QC', 'v_Central_Air', 'v_Electrical', 'v_1st_Flr_SF',
           'v_2nd_Flr_SF', 'v_Low_Qual_Fin_SF', 'v_Gr_Liv_Area',
           'v_Bsmt_Full_Bath', 'v_Bsmt_Half_Bath', 'v_Full_Bath', 'v_Half_Bath',
           'v_Bedroom_AbvGr', 'v_Kitchen_AbvGr', 'v_Kitchen_Qual',
           'v_TotRms_AbvGrd', 'v_Functional', 'v_Fireplaces', 'v_Fireplace_Qu',
           'v_Garage_Type', 'v_Garage_Yr_Blt', 'v_Garage_Finish', 'v_Garage_Cars',
           'v_Garage_Area', 'v_Garage_Qual', 'v_Garage_Cond', 'v_Paved_Drive',
           'v_Wood_Deck_SF', 'v_Open_Porch_SF', 'v_Enclosed_Porch', 'v_3Ssn_Porch',
           'v_Screen_Porch', 'v_Pool_Area', 'v_Pool_QC', 'v_Fence',
           'v_Misc_Feature', 'v_Misc_Val', 'v_Mo_Sold', 'v_Yr_Sold', 'v_Sale_Type',
           'v_Sale_Condition', 'v_SalePrice'],
          dtype='object')
    parcel                0
    v_MS_SubClass         0
    v_MS_Zoning           0
    v_Lot_Frontage      321
    v_Lot_Area            0
                       ... 
    v_Mo_Sold             0
    v_Yr_Sold             0
    v_Sale_Type           0
    v_Sale_Condition      0
    v_SalePrice           0
    Length: 81, dtype: int64



    
![png](output_2_1.png)
    



    
![png](output_2_2.png)
    



    
![png](output_2_3.png)
    


## Part 2: Running Regressions

**Run these regressions on the RAW data, even if you found data issues that you think should be addressed.**

_Insert cells as needed below to run these regressions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Yr_Sold}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * (\text{v_Yr_Sold==2007})+ \beta_2 * (\text{v_Yr_Sold==2008})$
1. Choose your own adventure: Pick any five variables from the dataset that you think will generate good R2. Use them in a regression of $log(\text{Sale Price}_{i,t})$ 
    - Tip: You can transform/create these five variables however you want, even if it creates extra variables. For example: I'd count Model 6 above as only using one variable: `v_Yr_Sold`.
    - I got an R2 of 0.877 with just "5" variables. How close can you get? I won't be shocked if someone beats that!
    

**Bonus formatting trick:** Instead of reporting all regressions separately, report all seven regressions in a _single_ table using `summary_col`.



```python
#1
model1 = sm_ols(formula='v_SalePrice ~ v_Lot_Area', data=housing_df).fit()

# print(model1.summary())

#2
housing_df['log_Lot_Area'] = np.log(housing_df['v_Lot_Area'])

model2 = sm_ols(formula='v_SalePrice ~ log_Lot_Area', data=housing_df).fit()

# print(model2.summary())

#3
housing_df['log_SalePrice'] = np.log(housing_df['v_SalePrice'])

model3 = sm_ols(formula='log_SalePrice ~ v_Lot_Area', data=housing_df).fit()

# print(model3.summary())

#4
model4 = sm_ols(formula='log_SalePrice ~ log_Lot_Area', data=housing_df).fit()

# print(model4.summary())

#5
model5 = sm_ols(formula='v_SalePrice ~ v_Yr_Sold', data=housing_df).fit()

# print(model5.summary())

#6
housing_df['yr_2007'] = (housing_df['v_Yr_Sold'] == 2007).astype(int)
housing_df['yr_2008'] = (housing_df['v_Yr_Sold'] == 2008).astype(int)

model6 = sm_ols(formula='log_SalePrice ~ yr_2007 + yr_2008', data=housing_df).fit()

# print(model6.summary())

#7
model7 = sm_ols(formula='log_SalePrice ~ v_Lot_Area + v_Year_Built + v_Neighborhood + v_MS_Zoning + v_Overall_Cond', data=housing_df).fit()

# print(model7.summary())
```


```python
from statsmodels.iolib.summary2 import summary_col

model_results = [model1, model2, model3, model4, model5, model6, model7]

table = summary_col(model_results, 
                    model_names=['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6', 'Model 7'], 
                    float_format='%.2f', stars=True)

print(table)
```

    
    ====================================================================================================
                                Model 1       Model 2    Model 3  Model 4   Model 5    Model 6  Model 7 
    ----------------------------------------------------------------------------------------------------
    Intercept                 154789.55*** -327915.80*** 11.89*** 9.41*** 3103798.15   12.02*** 0.32    
                              (2911.59)    (30221.35)    (0.01)   (0.15)  (4570621.48) (0.02)   (0.80)  
    R-squared                 0.07         0.13          0.06     0.13    0.00         0.00     0.68    
    R-squared Adj.            0.07         0.13          0.06     0.13    -0.00        0.00     0.68    
    log_Lot_Area                           56028.17***            0.29***                               
                                           (3315.14)              (0.02)                                
    v_Lot_Area                2.65***                    0.00***                                0.00*** 
                              (0.23)                     (0.00)                                 (0.00)  
    v_MS_Zoning[T.C (all)]                                                                      1.14*** 
                                                                                                (0.17)  
    v_MS_Zoning[T.FV]                                                                           1.22*** 
                                                                                                (0.17)  
    v_MS_Zoning[T.I (all)]                                                                      1.36*** 
                                                                                                (0.28)  
    v_MS_Zoning[T.RH]                                                                           1.28*** 
                                                                                                (0.18)  
    v_MS_Zoning[T.RL]                                                                           1.34*** 
                                                                                                (0.16)  
    v_MS_Zoning[T.RM]                                                                           1.37*** 
                                                                                                (0.16)  
    v_Neighborhood[T.Blueste]                                                                   -0.29** 
                                                                                                (0.13)  
    v_Neighborhood[T.BrDale]                                                                    -0.53***
                                                                                                (0.08)  
    v_Neighborhood[T.BrkSide]                                                                   -0.32***
                                                                                                (0.07)  
    v_Neighborhood[T.ClearCr]                                                                   -0.08   
                                                                                                (0.07)  
    v_Neighborhood[T.CollgCr]                                                                   -0.08   
                                                                                                (0.05)  
    v_Neighborhood[T.Crawfor]                                                                   0.12*   
                                                                                                (0.06)  
    v_Neighborhood[T.Edwards]                                                                   -0.34***
                                                                                                (0.06)  
    v_Neighborhood[T.Gilbert]                                                                   -0.13** 
                                                                                                (0.06)  
    v_Neighborhood[T.Greens]                                                                    0.08    
                                                                                                (0.12)  
    v_Neighborhood[T.GrnHill]                                                                   0.29*   
                                                                                                (0.17)  
    v_Neighborhood[T.IDOTRR]                                                                    -0.39***
                                                                                                (0.07)  
    v_Neighborhood[T.Landmrk]                                                                   -0.26   
                                                                                                (0.24)  
    v_Neighborhood[T.MeadowV]                                                                   -0.64***
                                                                                                (0.07)  
    v_Neighborhood[T.Mitchel]                                                                   -0.26***
                                                                                                (0.06)  
    v_Neighborhood[T.NAmes]                                                                     -0.25***
                                                                                                (0.06)  
    v_Neighborhood[T.NPkVill]                                                                   -0.26***
                                                                                                (0.09)  
    v_Neighborhood[T.NWAmes]                                                                    -0.09   
                                                                                                (0.06)  
    v_Neighborhood[T.NoRidge]                                                                   0.44*** 
                                                                                                (0.06)  
    v_Neighborhood[T.NridgHt]                                                                   0.34*** 
                                                                                                (0.06)  
    v_Neighborhood[T.OldTown]                                                                   -0.27***
                                                                                                (0.07)  
    v_Neighborhood[T.SWISU]                                                                     -0.12*  
                                                                                                (0.07)  
    v_Neighborhood[T.SawyerW]                                                                   -0.11*  
                                                                                                (0.06)  
    v_Neighborhood[T.Sawyer]                                                                    -0.30***
                                                                                                (0.06)  
    v_Neighborhood[T.Somerst]                                                                   0.14**  
                                                                                                (0.07)  
    v_Neighborhood[T.StoneBr]                                                                   0.37*** 
                                                                                                (0.06)  
    v_Neighborhood[T.Timber]                                                                    0.08    
                                                                                                (0.06)  
    v_Neighborhood[T.Veenker]                                                                   0.10    
                                                                                                (0.07)  
    v_Overall_Cond                                                                              0.07*** 
                                                                                                (0.01)  
    v_Year_Built                                                                                0.01*** 
                                                                                                (0.00)  
    v_Yr_Sold                                                             -1455.79                      
                                                                          (2277.34)                     
    yr_2007                                                                            0.03             
                                                                                       (0.02)           
    yr_2008                                                                            -0.01            
                                                                                       (0.02)           
    ====================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. If you didn't use the `summary_col` trick, list $\beta_1$ for Models 1-6 to make it easier on your graders.
1. Interpret $\beta_1$ in Model 2. 
1. Interpret $\beta_1$ in Model 3. 
    - HINT: You might need to print out more decimal places. Show at least 2 non-zero digits. 
1. Of models 1-4, which do you think best explains the data and why?
1. Interpret $\beta_1$ In Model 5
1. Interpret $\alpha$ in Model 6
1. Interpret $\beta_1$ in Model 6
1. Why is the R2 of Model 6 higher than the R2 of Model 5?
1. What variables did you include in Model 7?
1. What is the R2 of your Model 7?
1. Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 
1. Speculate (not graded): Could you use the specification of Model 5 in a predictive regression? 


# Answers
1. See above

2. 𝛽1 = 56,028.17 - A 1% increase in lot area results in a \$560 increase in sales price holding all other variables constant.

3. 𝛽1 = .00001309 - A 1 sq ft increase in lot area is associated with a 0.0013% increase in sale price holding all other variables constant.

4. Of models 1-4, I think model 5 best explains the data because it has the highest R-squared and Adj. R-squared value of .13 out of the 4 models. At first glance I thought model 2 but upon further investigating model 4 has a higher R-squared.

5. 𝛽1 = .00001309 A 1 unit increase in year results in a decrease of 0.5% in sales price holding all other variables constant

6. 𝛼 = 12.02 which means that the average log of the sales price not sold in 2007 or 2008 (2006) 12.02.

7. 𝛽1 = .0256 - If a home is sold in year 2007, the sales price is 3% higher on average than 2006.

8. The R2 of Model 6 (.001) is higher than R2 of Model 5 (.000) because model 6 is the more flexible model of the 2 models. This is a result of the 2 indicator variables we created instead of the continuous year variable used in model 5.

9. The variables I included in model 7 were v_Lot_Area, v_Year_Built, v_Neighborhood, v_MS_Zoning, v_Overall_Cond.

10. The R2 of my model 7 is 0.68

11. I would think that Model 6 would not be a good fit for predictive regression because of the R-squared of .001 and if the home was sold in any time besides 2006-2008 the model would not be able to predict the sales price.

12. Although model 5 has a low R-squared (.000) as well, it is a able model for predictive regression, we just do not know how accurate the predictions will be. 
