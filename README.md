# Customer Lifetime Value(CLTV)
"Customer Lifetime Value is a monetary value that represents the amount of revenue or profit a customer will give the 
company over the period of the relationship" (Source). CLTV demonstrates the implications of acquiring long-term customers 
compare to short-term customers. Customer lifetime value (CLV) can help you to answers the most important questions about 
sales to every company:

1. How to Identify the most profitable customers?
2. How can a company offer the best product and make the most money?
3. How to segment profitable customers?
4. How much budget need to spend to acquire customers?

## Calculate Customer Lifetime Value
There are lots of approaches available for calculating CLTV. Everyone has his/her own view on it. For computing CLTV we 
need historical data of customers but we will be unable to calculate for new customers. To solve this problem Business 
Analyst develops machine learning models to predict the CLTV of newly customers. Let's explore one of the approaches for
CLTV Calculation:

Using the following equation to calculate CLTV
    
    CLTV = ((Average Order Value x Purchase Frequency)/Churn Rate) x Profit margin.

- **Average Order Value(AOV)**: The Average Order value is the ratio of your total revenue and the total number of orders. 
AOV represents the mean amount of revenue that the customer spends on an order.

    Average Order Value = Total Revenue / Total Number of Orders
    
- **Purchase Frequency**: Purchase Frequency is the ratio of the total number of orders and the total number of customer. 
It represents the average number of orders placed by each customer.    

    Purchase Frequency =  Total Number of Orders / Total Number of Customers
    
- **Churn Rate**: Percentage of customers who have not ordered again.

- **Customer Lifetime**: Customer Lifetime is the period of time that the customer has been continuously ordering.

    Customer lifetime = 1 / Churn Rate
    
- **Repeat Rate**: Repeat rate can be defined as the ratio of the number of customers with more than one order to the 
number of unique customers. Example: If you have 10 customers in a month out of who 4 come back, your repeat rate is 40%.

    Repeat Rate = 1 - Churn Rate
    
## CLTV Implementation

#### Importing the required library

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```   

#### Loading the dataset
Let's first load the required Online Retail dataset using the pandas read CSV function.

```python
data = pd.read_excel('online-retail-data/online-retail.xlsx')
print(data.head())
```

```sh
  InvoiceNo StockCode                          Description  Quantity         InvoiceDate  UnitPrice  CustomerID         Country
0    536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6 2010-12-01 08:26:00       2.55     17850.0  United Kingdom
1    536365     71053                  WHITE METAL LANTERN         6 2010-12-01 08:26:00       3.39     17850.0  United Kingdom
2    536365    84406B       CREAM CUPID HEARTS COAT HANGER         8 2010-12-01 08:26:00       2.75     17850.0  United Kingdom
3    536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6 2010-12-01 08:26:00       3.39     17850.0  United Kingdom
4    536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6 2010-12-01 08:26:00       3.39     17850.0  United Kingdom
```

##### Removing Duplicates to get good view
Sometimes we get a messy dataset to view it graphically. We may have to deal with duplicates, which will skew your 
analysis. 
```python
# let us drop duplicates
filteredData = data[['Country', 'CustomerID']].drop_duplicates()
```

```python
# Top ten country's customer
filteredData.Country.value_counts()[:10].plt(kind='bar')
plt.show(block=True)
```
![top_10_customer_country_insight](https://user-images.githubusercontent.com/35737777/67055654-e09b8400-f140-11e9-9bb4-fbe0e7e5f329.png)

In the given dataset, we can observe most of the customers are from "United Kingdom". So, we can filter data for United 
Kingdom customer.

```python
# Filter only UK data
uk_data = data[data['Country'] == 'United Kingdom']
print(uk_data.describe())
```

```sh 
            Quantity      UnitPrice     CustomerID
count  495478.000000  495478.000000  361878.000000
mean        8.605486       4.532422   15547.871368
std       227.588756      99.315438    1594.402590
min    -80995.000000  -11062.060000   12346.000000
25%         1.000000       1.250000   14194.000000
50%         3.000000       2.100000   15514.000000
75%        10.000000       4.130000   16931.000000
max     80995.000000   38970.000000   18287.000000
```
Here, we can observe some of the customers have ordered in a negative quantity, which is not possible. So, we need to 
filter Quantity greater than zero.

```python
# Removing negative quantity values
uk_data = uk_data[uk_data['Quantity'] > 0]
print(uk_data.describe())
```

#### Filter required Columns
We can filter the necessary columns for calculating CLTV. We only need five columns CustomerID, InvoiceDate, InvoiceNo, 
Quantity, and UnitPrice.

- _CustomerID_ will uniquely define your customers.
- _InvoiceDate_ will help us to calculate numbers of days customer stayed with the product.
- _InvoiceNo_ helps us to count the number of time transaction performed(frequency).
- _Quantity_ is purchased item units in each transaction
- _UnitPrice_ of each unit purchased by the customer will help us to calculate the total purchased amount.

```python
# use only required columns for CLTV
uk_data = uk_data[['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']]
```

Add a new column of total purchase
```python
# calculate total purchase
uk_data['TotalPurchase'] = uk_data['Quantity'] * uk_data['UnitPrice']
```

Now we have to perform 3 operations:
- Calculate the number of days between the present date and the date of last purchase from each customer.
- Calculate the number of orders for each customer.
- Calculate sum of purchase price for each customer.

```python
# perform aggregate operations
uk_data_grp = uk_data.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (date.max() - date.min()).days,
    'InvoiceNo': lambda num: len(num),
    'Quantity': lambda quantity: quantity.sum(),
    'TotalPurchase': lambda total_pur: total_pur.sum()
})

print(uk_data_grp.head())
```
```sh
            InvoiceDate  InvoiceNo  Quantity  TotalPurchase
CustomerID                                                 
12346.0               0          1     74215       77183.60
12747.0             366        103      1275        4196.01
12748.0             372       4596     25748       33719.73
12749.0             209        199      1471        4090.88
12820.0             323         59       722         942.34
```

### Calculate CLTV using following formula
    
    CLTV = ((Average Order Value x Purchase Frequency)/Churn Rate) x Profit margin.
    Customer Value = Average Order Value * Purchase Frequency
    
#### 1. Calculate Average Order Value
```python
uk_data_grp['avg_order_val'] = uk_data_grp['TotalPurchase'] / uk_data_grp['Quantity']
```

#### 2. Calculate Purchase Frequency
```python
purchase_frequency = sum(uk_data_grp['InvoiceNo'])/uk_data_grp.shape[0]
```

#### 3. Calculate Repeat and Churn Rate
```python
repeat_rate=uk_data_grp[uk_data_grp['InvoiceNo'] > 1].shape[0]/uk_data_grp.shape[0]
churn_rate = 1 - repeat_rate

print(purchase_frequency, repeat_rate, churn_rate)
```

```sh
90.37107880642694 0.9818923743942872 0.018107625605712774
```

#### 4. Calculate Profit Margin
Profit margin is the commonly used profitability ratio. It represents how much percentage of total sales has earned as 
the gain. Let's assume our business has approx 5% profit on the total sale.

```python
# Calculate Profit margin assuming gain of 5%
uk_data_grp['profit_margin'] = uk_data_grp['TotalPurchase'] * 0.05
```

#### 5. Calculate Customer Lifetime Value
```python
uk_data_grp['cust_lifetime_value'] = ((uk_data_grp['avg_order_val'] * purchase_frequency)/churn_rate) * \
                                     uk_data_grp['profit_margin']

print(uk_data_grp.sample(n=10))
```
```sh 
            InvoiceDate  InvoiceNo  Quantity  TotalPurchase  avg_order_val  profit_margin  cust_lifetime_value
CustomerID                                                                                                    
13643.0             154         28       273         519.44       1.902711        25.9720         2.466301e+05
17288.0             349        142      1203        1419.73       1.180158        70.9865         4.181036e+05
13381.0             267        172      2029        3639.31       1.793647       181.9655         1.628898e+06
15553.0               0         56       451         437.23       0.969468        21.8615         1.057746e+05
15713.0             222         19       150         356.70       2.378000        17.8350         2.116669e+05
13767.0             371        368      7322       17220.36       2.351866       861.0180         1.010631e+07
17118.0               0         10       154         157.02       1.019610         7.8510         3.995096e+04
13732.0               0         33       498         491.86       0.987671        24.5930         1.212248e+05
14782.0               0          6       114         200.10       1.755263        10.0050         8.764503e+04
17732.0               0         18        93         303.97       3.268495        15.1985         2.479228e+05

```

### Prediction Model for CLTV

Extract month and year from InvoiceDate
```python
uk_data['month_yr'] = uk_data['InvoiceDate'].apply(lambda x: x.strftime('%b-%Y'))
```

The pivot table takes the columns as input, and groups the entries into a two-dimensional table in such a way that 
provides a multidimensional summarization of the data.

```python
sale = uk_data.pivot_table(index=['CustomerID'], columns=['month_yr'], values='TotalPurchase', aggfunc='sum', fill_value=0).reset_index()
sale['CLV'] = sale.iloc[:, 2:].sum(axis=1)

print(sale.sample(n=10))
```

```sh 
month_yr  CustomerID  Apr-2011  Aug-2011  Dec-2010  Dec-2011  Feb-2011  Jan-2011  Jul-2011  Jun-2011  Mar-2011  May-2011  Nov-2011  Oct-2011  Sep-2011
3552         17770.0      0.00      0.00       0.0      0.00      0.00      0.00      0.00       0.0    864.77    278.50      0.00      0.00      0.00
2672         16551.0      0.00    306.10       0.0      0.00      0.00    311.07      0.00       0.0      0.00      0.00      0.00      0.00    304.95
1572         15051.0      0.00      0.00       0.0      0.00      0.00      0.00    377.74       0.0      0.00      0.00    245.68    768.19      0.00
278          13208.0    205.28      0.00       0.0      0.00    168.45      0.00    134.86       0.0      0.00    326.48      0.00    388.83      0.00
3012         17015.0      0.00      0.00       0.0      0.00      0.00      0.00      0.00     524.5      0.00      0.00    272.58      0.00   1045.48
2650         16516.0      0.00      0.00       0.0      0.00      0.00      0.00      0.00       0.0      0.00      0.00    101.70      0.00      0.00
1624         15117.0      0.00    363.33       0.0      0.00      0.00      0.00      0.00       0.0      0.00    312.40    349.75    526.30      0.00
2794         16720.0      0.00      0.00       0.0      0.00      0.00      0.00      0.00       0.0      0.00      0.00    155.24      0.00      0.00
2646         16510.0      0.00      0.00     248.1      0.00      0.00      0.00      0.00       0.0      0.00      0.00      0.00      0.00      0.00
467          13471.0      0.00      0.00       0.0    985.18      0.00      0.00      0.00       0.0      0.00      0.00    604.73    941.67      0.00
```

#### Selecting Feature
We need to divide the given columns into two types of variables dependent(or target variable) and independent variable(or 
feature variables). Select latest 6 month as independent variable.

```python
X=sale[['Dec-2011','Nov-2011', 'Oct-2011','Sep-2011','Aug-2011','Jul-2011']]
y=sale[['CLV']]
```

#### Splitting Data

```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

#### Train and fit using linear regression
```python
# Train and predict the model
LR = LinearRegression()
LR.fit(X_train, y_train)

y_pred_score = LR.score(X_test, y_test)
print("Prediction score is: {}".format(y_pred_score))

y_pred = LR.predict(X_test)
```

#### How well the model fit the data
```python
# Check R-Squared value
print("R-Squared: {}".format(metrics.r2_score(y_test, y_pred)))
```

This model has a higher R-squared (0.91). This model provides a better fit to the data.



