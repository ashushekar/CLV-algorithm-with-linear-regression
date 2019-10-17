import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# To decide display window width on console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 30)

# Read the dataset
data = pd.read_excel('online-retail-data/online-retail.xlsx')

# let us drop duplicates
filteredData = data[['Country', 'CustomerID']].drop_duplicates()

# Top ten country's customer
filteredData.Country.value_counts()[:10].plot(kind='bar')
plt.show(block=False)

# Filter only UK data
uk_data = data[data['Country'] == 'United Kingdom']

# Removing negative quantity values
uk_data = uk_data[uk_data['Quantity'] > 0]
# print(uk_data.describe())

# use only required columns for CLTV
uk_data = uk_data[['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Quantity', 'UnitPrice']]

# calculate total purchase
uk_data['TotalPurchase'] = uk_data['Quantity'] * uk_data['UnitPrice']

# perform aggregate operations
uk_data_grp = uk_data.groupby('CustomerID').agg({
    'InvoiceDate': lambda date: (date.max() - date.min()).days,
    'InvoiceNo': lambda num: len(num),
    'Quantity': lambda quantity: quantity.sum(),
    'TotalPurchase': lambda total_pur: total_pur.sum()
})

# print(uk_data_grp.head())

# Calculate Average Order Value
uk_data_grp['avg_order_val'] = uk_data_grp['TotalPurchase'] / uk_data_grp['Quantity']

# Calculate Purchase Frequency
purchase_frequency = sum(uk_data_grp['InvoiceNo'])/uk_data_grp.shape[0]

# Calculate Repeat and Churn Rate
repeat_rate = uk_data_grp[uk_data_grp['InvoiceNo'] > 1].shape[0]/uk_data_grp.shape[0]
churn_rate = 1 - repeat_rate

# print(purchase_frequency, repeat_rate, churn_rate)

# Calculate Profit margin assuming gain of 5%
uk_data_grp['profit_margin'] = uk_data_grp['TotalPurchase'] * 0.05

# Calculate Customer Lifetime Value
uk_data_grp['cust_lifetime_value'] = ((uk_data_grp['avg_order_val'] * purchase_frequency)/churn_rate) * \
                                     uk_data_grp['profit_margin']

# print(uk_data_grp.sample(n=10))

# Let us apply some prediction model

uk_data['month_yr'] = uk_data['InvoiceDate'].apply(lambda x: x.strftime('%b-%Y'))

sale = uk_data.pivot_table(index=['CustomerID'], columns=['month_yr'], values='TotalPurchase', aggfunc='sum', fill_value=0).reset_index()
sale['CLV'] = sale.iloc[:, 2:].sum(axis=1)

# print(sale.sample(n=10))

# Select dependent and independent variables
X = sale[['Dec-2011', 'Nov-2011', 'Oct-2011', 'Sep-2011', 'Aug-2011', 'Jul-2011']]
y = sale[['CLV']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train and predict the model
LR = LinearRegression()
LR.fit(X_train, y_train)

y_pred_score = LR.score(X_test, y_test)
print("Prediction score is: {}".format(y_pred_score))

