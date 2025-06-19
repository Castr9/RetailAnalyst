"""
    Notes:
        Why reset_index?
                Without it, groupby returns a Series  with (Year, Month) as the index, wich Seaborn can't use directly for ploting

        Why hue='Year'?
                It enables multi-line comparsion of sales across years(e.g., to see if December peaks every year).

        Alternatives
                . sns.barplot(for a bar chart instead of lines)
                . add marker='o' to lineplot to highlight data points.                       


        Why barh (horizontal bar)?
                Easier to read long product names on the Y-axis compared to a vertical bar chart.

        Why autopct in pie charts?
                Automatically adds percentages for better readability

        Alternatives:
                Use sns bar.plot() for more syling options(Seaborn).
                Add explode=[0. 1, 0, 0] in plot() to highlight a pie slice                        

        Potential Enchancements:
                top_customers: Add labels to the bars for clarity 
                        Ex: top_customers.plot(kind='barh', figsize=(10, 5))
                            plt.xlabel("Total Sales ($)")


                Repeat customers: Calculate the percentage of repeaters
                        Ex: total_customers = len(customer_orders)
                            print(f"Repeat Customer Rate: {repeat_customers/total_customers:.1%}")

                            
                Regional Sales: Use colors to emphasiz top regions 
                        Ex: region_sales.plot(kind='bar', color=['green', 'blue', 'red', 'orange'])

                        
                Color-coded Scatter Plot: highlight categories/regions;
                        Ex: sns.scatterplot(data=df, x='Sales', y='Profit', hue='Category')

                Add Trendline: Show overall relationship;                    
                        Ex: sns.regplot(data=df, x='Sales', y='Profit', scatter=False, color='red')
                
                Analyzze Loss-Makers Further:
                        Ex: loss_makers.groupby('Category')['Profit'].sum().plot(kind='bar')
                        
                """




import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')#Seting backend to 'Agg'(non-interactive)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA #model time series forecasting
from sklearn.metrics import mean_squared_error #Calculating forecast accuracy
import numpy as np #For numerical operations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#Loading and exploring the data


#Loading the dataset 
df= pd.read_csv("data/Superstore.csv", encoding='latin1')

#Check first 5 rows
print("First 5 Rows - ", df.head())

#Basic info
print("Basic Info - ",df.info())

#Summary stats 
print("Summary Tests - ",df.describe())




#------------------------------------- DATA CLEANING ---------------------------------

#Handling missing values
#Converting date columns to datetime
#Removing duplicates 

#Converting Order Date to Datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

#Extracting year/month for trend analysis
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month_name()


#--------------------------------------- Monthly Revenue -----------------------------------


#Grouping sales data by year and month, then sum the sales for each group
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
plt.figure(figsize=(12, 6))                  # Calculating the sum of the sales for each group
                                             #Ex: All rows with Year=2023 and Month="Jaanuary" are grouped together
                                             #.reset_index Converts the grouped result back into a DataFrame with Year, Month and Sales as columns (instead of multi-index)

sns.lineplot(data=monthly_sales, x='Month', y='Sales', hue='Year')#Using Seaborn lineplot to create a line chart
                                                                  #Data=monthly_sales  to use the grouped data frame
                                                                  #x='Month' seting th x-axis to the Month column
                                                                  #y='Sales' seting y-axis to the sales value
                                                                  #hue='Year' Adding different colored lines for each Year(2022,2023)
plt.title("Monthly Sales Trend")#Giving a title to the chart                                    
plt.xticks(rotation=45)#Rotating the x-axis labels by 45 degrees fr readability
plt.savefig('monthly_sales_trend.png')
plt.show()#Calling the function


#--------------------------------------------Best Selling Products ------------------------------------

#Top 10 products by sales 
                             #Suming sales   
plt.figure(figsize=(10, 6))#Creating figure                             
top_products  = df.groupby('Product Name')['Sales'].sum().nlargest(10)                                                         #selecting the top 10
top_products.plot(kind='barh') #Creating an horizontal bar plot           
plt.title("Top 10 Best-Selling Products")
plt.tight_layout()#Preventing labeloverlap
plt.savefig('top_products.png')
plt.show()

#Sales by category
plt.figure(figsize=(10,6))#Creating new figure
category_sales = df.groupby('Category')['Sales'].sum()#Grouping the data frame by category
category_sales.plot(kind='pie', autopct='%1.1f%%')#Creating a pie chart with percentage
                                #Displays percentages on each slice with 1 decimal pal,ce(eg.,"25,5%")
plt.title("Sales Distribuition by Category")
plt.tight_layout()
plt.savefig('sales_distribuition_category.png')
plt.show()


#-----------------------------------------Customers Behavior Analysis--------------------------------------------
#Top customers by saless
plt.figure(figsize=(10, 6))
top_customers = df.groupby('Customer Name')['Sales'].sum().nlargest(5)
top_customers.plot(kind='barh')                            #Grouping the data frame by the 'Customer Name' column and then sums the sales for each customer
plt.title("Top 5 customers")
plt.tight_layout()                            #And then takes the top 5 customers by sales
plt.savefig("topCustomers.png")


#Repeat customers vs. one-time buyers
customer_orders = df['Customer ID'].value_counts()
                  #Count how many orders each customer made (Using Customer Id.value_counts)  
repeat_customers =  customer_orders[customer_orders > 1].count()
                                    #Filters customers with >1 order and count them 
print(f"Repeat customers: {repeat_customers}")


#Segment by region
plt.figure(figsize=(10, 6))
region_sales = df.groupby('Region')['Sales'].sum()#Group data by 'Region' and sum their sales 
region_sales.plot(kind='barh')#Generates vrtical bar chart
plt.title("Sales By Region")
plt.savefig('SegmentbyRegion.png')
plt.tight_layout()
plt.show()


#------------------------------------------Profitability Analysis-----------------------------------------------------

#Profit vs . Sales scatter plot
#Create a scatter a plot of Profit (Y-axis) against Sales (X - axis)
sns.scatterplot(data=df, x='Sales', y='Profit')
    #User Seaborn to create Scatterplot
    #data=df Uses the data frame  df as the data souorce 
    #x='Sales' sets the sales column for the X-axis
    #y='Profit' sets the profit column fot Y-axis 
"""
    Interpretation
      - Each point represent a transaction 
      - Patterns to look for:
            .Positive trend: Higher sales > Higher profit (ideal)
            .Negative trend: Higher sales > Lower Profit
            .Outliers: High sales but negative proit(critical to investigate)

"""
plt.title("Profit vs. Sales")
plt.savefig('Profit vd')
plt.show()

#Products with negative profit
loss_makers = df[df['Profit'] < 0]
               #Filter rows where profit is negative(losses)
               # #df[df['Profit'] < 0]
               # Filters the data frame to only include rows where the Profit column is negative 
#Count ocurrences of each product name in losss_makers and show top 5
print("Loss Makers - ", loss_makers['Product Name'].value_counts().head(5))
#(loss_makers['Product Name'].value_counts().head(5))
#head(5)
#Returns the top 5 most frequent loss-making subset


#----------------------------------------Advancedd analysis-------------------------------------------------
#RFM(Recency, Frequency, MOneatary) Analysis to segment
#Time Series Forecasting(e.g., predict future sales with ARIMA).
#Correlation Heatmap to find relationships between variables

#Correlation heatmap
corr = df[['Sales', 'Profit','Quantity']].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig('correlaion_heatmap.png')
plt.show()


#-----------------------------------------Forecasting time ---------------------------------------------------

# -----------------------------------------
# Time Series Forecasting (Fixed Version)
# -----------------------------------------

# 1. Prepare time series data
time_series = df.groupby('Order Date')['Sales'].sum()
time_series.index = pd.to_datetime(time_series.index)

# 2. Set explicit frequency and handle missing values
time_series = time_series.asfreq('D').fillna(0)  # Set daily frequency


# 3. Verify data
print(f"Total Data Points: {len(time_series)}")
print(f"Data Range: {time_series.index.min()} to {time_series.index.max}")

# 4. Train-test split (last 30 days as test)
test_size = min(30, len(time_series)//4) #Use 30 days or 25% of data, wichever is smaller
split_date = time_series.index[-test_size]

train = time_series[:split_date]
test = time_series[split_date:]

print(f"\nTrain Size: {len(train)} ({train.index.min()} to {train.index.max()})")
print(f"Test Size: {len(test)} ({test.index.min()} to {test.index.max()})")

# 5. Fit ARIMA model (only once)
try:
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast only as many steps as we have test data
    forecast_steps = len(test)
    forecast_result = model_fit.get_forecast(steps=forecast_steps)
    forecast_values = forecast_result.predicted_mean
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast_values))
    print(f"\nRMSE: {rmse:.2f}")
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Sales')
    plt.plot(test.index, forecast_values, label='Forecast', linestyle='--')
    plt.title(f'Sales Forecast ({forecast_steps}-day prediction)')
    plt.legend()
    plt.savefig('forecast_results.png')
    plt.close()
    
except Exception as e:
    print(f"\nModeling failed: {str(e)}")
    print("Possible solutions:")
    print("- Increase your training data size")
    print("- Reduce the forecast horizon (steps)")
    print("- Check for missing dates in your time series")