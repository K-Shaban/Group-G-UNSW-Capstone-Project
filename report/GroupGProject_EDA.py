import re

#Import libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from scipy.stats import normaltest
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
import os

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
#sns.set_style('darkgrid')

#loading data, data transformation
#File 1 - NSW forecast demand dataset
data1 = pd.read_csv(r'E:\study&work\unswStudy\DataScienceProject\datasets\projectdata\forecastdemand_nsw.csv')
print(data1.columns.is_unique)
print(data1.shape)
print(data1[data1.duplicated() == True])
data1.head()
#In the dataset1, all columns are unique, no duplicate rows identified.
#Rename 'LASTCHANGED'column to 'Date_Time'; 'FORECASTDEMAND' column to 'Forecast_Demand' to make better
# understanding while analyzing
#Remove non-use columns,for exsample all datasets's location/reginon in NSW
data1 = data1.drop (['REGIONID','DATETIME','PREDISPATCHSEQNO','PERIODID'],axis=1)
#Rename columns for better undstading dataset
data1 = data1.rename(
    columns = {'LASTCHANGED': 'Date_Time','FORECASTDEMAND':'Forecast_Demand'})
print(data1.describe())
data1.info()
#The dataset is now clear to the next step.
# Notice that there are no null values, negative value, Very large/very small values appearing in the dataset.



#File 2 - NSW temperature dataset
data2 = pd.read_csv(r'E:\study&work\unswStudy\DataScienceProject\datasets\projectdata\temp_df_stc.csv')
#data2.columns = data2.columns.str.lower()
print(data2.columns.is_unique)
print(data2.shape)
print(data2[data2.duplicated() == True])
data2.head()
#In the dataset2, all columns are unique, no duplicate row identified.

#Rename 'DATETIME'column to 'Date_Time','TEMPERATURE' column to 'Temperature' to make consistency with other datasets
#Remove non-use columns,for exsample all datasets's location/reginon in NSW
data2 = data2.drop (['LOCATION'],axis=1)
#Rename columns for consistency
data2 = data2.rename(
    columns = {'DATETIME': 'Date_Time','TEMPERATURE':'Temperature'})

#print(data2.describe())
#data2.info()
#An issue was found in analysis, the min temperature has an especially large parameter -9999.
# Noticed these values could be errors. Thus it has been removed from the dataset.
data2= data2.loc[data2['Temperature'] != -9999]
print(data2.describe())
data2.info()
data2= data2.set_index('Date_Time')

data2.index=pd.to_datetime(data2.index)
data2 = data2.loc[data2.index>='01-01-2016']
#winsorizing reduce outliers
p5 = np.percentile(data2['Temperature'],5)
p95 = np.percentile(data2['Temperature'],95)
print('5th percentile:',p5)
print('95th percentile:',p95)
winsorized = np.copy(data2['Temperature'])
winsorized[data2['Temperature']<p5] = p5
winsorized[data2['Temperature']>p95] = p95

data2['Winsorized']= winsorized
data2 = data2.drop (['Temperature'],axis=1)
data2 = data2.rename(
 columns = {'Winsorized': 'Temperature'})

data2_wined=data2
#The dataset is now clear to the next step.
# Notice that there are no null values, negative value, Very large/very small values appearing in the dataset.

def creat_features2(data2):
    data2['hour']=data2.index.hour
    data2['dayofweek']=data2.index.day_of_week
    data2['quarter'] = data2.index.quarter
    data2['month']=data2.index.month
    data2['year']=data2.index.year
    data2['dayofyear']=data2.index.dayofyear
    return data2

data2 = creat_features2(data2)
# Plot each year's time series in its own facet

sns.set_theme(style='ticks')
fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Temperature by Hours')
sns.boxplot(data=data2,x='hour',y='Temperature',palette='coolwarm')

fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Temperature by Weekdays')
sns.boxplot(data=data2,x='dayofweek',y='Temperature',palette='coolwarm')

fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Temperature by Months')
sns.boxplot(data=data2,x='month',y='Temperature',palette='coolwarm')

fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Temperature by Year')
sns.boxplot(data=data2,x='year',y='Temperature',palette='coolwarm') #YlOrBr
sns.despine(offset=10,trim=True)
#plt.show()

g = sns.relplot(
    data=data2,
    x="month", y="Temperature", col='year', hue="year",
    kind="line", palette="coolwarm", linewidth=1, zorder=5,
    col_wrap=3, height=2, aspect=1.5, legend=False,
)

# Iterate over each subplot to customize further
for year, ax in g.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

    # Plot every year's time series in the background
    sns.relplot(
        data=data2, x="month", y="Temperature", hue="year",
        kind='line', palette="coolwarm", linewidth=1#, ax=ax,
    )

# Reduce the frequency of the x axis ticks
ax.set_xticks(ax.get_xticks()[::2])

# Tweak the supporting aspects of the plot
g.set_titles("")
g.set_axis_labels("", "Temperature")
g.tight_layout()
#plt.show()


#File 3 - NSW total demand dataset
data3 = pd.read_csv(r'E:\study&work\unswStudy\DataScienceProject\datasets\projectdata\demand_df_stc.csv')
print(data3.columns.is_unique)
print(data3.shape)
print(data3[data3.duplicated() == True])
data3.head()
#In the dataset2, all columns are unique, no duplicate row identified.

#Rename 'DATETIME'column to 'Date_Time','TOTALDEMAND' column to 'Total_Demand' to make consistency with other datasets
#Remove non-use columns,for exsample all datasets's location/reginon in NSW
data3 = data3.drop (['REGIONID'],axis=1)
#Rename columns for consistency
data3 = data3.rename(
 columns = {'DATETIME': 'Date_Time','TOTALDEMAND':'Total_Demand'})
print(data3.describe())
data3.info()
#The dataset is now clear to the next step.
# Notice that there are no null values, negative value, Very large/very small values appearing in the dataset.

data3= data3.set_index('Date_Time')

data3.index=pd.to_datetime(data3.index)

########
#data3 = data3.loc[data3.index>='01-01-2016']
data3 = data3.loc[data3.index>='01-01-2018']
#winsorizing reduce outliers
p10 = np.percentile(data3['Total_Demand'],5)
p90 = np.percentile(data3['Total_Demand'],95)
print('5th percentile:',p10)
print('95th percentile:',p90)
winsorized = np.copy(data3['Total_Demand'])
winsorized[data3['Total_Demand']<p10] = p10
winsorized[data3['Total_Demand']>p90] = p90

data3['Winsorized']= winsorized
data3 = data3.rename(
 columns = {'Winsorized': 'Demand'})
data3 = data3.drop (['Total_Demand'],axis=1)
#print(data3)
data3_wined=data3

#feature creation: creat time series features based on time series index.
def creat_features(data3):
    data3['hour']=data3.index.hour
    data3['dayofweek']=data3.index.day_of_week
    data3['quarter'] = data3.index.quarter
    data3['month']=data3.index.month
    data3['year']=data3.index.year
    data3['dayofyear']=data3.index.dayofyear
    return data3

data3 = creat_features(data3)

h = sns.relplot(
    data=data3,
    x="hour", y="Demand", col='year', hue="year",
    kind="line", palette="coolwarm", linewidth=3, zorder=5,
    col_wrap=3, height=2, aspect=1.5, legend=False,
)

# Iterate over each subplot to customize further
for year, ax in h.axes_dict.items():

    # Add the title as an annotation within the plot
    ax.text(.8, .85, year, transform=ax.transAxes, fontweight="bold")

    # Plot every year's time series in the background

    sns.relplot(
        data=data3, x="hour", y="Demand", hue="year",
        kind='line', palette="coolwarm", linewidth=1#, ax=ax,
    ).set_axis_labels("", "Daily Electricity Demand")

# Reduce the frequency of the x axis ticks
ax.set_xticks(ax.get_xticks()[::2])

# Tweak the supporting aspects of the plot
h.set_titles("")
h.set_axis_labels("", "Daily Electricity Demand")
h.tight_layout()
#plt.show()


#Visualize feature/Target Relationship
sns.set_theme(style='ticks')
fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Demand by Hours')
sns.boxplot(data=data3,x='hour',y='Demand',palette='coolwarm')

fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Demand by Weekdays')
sns.boxplot(data=data3,x='dayofweek',y='Demand',palette='coolwarm')

fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Demand by Months')
sns.boxplot(data=data3,x='month',y='Demand',palette='coolwarm')

fig,ax = plt.subplots(figsize=(15,5))
ax.set_title('Demand by Year')
sns.boxplot(data=data3,x='year',y='Demand',palette='coolwarm')
sns.despine(offset=10,trim=True)
#plt.show()


#File 4 - NSW Price VS demand 2014-23 dataset
#combine price demand each month datasets to one dataset 'Price_Demand2014_23'
#path = 'E:/study&work/unswStudy/DataScienceProject/datasets/projectdata/PriceDemand'
#excl_list = []
#for file in os.listdir(path):
 #   if file.endswith('.csv'):
  #      print('Loading file {0}.....'.format(file))
   #     excl_list.append(pd.read_csv(os.path.join(path,file)))

#output combined dataset
#excl_list = pd.concat(excl_list, axis=0)
#excl_list.to_csv('Price_Demand2014_23.csv',index=False)

data4 = pd.read_csv(r'E:\study&work\unswStudy\DataScienceProject\datasets\projectdata\Price_Demand2014-23.csv',
                    index_col='SETTLEMENTDATE',parse_dates=True)
#data4.columns = data4.columns.map(lambda x: x.lower())

print(data4.columns.is_unique)
print(data4.shape)
print(data4[data4.duplicated() == True])
data4.head()
#In the dataset4, all columns are unique, no duplicate row identified.
#Rename 'SETTLEMENTDATE'column to 'Date_Time','TOTALDEMAND' column to 'Total_Demand', 'RRP' column to 'Price
# to make consistency with other datasets
#Remove non-use columns,for exsample all datasets's location/reginon in NSW
data4 = data4.drop (['REGION','PERIODTYPE'],axis=1)
#Rename columns for consistency
data4 = data4.rename(
 columns = {'SETTLEMENTDATE': 'Date_Time','TOTALDEMAND':'Total_Demand','RRP':'Recommended_Retail_Price'})
print(data4.describe())
data4.info()

total_demand = data4['Total_Demand'].to_frame()
total_demand['SMA_Monthly'] = total_demand.rolling(147).mean()
total_demand.dropna(inplace=True)
total_demand['CMA_Monthly'] = total_demand['Total_Demand'].expanding(min_periods=147).mean()
total_demand['EWMA_Monthly'] = total_demand['Total_Demand'].ewm(span=147).mean()
total_demand[['Total_Demand', 'SMA_Monthly','CMA_Monthly','EWMA_Monthly']].plot(label='Total_Demand', figsize=(16, 8))
#plt.show()

rrp = data4['Recommended_Retail_Price'].to_frame()
rrp['SMA_Monthly'] = rrp.rolling(147).mean()
rrp.dropna(inplace=True)
rrp['CMA_Monthly'] = rrp['Recommended_Retail_Price'].expanding(min_periods=147).mean()
rrp['EWMA_Monthly'] = rrp['Recommended_Retail_Price'].ewm(span=147).mean()
rrp[['Recommended_Retail_Price', 'SMA_Monthly','CMA_Monthly','EWMA_Monthly']].plot(label='Recommended_Retail_Price', figsize=(16, 8))
#plt.show()


sns.lineplot(data=data4, palette="tab10", linewidth=2.5)
#plt.show()

#File 5 - NSW Ausgrid Electricity Consumption 2016-21 dataset
#path2 = 'E:/study&work/unswStudy/DataScienceProject/datasets/projectdata/ausgrid'
#excl_list2 = []
#for file in os.listdir(path2):
 #   if file.endswith('.csv'):
  #      print('Loading file {0}.....'.format(file))
   #     excl_list2.append(pd.read_csv(os.path.join(path2,file)))
#print(len(excl_list2))

#excl_list2 = pd.concat(excl_list2, axis=0)
#excl_list2.to_csv('AusgridElectricityConsumption2016_21.csv',index=False)
data5 = pd.read_csv(r'E:\study&work\unswStudy\DataScienceProject\datasets\projectdata\AusgridElectricityConsumption2016_21.csv')
print(data5.columns.is_unique)
print(data5.shape)
print(data5[data5.duplicated() == True])
data5.head()
#In the dataset5, all columns are unique, no duplicate row identified.
#Remove non-use columns,for exsample 'Number of Customers'
#Rename to make consistency with other datasets
data5 = data5.drop (['Residential General Supply (MWh)','Residential Off Peak Hot Water (MWh)','Residential No of Off Peak Customers',
                     'Residential Total number of Customers','Number of solar customers Res','Number of solar customers Non-Res',
                     'Solar  Generation capacity (kWp) Res\n (kWp)','Solar  Generation capacity (kWp) Non-Res (kWp)',
                     'Non-residential small sites\n(0-160 MWh pa)Number of Customers','Non-residential med-large sites\n(>160 MWh pa) Number of Customers','Local Government Area'],axis=1)

data5 = data5.rename(
 columns = {'Non-residential small sites\n(0-160 MWh pa) MWh': 'Non-residential small sites(0-160 MWh pa)MWh',
            'Non-residential med-large sites\n(>160 MWh pa)MWh':'Non-residential med-large sites(>160 MWh pa)MWh'})
with pd.option_context('display.max_rows', 10, 'display.max_column', 100):
    print(data5.describe())
data5.info()
#The dataset is now clear to the next step.
# Notice that there are no null values, negative value, Very large/very small values appearing in the dataset.
#transfomation###################################################
data5 = pd.melt(data5,'Year',var_name='Energy')


#File 6 - NSW Energy Generation Statisics 2023 dataset
data6 = pd.read_csv(r'E:\study&work\unswStudy\DataScienceProject\datasets\projectdata\NSW_EnergyGeneration_Statistics_2023_CYTable.csv')
with pd.option_context('display.max_rows', 10, 'display.max_column', 100):
    print(data6)
print(data6.columns.is_unique)
print(data6.shape)
print(data6[data6.duplicated() == True])
data6.head()
print(data6.describe())
data6.info()
non_renewable=data6.iloc[:4]
print(non_renewable)
renewable=data6.iloc[5:11]

f1, (ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8, 1, figsize=(5, 7), sharex=True)
values = non_renewable.groupby('Energy Source').sum().reset_index()
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2015(GWh)'], data=values, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.bar_label(ax1.containers[0])
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2016(GWh)'], data=values, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.bar_label(ax2.containers[0])
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2017(GWh)'], data=values, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.bar_label(ax3.containers[0])
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2018(GWh)'], data=values, palette="rocket", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
ax4.bar_label(ax4.containers[0])
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2019(GWh)'], data=values, palette="rocket", ax=ax5)
ax5.axhline(0, color="k", clip_on=False)
ax5.bar_label(ax5.containers[0])
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2020(GWh)'], data=values, palette="rocket", ax=ax6)
ax6.axhline(0, color="k", clip_on=False)
ax6.bar_label(ax6.containers[0])
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2021(GWh)'], data=values, palette="rocket", ax=ax7)
ax7.axhline(0, color="k", clip_on=False)
ax7.bar_label(ax7.containers[0])
sns.barplot(x=non_renewable['Energy Source'], y=non_renewable['2022(GWh)'],data=values, palette="rocket", ax=ax8)
ax8.axhline(0, color="k", clip_on=False)
ax8.bar_label(ax8.containers[0])
sns.despine(bottom=True)
#plt.show()

f2, (ax1, ax2, ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8, 1, figsize=(5, 7), sharex=True)
values = renewable.groupby('Energy Source').sum().reset_index()
sns.barplot(x=renewable['Energy Source'], y=renewable['2015(GWh)'], data=values, palette="deep", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.bar_label(ax1.containers[0])
sns.barplot(x=renewable['Energy Source'], y=renewable['2016(GWh)'], data=values, palette="deep", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.bar_label(ax2.containers[0])
sns.barplot(x=renewable['Energy Source'], y=renewable['2017(GWh)'], data=values, palette="deep", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.bar_label(ax3.containers[0])
sns.barplot(x=renewable['Energy Source'], y=renewable['2018(GWh)'], data=values, palette="deep", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
ax4.bar_label(ax4.containers[0])
sns.barplot(x=renewable['Energy Source'], y=renewable['2019(GWh)'], data=values, palette="deep", ax=ax5)
ax5.axhline(0, color="k", clip_on=False)
ax5.bar_label(ax5.containers[0])
sns.barplot(x=renewable['Energy Source'], y=renewable['2020(GWh)'], data=values, palette="deep", ax=ax6)
ax6.axhline(0, color="k", clip_on=False)
ax6.bar_label(ax6.containers[0])
sns.barplot(x=renewable['Energy Source'], y=renewable['2021(GWh)'], data=values, palette="deep", ax=ax7)
ax7.axhline(0, color="k", clip_on=False)
ax7.bar_label(ax7.containers[0])
sns.barplot(x=renewable['Energy Source'], y=renewable['2022(GWh)'],data=values, palette="deep", ax=ax8)
ax8.axhline(0, color="k", clip_on=False)
ax8.bar_label(ax8.containers[0])
sns.despine(bottom=True)
plt.show()

#The dataset is now clear to the next step.
# Notice that there are no null values, negative value, Very large/very small values appearing in the dataset.
