#removes missing values and values that suggest player gave up
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('match_data.csv')

#table displaying the amount of missing values
#count of missing values
missing_values = df1.isnull().sum()

#table set up
fig, ax = plt.subplots(figsize=(6, 2))  # Set figure size
ax.axis('tight')
ax.axis('off')
table_data = [['Column', 'Missing Values']] + list(zip(missing_values.index, missing_values.values))
table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
#display table
plt.show()

#columns to check for missing values, also removes values that suggest giving up.
columns_data_error= ['match_id', 'participant', 'gold_left', 'level', 'total_damage_to_players', 'players_eliminated', 'augments', 'placement']
columns_giving_up = ['units', 'items']
columns_check_tot = columns_data_error + columns_giving_up
# count rows that have missing values based on error and givng up
total_na_error = df1[columns_data_error].isnull().sum().sum()
total_na_give = df1[columns_giving_up].isnull().sum().sum()
print(total_na_error)
print(total_na_give)
df1_cleaned = df1.dropna(subset=columns_check_tot)
df1_cleaned.to_csv('no_missing_match_data.csv')

#table displaying the amount of missing values after cleaning.
#count of missing values after cleaning
missing_values1 = df1_cleaned.isnull().sum()
#after cleaning table set up
fig, ax = plt.subplots(figsize=(6, 2))  # Set figure size
ax.axis('tight')
ax.axis('off')
table_data = [['Column', 'Missing Values']] + list(zip(missing_values1.index, missing_values1.values))
table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
#display table
plt.show()

#identifies impossible values from the dataset
#numeric columnss for min and max
num_cols = ['gold_left', 'level', 'total_damage_to_players', 'players_eliminated', 'num_traits', 'placement']
#stores min/max values
min_max_vals = df1_cleaned[num_cols].agg(['min','max']).T

#counting the strings in augments and getting the min and max
aug_len = df1_cleaned['augments'].apply(lambda x: len(x.split(';')))
min_aug, max_aug = aug_len.min(), aug_len.max()

#append augment count data to the rest
min_max_vals.loc['Count_Augments'] = [min_aug, max_aug]

#new fig
fig, ax = plt.subplots(figsize=(8, 4))  # Set figure size
ax.axis('tight')
ax.axis('off')

#data for the table
table_data = [['Metric', 'Min', 'Max']] + [[index, row['min'], row['max']] for index, row in min_max_vals.iterrows()]
#createtable
table = ax.table(cellText=table_data, cellLoc='center', loc='center')
plt.show()
