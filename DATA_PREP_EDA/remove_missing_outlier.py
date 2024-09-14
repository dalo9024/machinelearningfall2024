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