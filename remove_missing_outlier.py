#removes missing values and values that suggest player gave up
import pandas as pd

df1 = pd.read_csv('match_data_test.csv')

#columns to check for missing values, also removes values that suggest giving up.
columns_data_error= ['match_id', 'participant', 'gold_left', 'level', 'total_damage_to_players', 'players_eliminated', 'augments', 'placement']
columns_giving_up = ['units', 'items']
columns_check_tot = columns_data_error + columns_giving_up
#rows with missing values
rows_missing_val_error = df1[df1[columns_data_error].isna().any(axis=1)]
rows_missing_val_give = df1[df1[columns_giving_up].isna().any(axis=1)]
#number of rows with missing values
num_rows_error = len(rows_missing_val_error)
num_rows_give = len(rows_missing_val_give)
print(num_rows_error)
print(num_rows_give)
df1_cleaned = df1.dropna(subset=columns_check_tot)
