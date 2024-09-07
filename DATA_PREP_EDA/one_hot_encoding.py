#one hot encodes each item from the list columns and removes the id columns.

#library used
import pandas as pd

df2 = pd.read_csv('no_missing_match_data.csv')
#columns that need to be one-hot encoded and the id columns that they need to be matched to.
columns_encode = ['augments', 'traits', 'units', 'items']
id_columns = ['participant', 'match_id']  # Grouping by both participant and match_id
#starting encoded df with non list items.
encoded_df = df2[['match_id', 'participant', 'gold_left', 'level', 'total_damage_to_players', 
                 'players_eliminated', 'num_traits', 'placement']].copy()
# Step 3: One-hot encode each of the specified columns
for col in columns_encode:
    #lists in each entry are split based on semi-colon
    df2[col] = df2[col].apply(lambda x: x.split(';') if isinstance(x, str) else x) 
    #seperate rows
    exploded_df = df2.explode(col) 
    #create encoded columns 
    dummies = pd.get_dummies(exploded_df[col])
    # concatenate the one-hot encoded df with id columns 
    exploded_df = pd.concat([exploded_df[id_columns], dummies], axis=1)
    #group by participant and match_id to sum up the encoded values for each unique combination
    grouped = exploded_df.groupby(id_columns).sum().reset_index()
    #merge the grouped data back with the encoded df
    encoded_df = pd.merge(encoded_df, grouped, how='left', on=id_columns)
#drop the identifier columns as they are no longer needed.
encoded_df = encoded_df.drop(columns=['match_id', 'participant'])
#write to csv
encoded_df.to_csv('encoded_data.csv', index=False)  