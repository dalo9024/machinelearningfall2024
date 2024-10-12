#libraries used
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#read in data
df = pd.read_csv('match_data.csv')

#split the ; seperated lists and keep only the important part
def split_str(text):
    if isinstance(text, str) and text:
        return [part.split('_')[-1] for part in text.split(';') if '_' in part]
    return []  

for col in ['augments', 'traits', 'units', 'items']:
    df[col] = df[col].apply(split_str)

#change placement to string
df['placement'] = df['placement'].apply(lambda x: 'top 4' if x <= 4 else 'bottom 4')

#tag the items so the original column is known
def tag_items(row):
    augments = [f'augment_{item}' for item in row['augments']] if isinstance(row['augments'], list) else []
    traits = [f'trait_{item}' for item in row['traits']] if isinstance(row['traits'], list) else []
    units = [f'unit_{item}' for item in row['units']] if isinstance(row['units'], list) else []
    items = [f'item_{item}' for item in row['items']] if isinstance(row['items'], list) else []
    return augments + traits + units + items

#create transaction data and tag each item
df['basket_items'] = df[['augments', 'traits', 'units', 'items']].apply(tag_items, axis=1)
df['basket_items'] = df[['basket_items', 'placement']].apply(lambda row: row['basket_items'] + [row['placement']], axis=1)
df['basket_items'].to_csv('basket.csv')

#change the data to a list
transactions = df['basket_items'].tolist()


#remove and duplicates 
transactions = [list(set(transaction)) for transaction in transactions]


#create one hot encoding array
all_items = sorted(set(item for transaction in transactions for item in transaction))
num_transactions = len(transactions)
one_hot_array = np.zeros((num_transactions, len(all_items)), dtype=bool)

for i, transaction in enumerate(transactions):
    one_hot_array[i, [all_items.index(item) for item in transaction]] = True

#convert to dataframe for apriori()
one_hot_df = pd.DataFrame(one_hot_array, columns=all_items)
one_hot_df.to_csv('arm_data.csv')

#find frequent itemsets
frequent_itemsets = apriori(one_hot_df, min_support=0.1, use_colnames=True)

#get association rules above confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

#top 15 rules based on measures
top_support = rules.nlargest(15, 'support')
top_confidence = rules.nlargest(15, 'confidence')
top_lift = rules.nlargest(15, 'lift')

top_support.to_csv('top_support.csv', index=False)
top_confidence.to_csv('top_confidence.csv', index=False)
top_lift.to_csv('top_lift.csv', index=False)


print("Top 15 Rules by Support:")
print(top_support[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print("\nTop 15 Rules by Confidence:")
print(top_confidence[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print("\nTop 15 Rules by Lift:")
print(top_lift[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

#combine all the rules to a df
combined_rules = pd.concat([top_support, top_confidence, top_lift]).drop_duplicates()

#create graph
G = nx.DiGraph()

#add edges for the rules
for _, rule in combined_rules.iterrows():
    antecedents = list(rule['antecedents'])
    consequents = list(rule['consequents'])
    for ant in antecedents:
        for con in consequents:
            G.add_edge(ant, con, weight=rule['support'])

#color based on original column
color_map = {
    'augments': 'blue',
    'traits': 'green',
    'items': 'orange',
    'units': 'purple',
    'placement': 'cyan'
}

#stores node colors
node_colors = []

#assign colors to nodes
for node in G.nodes:
    if 'augment_' in node:
        node_colors.append(color_map['augments'])
    elif 'trait_' in node:
        node_colors.append(color_map['traits'])
    elif 'unit_' in node:
        node_colors.append(color_map['units'])
    elif 'item_' in node:
        node_colors.append(color_map['items'])
    elif 'Placement_' in node:
        node_colors.append(color_map['placement'])

#draw network graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k = 0.5, seed=42) 

#node size based on amount of connections
node_sizes = [G.degree(n) * 50 for n in G.nodes]
 
#draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,alpha=0.6)
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Association Rule Network (Top 15 by Support, Confidence, and Lift)")
plt.axis('off')  
