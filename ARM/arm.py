#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:53:37 2024

@author: daniel_long
"""
#libraries used
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#read in data
df = pd.read_csv('match_data.csv')  

#go from ; separated strings to lists also shorten the strings so the important parts are there
def split_str(text):
    if isinstance(text, str) and text:  # Check if text is a non-empty string
        return [part.split('_')[-1] for part in text.split(';') if '_' in part]
    return []  # Return an empty list for non-strings or empty strings

for col in ['augments', 'traits', 'units', 'items']:
    df[col] = df[col].apply(split_str)

#change placement to categorical data
df['placement'] = df['placement'].apply(lambda x: f'Placement_{x}')

#combine everything into a basket
df['basket_items'] = df[['augments', 'traits', 'units', 'items']].apply(lambda row: sum(row, []), axis=1)
df['basket_items'] = df[['basket_items', 'placement']].apply(lambda row: row['basket_items'] + [row['placement']], axis=1)

#transform data for apriori
te = TransactionEncoder()
te_ary = te.fit(df['basket_items']).transform(df['basket_items'])
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

#apriori
frequent_itemsets = apriori(df_trans, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

#get top 15 for each measure
top_support = rules.sort_values(by='support', ascending=False).head(15)
top_confidence = rules.sort_values(by='confidence', ascending=False).head(15)
top_lift = rules.sort_values(by='lift', ascending=False).head(15)

#display top rules
print("Support")
print(top_support[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("Confidence")
print(top_confidence[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("Lift")
print(top_lift[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

#create network graph
G = nx.DiGraph()

#add edges from rules
def add_edges_from_rules(rules_df):
    for _, rule in rules_df.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        for ant in antecedents:
            for con in consequents:
                G.add_edge(ant, con, weight=rule['support'], 
                           confidence=rule['confidence'], lift=rule['lift'])

#add edges from each set of rules
add_edges_from_rules(top_support)
add_edges_from_rules(top_confidence)
add_edges_from_rules(top_lift)

#color map based on original column
color_map = {
    'augments': 'blue',
    'traits': 'green',
    'items': 'orange',
    'units': 'purple',
    'placement': 'cyan'
}

#list to store node colors
node_colors = []

#colors based on original column
for node in G.nodes:
    for col in color_map.keys():
        exploded_col = df[col].explode().apply(lambda x: x.split('_')[-1] if isinstance(x, str) else x)
        if node in exploded_col.unique():
            node_colors.append(color_map[col])
            break

#adjust edge width based on confidence and node sizes based on node connections
edge_widths = [G[u][v]['confidence'] * 2 for u, v in G.edges]
node_sizes = [G.degree(n) * 200 for n in G.nodes]

plt.figure(figsize=(12, 12))

#layout type and drawing nodes and edges
pos = nx.spring_layout(G, scale = 10, seed = 100)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.3)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')  
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Association Rule Network (Top 15 by Support, Confidence, and Lift)")
plt.show()

