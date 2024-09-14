#creates various plots and tables for EDA

#libraries used
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

##################
#Creates box plots
##################

#data used and cols of interest
df = pd.read_csv('encoded_data.csv')
columns1 = ['gold_left', 'level', 'total_damage_to_players', 'players_eliminated', 'num_traits']
#figure set up
plt.figure(figsize=(12, 8))
#loops through each column name and plots in a subplot
for i, col in enumerate(columns1, 1):
    #creates each subplot
    plt.subplot(2, 3, i) 
    sns.boxplot(y=df[col])
    plt.title(f'{col}')
    #quartiles
    Q1 = df[col].quantile(0.25)
    Q2 = df[col].quantile(0.50)
    Q3 = df[col].quantile(0.75)
    #puts quartiles into the top left corner
    plt.text(0.02, 0.95, f'Q1: {Q1:}\nQ2 (Median): {Q2:}\nQ3: {Q3:}',
             color='black', va='top', ha='left', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
#layout, display, and save to png
plt.tight_layout()
plt.savefig('box_plots.png', format='png', dpi=300)
plt.show()

#####################################
# Creates histograms of distributions
#####################################

#new figure set up
plt.figure(figsize=(18, 12))
columns2 = ['gold_left', 'level', 'total_damage_to_players', 'players_eliminated', 'num_traits', 'placement']
#loops through each column name and plots in a subplot
for i, col in enumerate(columns2, 1):
    plt.subplot(2, 3, i)
    #adjusting bins for certain features that have less than 10 unique values
    if col == 'placement':
        bins = 8
    elif col == 'level':
        bins = 6
    elif col == 'players_eliminated':
        bins = 7
    else:
        bins = 10
    sns.histplot(df[col], bins=bins, color='skyblue', edgecolor='black', linewidth=1, stat='density')
    plt.title(f'Distribution: {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
#layout, display, and save to png
plt.tight_layout()
plt.savefig('feature_distributions.png', format='png', dpi=300)
plt.show()


#############################
#Scatter Plots with Placement
#############################

#new figure set up
plt.figure(figsize=(18, 12))
#loops through each column name and plots in a subplot
for i, col in enumerate(columns1, 1):
    plt.subplot(2, 3, i)  
    #creates scatter plot with slr line
    sns.regplot(x=df[col], y=df['placement'], scatter_kws={'color': 'blue', 'edgecolor': 'black'}, line_kws={'color': 'red'})
    plt.title(f'Scatter Plot: {col} vs Placement')
    plt.xlabel(col)
    plt.ylabel('Placement')

#layout, display, and save to png
plt.tight_layout()
plt.savefig('scatter_plots_with_fit.png', format='png', dpi=300)
plt.show()


#####################
#Correlation Heat Map
#####################

#select needed columns
df_selected = df[columns2]
#correlationmatrix
corr = df_selected.corr()

#newfigure
plt.figure(figsize=(10, 8))
# Create a heatmap of the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=16)

#display and save to png
plt.savefig('correlation_heatmap.png', format='png', dpi=300)
plt.show()

###################################################
#Sumamry statistics of scarce and binary variables
##################################################

################
#Augment Columns
################

augment_cols = df.columns[6:284]
#df for only augments
df_augment = df[augment_cols]
#detects if the entry contains any value greater than 1. Needed as you can have multiple of the same item.
df_binary = (df[augment_cols] > 0).astype(int)
#calculates the percentage of times the entry is present
percent_present = df_binary.mean() * 100
#grabs the top 10 present features
top_10_features = percent_present.nlargest(10).index
#creates a df of the top 10 features
df_top_10 = df_binary[top_10_features]

#calculates the percentage of presence and placement related metrics
summary_data = []
for feature in top_10_features:
    #when the feature is present
    feature_present = df[feature] > 0
    # Calculate the percentage of placements > 4 when the feature is present
    percent_placement_over_4 = (df[feature_present]['placement'] >= 4).mean() * 100
    # Calculate the percentage of placements ≤ 4 when the feature is present
    percent_placement_under_4 = (df[feature_present]['placement'] < 4).mean() * 100
    summary_data.append([feature, percent_present[feature], percent_placement_over_4, percent_placement_under_4])
#put summary into a dataframe
summary_df = pd.DataFrame(summary_data, columns=['Feature', '% Players', 'Top 4 (%)', ' Bottom 4 (%)'])

#wrap long features names
def wrap_text(text, width=4):
    return '\n'.join(wrap(text, width=width))
summary_df['Feature'] = summary_df['Feature'].apply(wrap_text)

#plot summary heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(summary_df.set_index('Feature').T, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Summary Table of Top 10 Auguments Present', fontsize=16)
plt.xlabel('Features')

#display and save to png
plt.savefig('top_10_Augments_summary_table.png', format='png', dpi=300)
plt.show()


##############
#Trait Columns
##############

trait_cols = df.columns[284:311]
# df for only traits
df_trait = df[trait_cols]
#detects if the entry contains any value greater than 1. Needed as you can have multiple of the same item.
df_binary = (df[trait_cols] > 0).astype(int)
#calculates the percentage of times the entry is present
percent_present = df_binary.mean() * 100
#grabs the top 10 present features
top_10_features = percent_present.nlargest(10).index
#creates a df of the top 10 features
df_top_10 = df_binary[top_10_features]

#calculates the percentage of presence and placement related metrics
summary_data = []
for feature in top_10_features:
    #when the feature is present
    feature_present = df[feature] > 0
    #calculates the percentage of placements > 4 when the feature is present
    percent_placement_over_4 = (df[feature_present]['placement'] >= 4).mean() * 100
    #calculates the percentage of placements ≤ 4 when the feature is present
    percent_placement_under_4 = (df[feature_present]['placement'] < 4).mean() * 100
    summary_data.append([feature, percent_present[feature], percent_placement_over_4, percent_placement_under_4])
#put summary into a dataframe
summary_df = pd.DataFrame(summary_data, columns=['Feature', '% Players', 'Top 4 (%)', ' Bottom 4 (%)'])

#wrap long features names
summary_df['Feature'] = summary_df['Feature'].apply(wrap_text)

#plot summary heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(summary_df.set_index('Feature').T, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Summary Table of Top 10 Traits Present', fontsize=16)
plt.xlabel('Features')

#display and save to png
plt.savefig('top_10_traits_summary_table.png', format='png', dpi=300)
plt.show()


##############
#Units Columns
##############

item_cols = df.columns[311:373]
# df for only units
df_unit = df[unit_cols]
#detects if the entry contains any value greater than 1. Needed as you can have multiple of the same item.
df_binary = (df[unit_cols] > 0).astype(int)
#calculates the percentage of times the entry is present
percent_present = df_binary.mean() * 100
#grabs the top 10 present features
top_10_features = percent_present.nlargest(10).index
#creates a df of the top 10 features
df_top_10 = df_binary[top_10_features]

#calculates the percentage of presence and placement related metrics
summary_data = []
for feature in top_10_features:
    #when the feature is present
    feature_present = df[feature] > 0
    #calculates the percentage of placements > 4 when the feature is present
    percent_placement_over_4 = (df[feature_present]['placement'] >= 4).mean() * 100
    #calculates the percentage of placements ≤ 4 when the feature is present
    percent_placement_under_4 = (df[feature_present]['placement'] < 4).mean() * 100
    summary_data.append([feature, percent_present[feature], percent_placement_over_4, percent_placement_under_4])
#put summary into a dataframe
summary_df = pd.DataFrame(summary_data, columns=['Feature', '% Players', 'Top 4 (%)', ' Bottom 4 (%)'])

#wrap long features names
summary_df['Feature'] = summary_df['Feature'].apply(wrap_text)

#plot summary heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(summary_df.set_index('Feature').T, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Summary Table of Top 10 Units')

#display and save to png
plt.savefig('top_10_Units_summary_table.png', format='png', dpi=300)
plt.show()

##############
#Items Columns
##############

item_cols = df.columns[373:]
# df for only items
df_item = df[item_cols]
#detects if the entry contains any value greater than 1. Needed as you can have multiple of the same item.
df_binary = (df[item_cols] > 0).astype(int)
#calculates the percentage of times the entry is present
percent_present = df_binary.mean() * 100
#grabs the top 10 present features
top_10_features = percent_present.nlargest(10).index
#creates a df of the top 10 features
df_top_10 = df_binary[top_10_features]

#calculates the percentage of presence and placement related metrics
summary_data = []
for feature in top_10_features:
    #when the feature is present
    feature_present = df[feature] > 0
    #calculates the percentage of placements > 4 when the feature is present
    percent_placement_over_4 = (df[feature_present]['placement'] >= 4).mean() * 100
    #calculates the percentage of placements ≤ 4 when the feature is present
    percent_placement_under_4 = (df[feature_present]['placement'] < 4).mean() * 100
    summary_data.append([feature, percent_present[feature], percent_placement_over_4, percent_placement_under_4])
#put summary into a dataframe
summary_df = pd.DataFrame(summary_data, columns=['Feature', '% Players', 'Top 4 (%)', ' Bottom 4 (%)'])

#wrap long features names
summary_df['Feature'] = summary_df['Feature'].apply(wrap_text)

#plot summary heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(summary_df.set_index('Feature').T, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Summary Table of Top 10 Items')

#display and save to png
plt.savefig('top_10_items_summary_table.png', format='png', dpi=300)
plt.show()