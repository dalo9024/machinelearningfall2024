#gets match info from game_id and saves it into a csv

#libraries used
import requests
import re
import pandas as pd 
import time

#API url
endpoint4 = 'https://americas.api.riotgames.com/tft/match/v1/matches/'
URL4 = {'api_key':'KEY_HERE'
          }
#filename for match data
filename4 = "match_data.csv"

#dataframe for match ids
match_df=pd.read_csv("game_ids.csv")
#dropping duplicate values
match_df.drop_duplicates(inplace=True)

#headers for csv
MyFILE=open(filename4, "w")
WriteThis7 = "match_id, participant, gold_left, level, total_damage_to_players, players_eliminated, num_traits, augments, traits, units,items, placement" + "\n"
MyFILE.write(WriteThis7)
MyFILE.close()

#opens file for append
MyFILE = open(filename4, "a")

#needed for list items before writting to csv
separator = ';'
#loop is needed as endpoint changes for each gameid
#writes puuids to csv
for ID in match_df["game_id"]:
    endpoint4 = endpoint4 + str(ID)
    response4=requests.get(endpoint4, URL4)
    #need to slow down due to API restrictions
    time.sleep(1)
    matchinfo_json = response4.json()
    #writes all the data we are interested in to csv
    for items in matchinfo_json["info"]["participants"]:
        #match id
        match_id = matchinfo_json["metadata"]["match_id"]
        #player id
        participant = items["puuid"]
        #gets left over gold
        gold_left = items["gold_left"]
        #gets level
        level = items["level"]
        #gets total damage to players
        total_damage_to_players = items["total_damage_to_players"]
        #gets amount of players eliminated
        players_eliminated = items["players_eliminated"]
        #gets total number of traits
        num_traits = 0
        for trait1 in items["traits"]:
            if trait1["tier_current"] > 0:
                num_traits = num_traits+1
        #gets augments
        aug_list = items["augments"]
        aug = separator.join(aug_list)
        #gets final placement
        placement = items["placement"]
        #gets traits
        trait_list = []
        for trait2 in items["traits"]:
            trait_list.append(trait2["name"])
        trait = separator.join(trait_list)
        #gets units and items
        unit_list = []
        item_list = []
        for units in items["units"]:
            unit_list.append(units["character_id"])
            item_list.extend(units["itemNames"])
        item = separator.join(item_list)
        unit = separator.join(unit_list)
        #writes it all to csv
        WriteThis8= str(match_id) + "," + str(participant) + "," + str(gold_left) + "," + str(level) + "," + str(total_damage_to_players) + "," + str(players_eliminated) + "," + str(num_traits) + "," + str(aug) + "," + str(trait) + "," + str(unit) + "," + str(item) + "," + str(placement) + "\n"
        MyFILE.write(WriteThis8)                   
    endpoint4 = 'https://americas.api.riotgames.com/tft/match/v1/matches/'
MyFILE.close()
