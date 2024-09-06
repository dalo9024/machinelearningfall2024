#gets match ids from and saves it into a csv
#needed for another API call

#libraries used
import requests
import re
import pandas as pd 
import time

#API url
endpoint3 = 'https://americas.api.riotgames.com/tft/match/v1/matches/by-puuid/'
URL3 = {'api_key':'RGAPI-cce76c81-d89f-4dcd-84e9-38c91de0fb33',
        'start':0,
        'endTime':1725408000,
        'startTime':1724889600,
        'count':20
          }
#new file for game ids
filename3 ="game_ids.csv"

#dataframe of player ids
puuids_df=pd.read_csv("Challenger_puuids.csv")

#headers for csv
MyFILE=open(filename3,"w")
WriteThis5="game_id" + "\n"
MyFILE.write(WriteThis5)
MyFILE.close()

#opens file so new data can be appended
MyFILE=open(filename3, "a")

#loop is needed as endpoint changes for each puuid
#writes puuids to csv
for ID in puuids_df["puuid"]:
    endpoint3 = endpoint3 + str(ID) + "/ids"
    response3=requests.get(endpoint3, URL3)
    #need to slow down due to API restrictions
    time.sleep(0.84)
    game_json = response3.json()
    for items in game_json:
        WriteThis6 = str(items) + "\n"
        MyFILE.write(WriteThis6)                   
    endpoint3 = 'https://americas.api.riotgames.com/tft/match/v1/matches/by-puuid/'
    
MyFILE.close()