#gets puuids from summoner ids and saves it into a csv
#needed for another API call

#libraries used
import requests
import re
import pandas as pd 
import time

#API url
endpoint2 = 'https://na1.api.riotgames.com/tft/summoner/v1/summoners/'
URL2 = {'api_key':'KEY_HERE'
          }
#new file for puuids
filename2 ="Challenger_puuids.csv"

#dataframe of summonerIDs
summoner_df=pd.read_csv("Challenger_IDS.csv")

#headers for csv
MyFILE=open(filename2,"w")
WriteThis3="puuid" + "\n"
MyFILE.write(WriteThis3)
MyFILE.close()

#opens file so new data can be appended
MyFILE=open("Challenger_puuids.csv", "a")

#loop is needed as endpoint changes for each summoner ID
#writes puuids to csv
for ID in summoner_df["summonerId"]:
    endpoint2 = endpoint2 + str(ID)
    response2=requests.get(endpoint2, URL2)
    #need to slow down due to API restrictions
    time.sleep(1)
    puuid_json = response2.json()
    puuid = puuid_json["puuid"]                     
    WriteThis4=str(puuid)+"\n"
    MyFILE.write(WriteThis4)
    endpoint2 = 'https://na1.api.riotgames.com/tft/summoner/v1/summoners/'
MyFILE.close()
