#reads challenger ranked summoner ids from the riotgames API and puts the list into a csv.
#needed for another API call

#libraries used
import requests
import re
import pandas as pd 
import time

#API url
endpoint1="https://na1.api.riotgames.com/tft/league/v1/challenger"
URL1 = {'api_key':'KEY_HERE',
                    'queue': 'RANKED_TFT'
          }

#calls API and reads json
response1=requests.get(endpoint1, URL1)
challenger_json = response1.json()

#new file for summoner IDs
filename1 ="Challenger_IDS.csv"

#writes headers for csv
MyFILE=open(filename1,"w")
WriteThis1="summonerId" + "\n"
MyFILE.write(WriteThis1)
MyFILE.close()

#writes in summoner ids into csv.
MyFILE=open("Challenger_IDS.csv", "a")

for items in challenger_json["entries"]:
    ID = items["summonerId"]                     
    WriteThis2=ID+"\n"
    MyFILE.write(WriteThis2)
MyFILE.close()