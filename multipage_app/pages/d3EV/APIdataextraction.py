import requests
import json
import pandas as pd

sr = requests.session()
#REQUEST YOUR OWN API KEY DO NOT USE MINE
#YOU CAN ONLY REQUEST UP TO 1000 RECORDS PER DAY
result = sr.get("https://developer.nrel.gov/api/alt-fuel-stations/v1.json?api_key=uU9XTsRJa2yNqE5CXBCCqpkBfk5PbiOc5r9YnsMl&fuel_type=ELEC&country=US&access_code=public&restricted_access=false&state=WA")
result.status_code
p = json.loads(result.text)
pull = {}
for i in p['fuel_stations']:
        pull[i['station_name']] ={
            'longitude': i['longitude'],
            'latitude' : i['latitude'],
            'address': i['street_address'],
            'state': i['state'],
            'zip': i['zip'],
            'connector_types': i["ev_connector_types"],
            'cards_accepted': i['cards_accepted'],
            'hours': i['access_days_time']
            }
ip = pd.DataFrame.from_dict(pull, orient='index')
ip.reset_index(inplace = True,drop=True)
ip['faddress'] = ip['address'] + ', ' + ip['state'] + ' ' + ip['zip']
ip = ip[~ip['hours'].str.contains('customer|employee|fleet',na = False,regex=True)]
df = ip.groupby('faddress',as_index= False).agg({'longitude':'mean','latitude':'mean','address':'count'})
df.rename(columns={'address':'station_count'},inplace= True)
df.sort_values('station_count',ascending=False)
df.reset_index(inplace = True, drop = True)
df.to_csv(r'E:\GT MASTERS\CSE 6242\Project\pull.csv', sep = ',', index = False)

