from pymongo import MongoClient, GEOSPHERE
from langchain_core.tools import tool
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

def get_database():
   user = os.getenv('DB_USER')
   password = os.getenv('DB_PASS')
   CONNECTION_STRING = f"mongodb+srv://{user}:{password}@voicebot.2ausgre.mongodb.net/?retryWrites=true&w=majority&appName=VoiceBot"
   client = MongoClient(CONNECTION_STRING)
   return client['VoiceBot']


def get_geo_tool():
    db = get_database()
    col = db['places']
    api = os.getenv('GEO_API_KEY') # API get for google geocoding

    @tool
    def geo_tool(location : str) -> str:
        """Searches the Mongo DB to find nearest branches.

        **Parameters:**
        - location (str) : The location of the user. Example : Siruseri, Chennai, Tamil Nadu

        **Returns:**
        - list : Of strings containing branch locations.
        """

        print('Geocoding Tool invoked!')

        location = location.replace(' ', '%20')
        link = f'https://maps.googleapis.com/maps/api/geocode/json?key={api}&address={location}'
        res = requests.get(link)

        try:
            data = json.loads(res.text)
        except Exception as e:
            print(f'Error in geocoding tool : {e}')

        # print('Status:', data['status'])

        if data['status'] == 'OK':
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
        else:
            print('Status not OK')
            return 'No locations nearby!'
        
        query = {
            "location": {
                "$near": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [lat, lng]
                    },
                    "$maxDistance": 5000
                }
            }
        }

        nearby_places = col.find(query)

        results = [place['address'] for place in nearby_places]
        print('Locations:', results)

        return '\n'.join(results)

    return geo_tool

        
        

    
