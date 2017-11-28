import requests
import json

app_id = 'Your App ID'
app_key = 'Your APP KEY'

language = 'en'

def retrieve_definition(word):
    url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/' + language + '/' + word.lower()
    r = requests.get(url, headers = {'app_id': app_id, 'app_key': app_key})
    json_text = r.json()
    response = json_text['results'][0]['lexicalEntries'][0]['entries'][0]['senses'][0]['definitions']
    return response
