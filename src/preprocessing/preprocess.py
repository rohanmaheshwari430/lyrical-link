import pandas as pd
import json

# read the CSV file
df = pd.read_csv('https://raw.githubusercontent.com/rohanmaheshwari430/Songmilarity/master/exploration/songs.csv')

# select the columns we want to include in the JSON file
columns = ['mbid', 'artist_mb', 'artist_lastfm', 'tags_mb', 'az_artist', 'song', 'lyrics', 'image']
df = df[columns]

# convert the DataFrame to a dictionary
data = df.to_dict(orient='records')

# write the data to a JSON file
with open('../../data/songs.json', 'w') as outfile:
    json.dump(data, outfile)


