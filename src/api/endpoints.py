import embedder
import sys
import openai
import pinecone
import os
import azapi


MODEL = 'text-davinci-003'
API = azapi.AZlyrics('google', accuracy=0.5)

# Correct Artist and Title are updated from webpage
print(API.title, API.artist)

def get_index():
    index_name = 'lyrics-based-semantic-search'
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
    return pinecone.Index(index_name)

def retrieve(query):
  limit = 3750
  embedded_query = embedder.embed(query)
  # retrieve from Pinecone
  vectorized_embeddding = embedded_query['data'][0]['embedding']
  index = get_index()
  # get relevant contexts (including the questions)
  index_res = index.query(vectorized_embeddding, top_k=10, include_metadata=True)
  contexts = [
        (retrieved_songs['metadata']['title'],retrieved_songs['metadata']['artist']) for retrieved_songs in index_res['matches']
    ]
  return contexts

def find_similar_songs():
    query_song = input('Enter a song name: ')
    API.title = query_song
    API.getLyrics(save=True, ext='lrc')
    query_song_lyrics = API.lyrics
    results = retrieve(query_song_lyrics)

    for (song, artist) in results:
       print('title: ' + song + ' |  artist: ' + artist)
       print('\n')
    

print(find_similar_songs())


    
