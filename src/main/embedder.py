import openai
import os
import pandas
import pinecone
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

MODEL = 'text-embedding-ada-002'

def embed(text):
    return openai.Embedding.create(
    input=text, engine=MODEL
)

def vectorize():
  # read in song data and extract lyrics
  songs_data = pandas.read_json('/Users/rohanmaheshwari/Desktop/Projects/GPT/lyrical-link/data/songs.json').to_dict(orient='records')
  lyrics = [song['lyrics'] for song in songs_data]

  # transform lyrics text into embeddings
  lyrics_embeddings = embed(lyrics)

  # initialize index to store embeddings
  index_name = 'lyrics-based-semantic-search'
  index_dim = len(lyrics_embeddings['data'][0]['embedding'])
  pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
  # check if index already exists (only create index if not)
  if index_name not in pinecone.list_indexes():
      pinecone.create_index(index_name, dimension=index_dim)
  # connect to index
  index = pinecone.Index(index_name)

  # create vector embedding for each song's lyrics and upsert the song name, lyrics embedding, and original lyrics for each phrase
  batch_size = 32  # process everything in batches of 32
  for i in tqdm(range(0, len(songs_data), batch_size)):
      # set end position of batch
      i_end = min(i + batch_size, len(songs_data))
      # get batch of songs lyrics and title (i.e ['song'])
      songs_batch = songs_data[i: i + batch_size]
      lyrics_batch = [song['lyrics'] for song in songs_batch]
      ids_batch =  [str(n) for n in range(i, i + i_end)]
      # create embeddings
      embeds = [song['embedding'] for song in lyrics_embeddings['data']]
      # prep metadata and upsert batch
      meta_data = [
          {
          'lyrics': song['lyrics'], 
          'title': song['song'], 
          'image': song['image'], 
          'artist': song['artist_mb'],
          'tags': song['tags_mb']
          } 
          for song in songs_batch]
      to_upsert = zip(ids_batch, embeds, meta_data)
      # upsert to Pinecone
      index.upsert(vectors=list(to_upsert))








