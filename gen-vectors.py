import pandas as pd
import openai
from decouple import config

print("dc: " + str(config('API_OPENAI')))

df = pd.read_csv('processed/short.csv', index_col=0)
df.columns = ['text', 'tokens']

print(df.head())

df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

df.to_csv('processed/embeddings.csv')
df.head()