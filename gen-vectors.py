from decouple import config
from datetime import datetime
import pandas as pd
import openai
from ratelimit import limits, sleep_and_retry

now = datetime.now()
openai.api_key = str(config('API_OPENAI'))

batch_size = 1000

@sleep_and_retry
@limits(calls=60, period=60)
def create_embedding(input_text):
    return openai.Embedding.create(input=input_text, engine='text-embedding-ada-002')['data'][0]['embedding']

df = pd.read_csv('processed/short.csv', index_col=0)
df.columns = ['text', 'tokens']

print(df.head())

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    while True:
        try:
            batch['embeddings'] = batch['text'].apply(create_embedding)
            batch.to_csv('processed/embeddings-' + str(batch) + '.csv')
            print(batch.head())
        except:
            continue
        break
    # for index, row in batch.iterrows():
    #     print(row)

# df['embeddings'] = df['text'].apply(create_embedding)

# print(df.head())

#df.to_csv('processed/embeddings.csv')
#df.to_csv('processed/embeddings' + ";rows:" + str(len(df.index)) + ";time:" + now.strftime("%m-%d-%Y %H:%M:%S") + '.csv')
