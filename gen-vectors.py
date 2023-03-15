from decouple import config
from datetime import datetime
import time
import pandas as pd
import openai
from ratelimit import limits, sleep_and_retry

now = datetime.now()
openai.api_key = str(config('API_OPENAI'))

batch_size = 600

@sleep_and_retry
@limits(calls=60, period=60)
def create_embedding(input_text, max_retries=30):
    num_retries = 0
    while num_retries < max_retries:
        try:
            return openai.Embedding.create(input=input_text, engine='text-embedding-ada-002')['data'][0]['embedding']
        except Exception as e:
            print(f"Error calling OpenAI embeddings API: {e}")
            num_retries += 1
            time.sleep(1)  # wait for 1 second before retrying
    raise Exception("Failed to call OpenAI embeddings API after multiple retries.")

df = pd.read_csv('processed/short.csv', index_col=0)
df.columns = ['text', 'tokens']

print(df.head())

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size].copy()
    batch['embeddings'] = batch['text'].apply(create_embedding)
    batch.to_csv('processed/embeddings-batch:' + str(i) + '.csv')
    batch.to_csv('processed/embeddings-batch:' + str(i) +  ";rows:" + str(len(batch.index)) + ";time:" + now.strftime("%m-%d-%Y %H:%M:%S") + '.csv')
    print(batch.head())
    