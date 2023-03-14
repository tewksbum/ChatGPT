from decouple import config
from datetime import datetime
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

openai.api_key = str(config('API_OPENAI'))

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df.columns = ['text', 'tokens', 'embeddings']
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()

def create_context(
    question, df, max_len, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    question,
    model="text-davinci-003",
    max_len=5000,
    size="ada",
    debug=str(config('BOOL_DEBUG')),
    max_tokens=1000,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    
answer_question(df, question="Should the New England Patriots attempt to sign Aaron Rodgers to be their QB?")

answer_question(df, question="Who was the most valuable player on the New England Patriots during the 2022 season?")

answer_question(df, question="Should the New England Patriots trade for Baltimore Ravens QB Lamar Jackson?")