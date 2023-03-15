from decouple import config
from datetime import datetime
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

openai.api_key = str(config('API_OPENAI'))

df=pd.read_csv('processed/embed/embed-comb.csv', index_col=0)
df.columns = ['batchid', 'text', 'tokens', 'embeddings']
df.head()
print(df.head())
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

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
        returns.append("Example " + str(i) + ":\n" + row["text"])

    # Return the context
    return "\n\n".join(returns)

def answer_question(
    df,
    question,
    model="text-davinci-003",
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
        max_len=2750,
        size=size,
    )
    
    author = str(config('AUTHOR'))
    
    # If debug, print the raw model response
    #if debug:
        #print(f'% System\n\nYou are a funny sports journalist.  You write humerous articles about trending football news.  You write in the style of {author}, use foul language, and close all of your articles with a witty tagline.  All of your articles have 4 paragraphs as follows:\n\n1. Introduction\n2. Supporting point 1\n3. Supporting point 2\n4. Conclusion\n\nRead and apply the following examples when responding to questions.\n\n% Context\n\n{context}\n\n----\n\n% Question\n%{question}\n\n% Answer\n%')
        # print("Context:\n" + context)
    #    print("\n\n************************************************************************\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f'"""\nYou are a funny sports journalist writing an article based on a prompt.  Write in the style of Bill Burr and use two curse words.  Use the context below to answer the question.  Use this format, replacing text in brackets with the result.  Do not inclued the brackets in the output:\n\nArtilce:\n[Introductory paragraph]\n\n# [Name of Topic 1]\n[Paragraph about topic 1]\n\n[Concluding paragraph]\n\nContext:\n\n{context}"""\n\nQuestion: {question}?\n',
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        
        print("\n\n************************************************************************\n\n")
        print(f'"""\nYou are a funny sports journalist writing an article based on a prompt.  Write in the style of Bill Burr and use two curse words.  Use the context below to answer the question.  Use this format, replacing text in brackets with the result.  Do not inclued the brackets in the output:\n\nArtilce:\n[Introductory paragraph]\n\n# [Name of Topic 1]\n[Paragraph about topic 1]\n\n[Concluding paragraph]\n\nContext:\n\n{context}"""\n\nQuestion: {question}?\n')
        print("\n-------------------------------------------\n")
        print(response["choices"][0]["text"].strip())
        print("\n\n************************************************************************\n\n")
        
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    
print(answer_question(df, question="Should the New England Patriots attempt to sign Aaron Rodgers to be their QB?"))

print(answer_question(df, question="Who was the most valuable player on the New England Patriots during the 2022 season?"))

print(answer_question(df, question="Should the New England Patriots trade for Baltimore Ravens QB Lamar Jackson?"))