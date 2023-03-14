import os
import re
import pandas as pd
import tiktoken
from datetime import datetime

max_tokens = 500
now = datetime.now()

def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie

def list_files(startpath):
    lf_contents=[]
    for root, dirs, files in os.walk(startpath):
        for file in files:
            with open(os.path.join(root, file), "r", encoding="UTF-8") as f:
                if not re.search("DS_", f.name):
                    print('***********************************************************')
                    print('processing: ' + os.path.join(root, file))
                    content = remove_newlines(f.read())
                    # print(content)
                    
                    content_parts = content.split(';DESC')
                    if len(content_parts) > 1:
                        print('title: ' + (content_parts[0]))
                        title = (content_parts[0].split(':')[1])
                        if len(title) > 1:
                            title = title.strip()
        
                        # print(content_parts[1])
                        content = content_parts[1]
                        if len(content_parts) > 1:
                            content_parts = content.split(';BODY')
                            # print(content_parts[0])
                            # print(content_parts[0].split('=')[1])
                            desc = (content_parts[0].split('=')[1])
                            
                            content = content_parts[1]
                            # print(content)
                            body = (content[1:]).strip()
                            # print(body)
                            
                            if len(body) > 100:
                                body = body[:-4] # remove " END" from end of body
                                # print(body)
                                lf_contents.append((root[7:], title, desc, body))
                                print("Added: " + root[7:] + " > " + title + " > " + os.path.join(root, file))
    return lf_contents

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

contents = list_files("./text/") #run from in app folder

df = pd.DataFrame(contents, columns = ['site', 'title', 'desc', 'body'])

tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df.body.apply(lambda x: len(tokenizer.encode(x)))

df['combined'] = df.title + "///" + df.body

df.to_csv('processed/combined' + ";rows:" + str(len(df.index)) + ";time:" + now.strftime("%m-%d-%Y %H:%M:%S") + '.csv')
df.to_csv('processed/combined.csv')

shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['body'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['combined'])
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['combined'] )
        
df = pd.DataFrame(shortened, columns = ['combined'])
df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))

df.to_csv('processed/shortened' + ";mtokens:" + str(max_tokens) + ";rows:" + str(len(df.index)) + ";time:" + now.strftime("%m-%d-%Y %H:%M:%S") + '.csv')
df.to_csv('processed/short.csv')

# df.head()