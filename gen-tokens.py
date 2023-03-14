import os
import re
import pandas as pd
import tiktoken


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
                    # print(os.path.join(root, file))
                    content = remove_newlines(f.read())
                    # print(content)
                    
                    content_parts = content.split(';DESC')
                    # print(content_parts[0])
                    # print(content_parts[0].split(':')[1])
                    title = content_parts[0].split(':')[1]
    
                    # print(content_parts[1])
                    content = content_parts[1]
                    content_parts = content.split(';BODY')
                    # print(content_parts[0])
                    # print(content_parts[0].split('=')[1])
                    desc = content_parts[0].split('=')[1]
                    
                    content = content_parts[1]
                    # print(content)
                    body = content[1:]
                    # print(body)
                    
                    if len(body) > 100:
                        body = body[:-4] # remove " END" from end of body
                        # print(body)
                        lf_contents.append((root[7:], title, desc, body))
                        print("Added: " + root[7:] + " > " + title + " > " + os.path.join(root, file))
    return lf_contents

contents = list_files("./text/") #run from in app folder

df = pd.DataFrame(contents, columns = ['site', 'title', 'desc', 'body'])

tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df.body.apply(lambda x: len(tokenizer.encode(x)))

df['combined'] = df.title + "///" + df.body

df.to_csv('processed/scraped.csv')
# df.head()