# Miracle is an alternative dataset for knowledge matching. Stick to BEIR for now.

import datasets

lang='en'  # or any of the 16 languages
miracl = datasets.load_dataset('miracl/miracl', lang, use_auth_token=True, streaming=True)

# training set:
for data in miracl['dev']:  # or 'dev', 'testA'
  print(data['query'])
  query_id = data['query_id']
  query = data['query']
  positive_passages = data['positive_passages']
  negative_passages = data['negative_passages']
  
  for entry in positive_passages: # OR 'negative_passages'
    docid = entry['docid']
    title = entry['title']
    text = entry['text']
    break
  break

