import pandas as pd
import re
from collections import defaultdict
import json
import time

# Step 1: Read and preprocess the data from the Excel file
read_excel_dict1 = lambda file, sheet_name: [
    {
        "title": row.pop("Title").lower(),
        "route": row.pop("Route").lower() if pd.notna(row.get("Route")) else '',
        "description": row.pop("Description").lower(),
        "blurb": row.pop("Blurb").lower()
    }
    for row in pd.read_excel(file, sheet_name=sheet_name).to_dict(orient='records')
]

read_excel_dict = lambda file, sheet_name: [
    {
        "title": str(row.pop("Title")).lower(),
        "route": str(row.pop("Route")).lower() if pd.notna(row.get("Route")) else '',
        "description": str(row.pop("Description")).lower(),
        "blurb": str(row.pop("Blurb")).lower()
    }
    for row in pd.read_excel(file, sheet_name=sheet_name).to_dict(orient='records')
]

# Read data
data_dict = read_excel_dict('data2.xlsx', 'Version 2')


def parse_and_transform_new_data(file):
    with open(file, 'r') as f:
        data = f.read()
        
    # Using regular expressions to extract dictionary-like strings
    pattern = r'{[^}]*}'
    matches = re.findall(pattern, data)
    
    # Converting matches to list of dictionaries and transforming keys to lowercase
    new_data = [{k.lower(): v.lower() if isinstance(v, str) else v for k, v in json.loads(match).items()} for match in matches]
    
    return new_data

data_dict = data_dict + parse_and_transform_new_data('id_data.txt')

#not necessary
def read_and_preprocess_data(self, file, sheet_name):
        df = pd.read_excel(file, sheet_name=sheet_name)
        df = df.applymap(lambda x: x.lower() if pd.notna(x) else '')
        dict_df = df.to_dict(orient='records')
        extra_data = parse_and_transform_new_data('id_data.txt')
        #extra_data = get_all_models_for_search()
        #for item in extra_data:
        #    for key, value in item.items():
        #        item[key] = value.lower() if pd.notna(value) else ''
        #dict_df.extend(extra_data)
        combined_data = dict_df + extra_data
        #return dict_df
        return combined_data


# Define stop words
stop_words = set([
    "i", "want", "to", "a", "and", "the", "is", "in", "it", "of", "for", "on", "with", "as", "this", "that"
])

# Step 2: Preprocess text
def preprocess_text(text):
    if text is None:
        return []
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize and remove stop words
    tokens = [word for word in text.split() if word not in stop_words]
    return tokens

# Step 3: Build the inverted index
def build_inverted_index(data):
    inverted_index = defaultdict(list)
    for idx, entry in enumerate(data):
        for key in entry:
            if key != "route":  # Exclude route from indexing
                tokens = preprocess_text(entry[key])
                for token in tokens:
                    if idx not in inverted_index[token]:
                        inverted_index[token].append(idx)
    return inverted_index

# Build the inverted index
inverted_index = build_inverted_index(data_dict)

# Step 4: Search using the inverted index
def search_query(inverted_index, query, data):
    tokens = preprocess_text(query)
    if not tokens:
        return []

    result_docs = set(inverted_index[tokens[0]])
    for token in tokens[1:]:
        result_docs &= set(inverted_index[token])
    
    return [data[idx] for idx in result_docs]

# Example usage
while(True):
  start_time = time.time()
  query = input("Enter a query: ")
  query = query.lower()
  if(query=="quit"):
      print("quit application")
      break
  start_time = time.time()
  result_entries = search_query(inverted_index, query, data_dict)
  end_time = time.time()
  print(f"Time taken to load and convert: {end_time - start_time} seconds")
  print(f"Entries matching '{query}':")
  
  for entry in result_entries:
      print(entry)
