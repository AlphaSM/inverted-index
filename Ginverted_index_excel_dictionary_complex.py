import pandas as pd
import re
from collections import defaultdict

# Step 1: Read and preprocess the data from the Excel file
read_excel_dict = lambda file, sheet_name: [
    {
        "title": row.pop("Title").lower(),
        "route": row.pop("Route").lower() if pd.notna(row.get("Route")) else '',
        "description": row.pop("Description").lower(),
        "blurb": row.pop("Blurb").lower()
    }
    for row in pd.read_excel(file, sheet_name=sheet_name).to_dict(orient='records')
]

# Read data
data_dict = read_excel_dict('SemanticSearch.xlsx', 'Version 2')

# Define stop words
stop_words = set([
    "i", "want", "to", "a", "and", "the", "is", "in", "it", "of", "for", "on", "with", "as", "this", "that"
])

# Step 2: Preprocess text
def preprocess_text(text):
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
  query = input("Enter a query: ")
  query = query.lower()
  if(query=="quit"):
      print("quit application")
      break
  result_entries = search_query(inverted_index, query, data_dict)

  print(f"Entries matching '{query}':")
  for entry in result_entries:
      print(entry)
