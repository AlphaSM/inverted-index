import time
import pandas as pd
import re
from collections import defaultdict

# Step 1: Read and preprocess the data from the Excel file
def read_excel_dict(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    data = df.to_dict(orient='records')
    processed_data = [
        {
            "title": str(row.get("Title", "")).lower(),
            "route": str(row.get("Route", "")).lower() if pd.notna(row.get("Route")) else '',
            "description": str(row.get("Description", "")).lower(),
            "blurb": str(row.get("Blurb", "")).lower(),
            "keywords": [k.strip().lower() for k in str(row.get("Keywords", "")).split(',')]
        }
        for row in data
    ]
    return processed_data

# Read data
data_dict = read_excel_dict('data_and_ids_g_keys.xlsx', 'Version 2')

# Define stop words
stop_words = set([
    "i", "want", "to", "a", "and", "the", "is", "in", "it", "of", "for", "on", "with", "as", "this", "that", "view"
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
            if key == "keywords":
                tokens = entry[key]  # Keywords are already tokenized and lowercased
            else:
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

    # Find all entries matching any token partially
    matching_indices = set()
    for token in tokens:
        for key in inverted_index.keys():
            if key.startswith(token):
                matching_indices.update(inverted_index[key])
    
    return [data[idx] for idx in matching_indices]

# Step 5: Handle complex queries
def handle_complex_query(query, inverted_index, data):
    # Split the query by conjunctions like 'and', 'or'
    subqueries = re.split(r'\band\b|\bor\b', query)
    all_results = []

    for subquery in subqueries:
        subquery = subquery.strip()
        result_entries = search_query(inverted_index, subquery, data)
        all_results.extend(result_entries)

    # Remove duplicates
    unique_results = {entry['title']: entry for entry in all_results}.values()
    return list(unique_results)

# Example usage
while True:
    query = input("Enter a query: ")
    query = query.lower()
    if query == "quit":
        print("quit application")
        break

    start_time = time.time()
    result_entries = handle_complex_query(query, inverted_index, data_dict)
    end_time = time.time()
    print(f"Time taken to load and convert: {end_time - start_time} seconds")

    # Print numbered results
    if result_entries:
        for i, entry in enumerate(result_entries):
            print(f"{i+1}. {entry}")
    else:
        print("No entries found matching your query.")

    # User selection
    selection = input("Enter the number of the entry you want to see or 'quit': ")
    try:
        selection_int = int(selection)
        if 1 <= selection_int <= len(result_entries):
            # Display details of selected entry (optional)
            selected_entry = result_entries[selection_int - 1]
            print(f"\nSelected Entry:\n{selected_entry}")
        else:
            print("Invalid selection number.")
    except ValueError:
        if selection.lower() == "quit":
            break
        else:
            print("Invalid input. Please enter a number or 'quit'.")
