import itertools
import time
import pandas as pd
import re
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from nltk.util import ngrams
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from difflib import get_close_matches

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

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

# New function for phrase matching
def phrase_match(query, text):
    query_words = query.lower().split()
    text_words = text.lower().split()
    for i in range(len(text_words) - len(query_words) + 1):
        if text_words[i:i+len(query_words)] == query_words:
            return True
    return False

# Step 4: Search using the inverted index (updated with scoring)
def search_query(inverted_index, query, data):
    tokens = preprocess_text(query)
    if not tokens:
        return []

    matching_indices = set()
    for token in tokens:
        for key in inverted_index.keys():
            if key.startswith(token):
                matching_indices.update(inverted_index[key])
    
    results = []
    for idx in matching_indices:
        entry = data[idx]
        score = 0
        
        # Check for exact phrase match in title, description, and blurb
        if phrase_match(query, entry['title']):
            score += 100
        if phrase_match(query, entry['description']):
            score += 50
        if phrase_match(query, entry['blurb']):
            score += 25
        
        # Add score for individual word matches
        for token in tokens:
            if token in entry['title'].lower():
                score += 10
            if token in entry['description'].lower():
                score += 5
            if token in entry['blurb'].lower():
                score += 2
            if token in entry['keywords']:
                score += 1
        
        results.append((entry, score))
    
    # Sort results by score in descending order
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return [entry for entry, score in sorted_results]

# Step 5: Handle complex queries (updated)
def handle_complex_query(query, inverted_index, data):
    subqueries = re.split(r'\band\b|\bor\b', query)
    all_results = []

    for subquery in subqueries:
        subquery = subquery.strip()
        result_entries = search_query(inverted_index, subquery, data)
        all_results.extend(result_entries)

    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for entry in all_results:
        if entry['title'] not in seen:
            seen.add(entry['title'])
            unique_results.append(entry)
    
    return unique_results

def create_custom_dictionary(data_dict):
    custom_dict = set()
    for entry in data_dict:
        custom_dict.update(preprocess_text(entry['title']))
        custom_dict.update(preprocess_text(entry['description']))
        custom_dict.update(preprocess_text(entry['blurb']))
        custom_dict.update(entry['keywords'])
    
    # Convert to a list for difflib usage
    custom_dict = list(custom_dict)
    
    return custom_dict

def spellchecker(query, inverted_index, data_dict, custom_dictionary):
    tokens = query.lower().split()
    corrected_queries = []
    
    # Single word corrections
    corrected_tokens = []
    for token in tokens:
        if token not in custom_dictionary:
            corrections = get_close_matches(token, custom_dictionary, n=5, cutoff=0.8)
            corrected_tokens.append(corrections + [token])
        else:
            corrected_tokens.append([token])
    
    # Generate all possible combinations of corrected tokens
    for correction in itertools.product(*corrected_tokens):
        corrected_queries.append(' '.join(correction))
    
    # N-gram matching for phrase suggestions
    n = len(tokens)
    while n > 1:
        query_ngrams = list(ngrams(tokens, n))
        for ng in query_ngrams:
            ng_text = ' '.join(ng)
            matches = get_close_matches(ng_text, custom_dictionary, n=2, cutoff=0.8)
            for match in matches:
                corrected_query = query.replace(ng_text, match)
                corrected_queries.append(corrected_query)
        n -= 1
    
    # Add common prefixes if not present
    common_prefixes = ["how to", "where can i", "i want to"]
    for prefix in common_prefixes:
        if not query.startswith(prefix):
            corrected_queries.append(f"{prefix} {query}")
    
    # Remove duplicates and sort by relevance
    unique_queries = list(set(corrected_queries))
    
    # Score queries based on relevance to the data
    scored_queries = []
    for q in unique_queries:
        score = sum(len(search_query(inverted_index, q, data_dict)) for _ in range(3))  # Run search 3 times and sum results
        scored_queries.append((q, score))
    
    # Sort by relevance score and then by similarity to original query
    sorted_queries = sorted(scored_queries, key=lambda x: (-x[1], -fuzz.ratio(x[0], query)))
    
    # Limit to top 5 queries
    sorted_queries = sorted_queries[:5]
    
    return [q[0] for q in sorted_queries]  # Return top 5 most relevant suggestions

# Main execution
if __name__ == "__main__":
    # Read data
    data_dict = read_excel_dict('data_and_ids_g_keys.xlsx', 'Version 2')

    # Build the inverted index
    inverted_index = build_inverted_index(data_dict)

    # Create custom dictionary
    custom_dictionary = create_custom_dictionary(data_dict)

    while True:
        query = input("Enter a query: ")
        query = query.lower()
        if query == "quit":
            print("Quit application.")
            break

        # Spellchecker
        corrected_queries = spellchecker(query, inverted_index, data_dict, custom_dictionary)
        if len(corrected_queries) > 1:
            print("Did you mean:")
            for i, corrected_query in enumerate(corrected_queries, start=1):
                print(f"{i}. {corrected_query}")

        # Handle the original query or one of the corrected queries
        selected_query = input("Select a query number (1-5) or enter '0' for original query: ")
        try:
            selection_int = int(selected_query)
            if 0 <= selection_int <= len(corrected_queries):
                query_to_use = query if selection_int == 0 else corrected_queries[selection_int - 1]
                start_time = time.time()
                result_entries = handle_complex_query(query_to_use, inverted_index, data_dict)
                end_time = time.time()
                print(f"Time taken: {end_time - start_time} seconds")

                # Print numbered results
                if result_entries:
                    for i, entry in enumerate(result_entries):
                        print(f"{i+1}. {entry['title']}")
                else:
                    print("No entries found matching your query.")

            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number or 'quit'.")