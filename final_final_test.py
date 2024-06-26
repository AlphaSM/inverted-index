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
#nltk.download('punkt')
#nltk.download('wordnet')

import random
import itertools
import pandas as pd
from difflib import SequenceMatcher
import re
from collections import defaultdict
import time

# Create a trie-like structure for faster prefix matching
trie = defaultdict(set)

def add_to_trie(word, file_path):
    word = word.lower()
    for i in range(len(word)):
        trie[word[:i+1]].add(word)

def add_file_to_index(file_path, index):
    with open(file_path, 'r') as f:
        text = f.read()
    tokens = re.findall(r'\w+', text.lower())
    for token in tokens:
        if token not in index:
            index[token] = []
        index[token].append(file_path)
        add_to_trie(token, file_path)

inverted_index = {}
add_file_to_index('words.txt', inverted_index)

# Dictionary
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

data_dict = read_excel_dict('data_and_ids_g_keys.xlsx', 'Version 2')
my_dict = ' '.join([entry['title'] + ' ' + ' '.join(entry['keywords']) for entry in data_dict])

def search_words2(user_input, exclude_list=[]):
    words_list = my_dict.split()
    search_substring = user_input.lower()
    result = [word for word in words_list if search_substring in word.lower() and word.lower() not in exclude_list]
    ranked_result = sorted(result, key=lambda x: similarity(search_substring, x), reverse=True)[:3]  # Limit to top 3 results
    if not ranked_result:
        return None
    else:
        return ranked_result

def similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def search_inverted_index(word, index, limit=3):
    word_lower = word.lower()
    matches = trie[word_lower]
    return list(matches)[:limit]

def get_user_selection(word, suggestions):
    shown_suggestions = suggestions
    exclude_list = set()
    while True:
        print(f"Suggestions for '{word}':")
        for idx, suggestion in enumerate(shown_suggestions, start=1):
            print(f"{idx}. {suggestion}")
        print(f"{len(shown_suggestions) + 1}. Keep the original word '{word}'")
        print(f"{len(shown_suggestions) + 2}. Get new autocorrect suggestions")
        print(f"{len(shown_suggestions) + 3}. Reorder letters and search")
        print(f"{len(shown_suggestions) + 4}. Suggest word with highest letter match percentage")
        print(f"{len(shown_suggestions) + 5}. Go back")

        selection = input(f"Select the number of the best suggestion for '{word}' or press 'q' to quit: ")
        if selection == 'q':
            return None
        try:
            selection = int(selection)
            if selection == len(shown_suggestions) + 1:
                return word
            elif selection == len(shown_suggestions) + 2:
                exclude_list.update(shown_suggestions)
                new_suggestions = autocorrect(word, inverted_index, exclude_list=exclude_list)
                if not new_suggestions:
                    print(f"No new autocorrect suggestions found for '{word}'. Keeping the original word.")
                    return word
                shown_suggestions = new_suggestions
            elif selection == len(shown_suggestions) + 3:
                reorder_count = 0
                while True:
                    reordered_matches = reorder_and_search(word, inverted_index, my_dict)
                    if not reordered_matches:
                        print(f"No matches found by reordering letters of '{word}'. Keeping the original word.")
                        return word
                    print(f"Reordered '{word}' suggestions: {reordered_matches}")
                    reorder_count += 1

                    while True:
                        new_word_selection = input(f"Select the number of the reordered word to search (1-{len(reordered_matches)}) or press 'q' to quit or 'r' to reorder again: ")
                        if new_word_selection == 'q':
                            return word
                        elif new_word_selection == 'r':
                            break
                        try:
                            new_word_selection = int(new_word_selection)
                            if 1 <= new_word_selection <= len(reordered_matches):
                                new_word = reordered_matches[new_word_selection - 1]
                                print(f"Searching for '{new_word}'...")
                                new_suggestions = search_words2(new_word)
                                if new_suggestions is None:
                                    new_suggestions = autocorrect(new_word, inverted_index)
                                shown_suggestions = new_suggestions
                                corrected_string = new_word  
                                return corrected_string
                            else:
                                print("Invalid selection. Please try again.")
                        except ValueError:
                            print("Invalid input. Please enter a number or 'q' or 'r'.")
                    if new_word_selection == 'r':
                        if reorder_count >= 3:
                            print("Maximum reordering attempts reached. Returning original word.")
                            return word
                        continue
                    else:
                        break
            elif selection == len(shown_suggestions) + 4:
                letter_match_word = suggest_highest_letter_match(word)
                print(f"Suggested word with highest letter match percentage: {letter_match_word}")
                return letter_match_word
            elif selection == len(shown_suggestions) + 5:
                return None  # Go back
            elif 1 <= selection <= len(shown_suggestions):
                return shown_suggestions[selection - 1]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

def suggest_highest_letter_match(word):
    word_lower = word.lower()
    possible_words = my_dict.split()
    best_match = ""
    highest_percentage = 0

    for possible_word in possible_words:
        match_percentage = calculate_letter_match_percentage(word_lower, possible_word)
        if match_percentage > highest_percentage:
            highest_percentage = match_percentage
            best_match = possible_word
    
    return best_match

def calculate_letter_match_percentage(word1, word2):
    matches = sum((word1.count(char) == word2.count(char)) for char in set(word1))
    return matches / len(word1) if len(word1) > 0 else 0

def process_user_input(user_input):
    user_words = user_input.split()
    final_results = {}

    for word in user_words:
        # Remove punctuation
        punctuation = ''
        if word[-1] in [',', '.', '!', '?']:
            punctuation = word[-1]
            word = word[:-1]

        result = search_words2(word)
        if result is None:
            print(f"No matches found for '{word}'. Searching for autocorrect suggestions...")
            autocorrect_results = autocorrect(word, inverted_index)
            if word in autocorrect_results:
                print(f"'{word}' is already correct.")
                final_results[word + punctuation] = [word + punctuation]
            else:
                print(f"Autocorrect suggestions for '{word}': {autocorrect_results}")
                selected_word = get_user_selection(word, autocorrect_results)
                if selected_word:
                    final_results[word + punctuation] = [selected_word + punctuation]
                else:
                    final_results[word + punctuation] = [word + punctuation]  # Keep original if no selection
        else:
            final_results[word + punctuation] = result
    
    return final_results

# Use a more efficient algorithm for edit distance
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# Replace edit_distance with levenshtein_distance
def autocorrect(word, index, exclude_list=[], limit=3):
    word_lower = word.lower()
    possible_words = [w for w in index.keys() if w.startswith(word_lower[0])]
    possible_words.sort(key=lambda w: levenshtein_distance(word_lower, w))
    closest_words = [w for w in possible_words if levenshtein_distance(word_lower, w) <= 2 and w not in exclude_list]
    return closest_words[:limit]

def reorder_and_search(word, index, my_dict):
    start_time = time.time()
    word_lower = word.lower()
    permutations = itertools.permutations(word_lower)
    index_matches = []
    my_dict_matches = []
    counter = 0
    
    print("Reordered permutations:")
    for p in permutations:
        perm = ''.join(p)
        print(perm)  # Print the permutation
        if perm in index:
            index_matches.append(perm)
        if perm in my_dict.split():
            my_dict_matches.append(perm)
        counter += 1
        if counter % 3 == 0:
            if index_matches or my_dict_matches:
                break

    all_matches = list(set(index_matches + my_dict_matches))
    end_time = time.time()
    print(f"Reordering took {end_time - start_time} seconds")
    return all_matches[:3]

def get_corrected_string(process_user_input):
    while True:
        user_in = input("Enter a substring to search or press 'q' to quit: ")
        if user_in.lower() == 'q':
            return None

        result = process_user_input(user_in)

        if not result:
            print("No results found.")
        else:
            corrected_string = []
            for word, results in result.items():
                corrected_string.append(results[0])  # Add the selected or original word

            corrected_string = ' '.join(corrected_string)
            print("Searching "+corrected_string)
            return corrected_string
        
        

"""
while True:
    user_in = input("Enter a substring to search or press 'q' to quit: ")
    if user_in.lower() == 'q':
        break

    result = process_user_input(user_in)

    if not result:
        print("No results found.")
    else:
        corrected_string = []
        for word, results in result.items():
            corrected_string.append(results[0])  # Add the selected or original word

        corrected_string = ' '.join(corrected_string)
        print(f"Corrected string: {corrected_string}")
"""

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


# Main execution
if __name__ == "__main__":
    # Read data
    data_dict = read_excel_dict('data_and_ids_g_keys.xlsx', 'Version 2')

    # Build the inverted index
    inverted_index = build_inverted_index(data_dict)

    while True:
        ###
        """
        query = input("Enter a query: ")

        query = query.lower()
        if query == "quit":
            print("Quit application.")
            break
        """
        ###
        #query = input("Enter a query: ").lower()
        
        
        # Correct the query using the get_corrected_string function
        """
        corrected_query = get_corrected_string(lambda user_input: process_user_input(user_input))
        
        if corrected_query is None:
            print("No corrections made or query was quit.")
            continue

        query = corrected_query
        """
        query = input("Enter a query: ")
        query = query.lower()
        if query == "quit":
            print("Quit application.")
            break
        # Handle the original query
        start_time = time.time()
        result_entries = handle_complex_query(query, inverted_index, data_dict)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        # Print numbered results
        if result_entries:
            for i, entry in enumerate(result_entries):
                print(f"{i+1}. {entry['title']}")
        else:
            print("No entries found matching your query.")




"""
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
"""
