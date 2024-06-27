import itertools
import pandas as pd
import re
from collections import defaultdict

# Download NLTK resources (if not already downloaded)
#nltk.download('punkt')
#nltk.download('wordnet')

from difflib import SequenceMatcher


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
    #words_list = my_dict.split()
    words_list = my_dict
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

def get_user_selection1( word, suggestions):
        shown_suggestions = suggestions
        all_valid_suggestions = []
        
        # Add original suggestions
        all_valid_suggestions.extend(shown_suggestions)
        
        # Add the original word
        all_valid_suggestions.append(word)
        
        # Add the word with highest letter match percentage
        letter_match_word = suggest_highest_letter_match(word)
        all_valid_suggestions.append(letter_match_word)
        
        #Add reorder if less than 4 letters 
        if len(word) <= 4:
            suggest_reordered = reorder_and_search(word, inverted_index, my_dict)
            all_valid_suggestions.append(suggest_reordered)

        # Remove duplicates and keep order
        all_valid_suggestions = list(dict.fromkeys(all_valid_suggestions))
        
        return all_valid_suggestions

def get_user_selection(word, suggestions):
    shown_suggestions = suggestions
    all_valid_suggestions = []
    
    # Add original suggestions
    all_valid_suggestions.extend(shown_suggestions)
    
    # Add the original word
    all_valid_suggestions.append(word)
    
    # Add the word with highest letter match percentage
    letter_match_word = suggest_highest_letter_match(word)
    all_valid_suggestions.append(letter_match_word)
    
    # Add reorder if less than 4 letters 
    if len(word) <= 4:
        suggest_reordered = reorder_and_search(word, inverted_index, my_dict)
        all_valid_suggestions.extend(suggest_reordered)  # Use extend instead of append

    # Remove duplicates and keep order
    all_valid_suggestions = list(dict.fromkeys(all_valid_suggestions))
    
    return all_valid_suggestions

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
          autocorrect_results = autocorrect(word, inverted_index)
          
          #print(f"Suggestions for '{word}': {autocorrect_results}")
          
          if result is not None:
              #print(f"Matches found for '{word}': {result}")
              all_suggestions = list(set(autocorrect_results + result))
          else:
              all_suggestions = autocorrect_results
          """
          if word in all_suggestions:
              print(f"'{word}' is in the suggestion list.")
          
          selected_word = get_user_selection(word, all_suggestions)
          if selected_word:
              final_results[word + punctuation] = [selected_word + punctuation]
          else:
              final_results[word + punctuation] = [word + punctuation]  # Keep original if no selection
            """
          valid_suggestions = get_user_selection(word, all_suggestions)
          final_results[word + punctuation] = valid_suggestions

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

def reorder_and_search1(word, index, my_dict):
    
    word_lower = word.lower()
    permutations = itertools.permutations(word_lower)
    index_matches = []
    my_dict_matches = []
    counter = 0
    
   
    for p in permutations:
        perm = ''.join(p)
        #print(perm)  # Print the permutation
        if perm in index:
            index_matches.append(perm)
        if perm in my_dict.split():
            my_dict_matches.append(perm)
        counter += 1
        if counter % 3 == 0:
            if index_matches or my_dict_matches:
                break

    all_matches = list(set(index_matches + my_dict_matches))

    #all_matches.append("Reorder")
    
    return all_matches[:3]

def reorder_and_search(word, index, my_dict):
    word_lower = word.lower()
    permutations = itertools.permutations(word_lower)
    index_matches = []
    my_dict_matches = []
    counter = 0
   
    #print(f"Reordering: {word}")
    for p in permutations:
        perm = ''.join(p)
        #print(f"Checking permutation: {perm}")
        if perm in index:
            #print(f"Found in index: {perm}")
            index_matches.append(perm)
        if perm in my_dict:
            #print(f"Found in my_dict: {perm}")
            my_dict_matches.append(perm)
        counter += 1
        if counter % 3 == 0:
            if index_matches or my_dict_matches:
                break

    all_matches = list(set(index_matches + my_dict_matches))
    return all_matches[:3]

def get_corrected_string(process_user_input):
    while True:
        user_in = input("Enter a substring to search or press 'q' to quit: ")
        if user_in.lower() == 'q':
            return None

        result = process_user_input(user_in)
        """
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
        return result
        
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

# Main execution
if __name__ == "__main__":
    # Read data
    data_dict = read_excel_dict('data_and_ids_g_keys.xlsx', 'Version 2')

    # Build the inverted index
    inverted_index = build_inverted_index(data_dict)

    while True:
        
        #query = input("Enter a query: ").lower()
        
        
        # Correct the query using the get_corrected_string function
        
        corrected_query = get_corrected_string(lambda user_input: process_user_input(user_input))
        
        if corrected_query is None:
            print("No corrections made or query was quit.")
            continue

        query = corrected_query
        print(query)
        #query = input("Enter a query: ")


