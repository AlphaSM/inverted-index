import random
import itertools
import pandas as pd
from difflib import SequenceMatcher
import re
from collections import defaultdict
import time

class QueryCorrector:
    def __init__(self, excel_file, sheet_name, word_file):
        self.trie = defaultdict(set)
        self.inverted_index = {}
        self.data_dict = self.read_excel_dict(excel_file, sheet_name)
        self.my_dict = ' '.join([entry['title'] + ' ' + ' '.join(entry['keywords']) for entry in self.data_dict])
        self.add_file_to_index(word_file, self.inverted_index)

    def add_to_trie(self, word):
        word = word.lower()
        for i in range(len(word)):
            self.trie[word[:i+1]].add(word)

    def add_file_to_index(self, file_path, index):
        with open(file_path, 'r') as f:
            text = f.read()
        tokens = re.findall(r'\w+', text.lower())
        for token in tokens:
            if token not in index:
                index[token] = []
            index[token].append(file_path)
            self.add_to_trie(token)

    def read_excel_dict(self, file, sheet_name):
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

    def search_words2(self, user_input, exclude_list=[]):
        words_list = self.my_dict.split()
        search_substring = user_input.lower()
        result = [word for word in words_list if search_substring in word.lower() and word.lower() not in exclude_list]
        ranked_result = sorted(result, key=lambda x: self.similarity(search_substring, x), reverse=True)[:3]
        if not ranked_result:
            return None
        else:
            return ranked_result

    @staticmethod
    def similarity(s1, s2):
        return SequenceMatcher(None, s1, s2).ratio()

    def search_inverted_index(self, word, limit=3):
        word_lower = word.lower()
        matches = self.trie[word_lower]
        return list(matches)[:limit]

    def get_user_selection(self, word, suggestions):
        shown_suggestions = suggestions
        all_valid_suggestions = []
        
        # Add original suggestions
        all_valid_suggestions.extend(shown_suggestions)
        
        # Add the original word
        all_valid_suggestions.append(word)
        
        # Add the word with highest letter match percentage
        letter_match_word = self.suggest_highest_letter_match(word)
        all_valid_suggestions.append(letter_match_word)
        
        # Remove duplicates and keep order
        all_valid_suggestions = list(dict.fromkeys(all_valid_suggestions))
        
        return all_valid_suggestions

    def suggest_highest_letter_match(self, word):
        word_lower = word.lower()
        possible_words = self.my_dict.split()
        best_match = ""
        highest_percentage = 0

        for possible_word in possible_words:
            match_percentage = self.calculate_letter_match_percentage(word_lower, possible_word)
            if match_percentage > highest_percentage:
                highest_percentage = match_percentage
                best_match = possible_word

        return best_match

    @staticmethod
    def calculate_letter_match_percentage(word1, word2):
        matches = sum((word1.count(char) == word2.count(char)) for char in set(word1))
        return matches / len(word1) if len(word1) > 0 else 0

    def process_user_input(self, user_input):
        user_words = user_input.split()
        final_results = {}
        
        for word in user_words:
            # Remove punctuation
            punctuation = ''
            if word[-1] in [',', '.', '!', '?']:
                punctuation = word[-1]
                word = word[:-1]
            
            result = self.search_words2(word)
            autocorrect_results = self.autocorrect(word)
            
            if result is not None:
                all_suggestions = list(set(autocorrect_results + result))
            else:
                all_suggestions = autocorrect_results
            
            valid_suggestions = self.get_user_selection(word, all_suggestions)
            final_results[word + punctuation] = valid_suggestions
        
        return final_results
    
    
    @staticmethod
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

    def autocorrect(self, word, exclude_list=[], limit=3):
        word_lower = word.lower()
        possible_words = [w for w in self.inverted_index.keys() if w.startswith(word_lower[0])]
        possible_words.sort(key=lambda w: self.levenshtein_distance(word_lower, w))
        closest_words = [w for w in possible_words if self.levenshtein_distance(word_lower, w) <= 2 and w not in exclude_list]
        return closest_words[:limit]

    def reorder_and_search(self, word):
        start_time = time.time()
        word_lower = word.lower()
        permutations = itertools.permutations(word_lower)
        index_matches = []
        my_dict_matches = []
        counter = 0

        print("Reordered permutations:")
        for p in permutations:
            perm = ''.join(p)
            print(perm)
            if perm in self.inverted_index:
                index_matches.append(perm)
            if perm in self.my_dict.split():
                my_dict_matches.append(perm)
            counter += 1
            if counter % 3 == 0:
                if index_matches or my_dict_matches:
                    break

        all_matches = list(set(index_matches + my_dict_matches))
        end_time = time.time()
        print(f"Reordering took {end_time - start_time} seconds")
        return all_matches[:3]

    def get_corrected_string(self):
        while True:
            user_in = input("Enter a substring to search or press 'q' to quit: ")
            if user_in.lower() == 'q':
                return None

            result = self.process_user_input(user_in)
            """
            if not result:
                print("No results found.")
            else:
                for word, suggestions in result.items():
                    print(f"Suggestions for '{word}': {suggestions}")
                
                return result
            """

            return result  # Return the dictionary of words and their suggestions

# Usage example:
query_corrector = QueryCorrector('data_and_ids_g_keys.xlsx', 'Version 2', 'words.txt')
suggestions_dict = query_corrector.get_corrected_string()
print(f"Suggestions dictionary: {suggestions_dict}")