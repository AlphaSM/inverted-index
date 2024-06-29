# main_program.py

from t2s_spellchecker import SpellChecker
from class_text_to_speech import SpeechToText  # Assuming you have this class

def main():
    # Initialize SpellChecker
    spell_checker = SpellChecker()
    spell_checker.add_file_to_index('words.txt')
    spell_checker.initialize('data_and_ids_g_keys.xlsx', 'Version 2')

    # Initialize SpeechToText
    stt = SpeechToText()

    print("Speech-to-Text and Spell Checker initialized.")
    print("Speak something to check spelling...")

    while True:
        # Get speech input
        print("Listening...")
        speech_text = stt.get_text()
        print(f"You said: {speech_text}")

        # Process the speech text with the spell checker
        result = spell_checker.process_user_input(speech_text)

        print("Spell check suggestions:")
        for word, suggestions in result.items():
            print(f"{word}: {suggestions}")
        
        print("\nSpeak again or say 'quit' to exit.")
        
        if 'quit' in speech_text.lower():
            break

if __name__ == "__main__":
    main()
