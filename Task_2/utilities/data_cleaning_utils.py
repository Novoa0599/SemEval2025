import re
import unicodedata
import emoji
import json
from typing import Optional
from nltk.corpus import stopwords
import spacy

# Global models and resources
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")
stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))

def get_language_resources(language: str):
    if language == "es":
        return nlp_es, stop_words_es
    return nlp_en, stop_words_en

def process_emojis(text: str, remove: bool = False, replace: bool = False, replacement: str = "EMOJI") -> str:
    if remove:
        return emoji.replace_emoji(text, '')
    if replace:
        return emoji.replace_emoji(text, replacement)
    return text

def process_urls(text: str, remove: bool = False, replace: bool = False, replacement: str = "URL") -> str:
    if remove:
        return re.sub(r'http\S+|www\S+|https\S+', '', text)
    if replace:
        return re.sub(r'http\S+|www\S+|https\S+', replacement, text)
    return text

def process_mentions_and_hashtags(text: str, remove: bool = False, replace_mentions: bool = False, 
                                replace_hashtags: bool = False, mention_replacement: str = "MENTION", 
                                hashtag_replacement: str = "HASHTAG") -> str:
    if remove:
        text = re.sub(r'@\w+|#\w+', '', text)
    elif replace_mentions:
        text = re.sub(r'@\w+', mention_replacement, text)
    elif replace_hashtags:
        text = re.sub(r'#\w+', hashtag_replacement, text)
    return text

def process_numbers(text: str, remove: bool = False, replace: bool = False, replacement: str = "NUMBER") -> str:
    if remove:
        return re.sub(r'\d+', '', text)
    if replace:
        return re.sub(r'\d+', replacement, text)
    return text

def process_special_characters(text: str, remove: bool = False, replace: bool = False, replacement: str = "") -> str:
    if remove:
        return re.sub(r'[^a-zA-Z0-9\s¿?!¡.,ñ]', '', text)
    if replace:
        return re.sub(r'[^a-zA-Z0-9\s¿?!¡.,ñ]', replacement, text)
    return text

def replace_accents(text: str) -> str:
    normalized = unicodedata.normalize('NFD', text)
    replaced = ''.join(
        c if unicodedata.category(c) != 'Mn' or c == '̃' else ''  # Mantiene la tilde sobre la ñ
        for c in normalized
    )
    return unicodedata.normalize('NFC', replaced)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text: str, stop_words: set) -> str:
    words = text.split()
    return ' '.join(word for word in words if word.lower() not in stop_words)

def lemmatize_text(text: str, nlp) -> str:
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_stop)

def load_config(file_path: str) -> dict:
    default_config = {
        "remove_emojis": False,
        "replace_emojis": False,
        "emoji_replacement": "EMOJI",
        "remove_urls": False,
        "replace_urls": False,
        "url_replacement": "URL",
        "remove_mentions_and_hashtags": False,
        "replace_mentions": False,
        "replace_hashtags": False,
        "mention_replacement": "MENTION",
        "hashtag_replacement": "HASHTAG",
        "remove_numbers": False,
        "replace_numbers": False,
        "number_replacement": "NUMBER",
        "remove_special_characters": False,
        "replace_special_characters": False,
        "special_character_replacement": " ",
        "replace_accents": False,
        "remove_stopwords": False,
        "lemmatize": False
    }
    
    try:
        with open(file_path, 'r') as f:
            user_config = json.load(f)
            return {**default_config, **user_config}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Config loading error: {str(e)}")
        return default_config

def clean_text(text: str, config_path: Optional[str] = None, language: str = "english") -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    
    config = load_config(config_path) if config_path else {}
    nlp, stop_words = get_language_resources(language)
    
    text = process_emojis(text, config.get("remove_emojis"), config.get("replace_emojis"),
                         config.get("emoji_replacement", "EMOJI"))
    text = process_urls(text, config.get("remove_urls"), config.get("replace_urls"),
                       config.get("url_replacement", "URL"))
    text = process_mentions_and_hashtags(text, config.get("remove_mentions_and_hashtags"),
                                       config.get("replace_mentions"), config.get("replace_hashtags"),
                                       config.get("mention_replacement", "MENTION"),
                                       config.get("hashtag_replacement", "HASHTAG"))
    text = process_numbers(text, config.get("remove_numbers"), config.get("replace_numbers"),
                         config.get("number_replacement", "NUMBER"))
    if config.get("replace_accents"):
        text = replace_accents(text)
    text = process_special_characters(text, config.get("remove_special_characters"),
                                    config.get("replace_special_characters"),
                                    config.get("special_character_replacement", " "))
    
    text = normalize_whitespace(text)
    
    if config.get("remove_stopwords"):
        text = remove_stopwords(text, stop_words)
        
    if config.get("lemmatize"):
        text = lemmatize_text(text, nlp)

    return text if text else None