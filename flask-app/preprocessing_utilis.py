import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging configuration
logger = logging.getLogger('data_transformation')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Download necessary NLTK resources (if not already done)
nltk.download('stopwords')
nltk.download('wordnet')

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)  # Ensure spaces between words

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text = [word for word in str(text).split() if word not in stop_words]
    return ' '.join(text)  # Ensure spaces between words

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

def lower_case(text):
    return text.lower()  # Convert text to lower case

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    try:
        logger.debug("Original text: %s", text)

        text = lower_case(text)
        logger.debug("After lower casing: %s", text)

        text = remove_urls(text)
        logger.debug("After removing URLs: %s", text)

        text = removing_punctuations(text)
        logger.debug("After removing punctuations: %s", text)

        text = removing_numbers(text)
        logger.debug("After removing numbers: %s", text)

        text = remove_stop_words(text)
        logger.debug("After removing stop words: %s", text)

        text = lemmatization(text)
        logger.debug("After lemmatization: %s", text)

        return text

    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise
