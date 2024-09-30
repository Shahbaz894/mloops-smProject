import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
# logging configration
logger=logging.getLogger('data_transformation')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')




def lemmatization(text):
    
    lematizer=WordNetLemmatizer()
    text=text.split()
    text=[lematizer.lemmatize(word) for word in text]
    return ''.join(text)

def remove_stop_words(text):
    stop_words=set(stopwords.words('english'))
    text=[word for word in str(text).split() if word not in stop_words]
    return ''.join(text)

def removing_numbers(text):
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def lower_case(text):
    text= re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text
    
def remove_urls(text):
    url_patteren=re.compile(r'https?://\S+|www\.\S+')
    return url_patteren.sub(r'',text)

def normalize_text(df):
    try:
        text=lower_case(text)
        text=remove_stop_words(text)  
        text=removing_numbers(text)
        text=removing_punctuations(text)
        text=remove_urls(text)
        text=lemmatization(text)
        return text
    
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise
    