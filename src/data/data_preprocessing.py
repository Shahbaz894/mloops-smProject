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

file_handler=logging.FileHandler('transformation_errors.log')
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('wordnet')
nltk.download('stopwords')

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
        df['content']=df['content'].apply(lower_case)
        logger.debug('converted to lower case')
        df['content']=df['content'].apply(remove_stop_words)
        logger.debug('stop words removed')
        df['content'] = df['content'].apply(removing_numbers)
        logger.debug('numbers removed')
        df['content'] = df['content'].apply(removing_punctuations)
        logger.debug('punctuations removed')
        df['content'] = df['content'].apply(remove_urls)
        logger.debug('urls')
        df['content'] = df['content'].apply(lemmatization)
        logger.debug('lemmatization performed')
        logger.debug('Text normalization completed')
        return df
    
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise
    
    
def main():
        try:
            train_data=pd.read_csv('./data/raw/train.csv')
            test_data=pd.read_csv('./data/raw/test.csv')
            logger.debug('data loaded properly')
            # transform the data
            
            train_processed_data=normalize_text(train_data)
            test_processed_data=normalize_text(test_data)
            
            # store the processsed data
            
            data_path=os.path.join("./data" , "interim")
            os.makedirs(data_path,exist_ok=True)
            
            train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"),index=False)
            test_processed_data.to_csv(os.path.join(data_path,"test_processed_data.csv"),index=False)
            
            logger.debug('Processed data saved to %s', data_path)
        except Exception as e:
            logger.error('Failed to complete the data transformation process: %s', e)
            print(f"Error: {e}")
            
            
if __name__ == '__main__':
    main()