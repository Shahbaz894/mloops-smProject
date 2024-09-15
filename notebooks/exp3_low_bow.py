# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import dagshub

# Download the necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set up DagsHub tracking
mlflow.set_tracking_uri('https://dagshub.com/Shahbaz894/mloops-smProject.mlflow')
dagshub.init(repo_owner='Shahbaz894', repo_name='mloops-smProject', mlflow=True)


# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

# Text Preprocessing Functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_words]
    return ' '.join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    """Convert text to lower case."""
    return text.lower()

def removing_punctuations(text):
    """Remove punctuations from the text."""
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

# Normalize the text data
df = normalize_text(df)

# Filter for two classes: happiness and sadness
df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name for MLflow
mlflow.set_experiment("Logistic Regression Hyperparameter Tuning")

# Define the parameter grid for Logistic Regression
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Start the parent run for hyperparameter tuning
with mlflow.start_run(run_name="Logistic Regression Hyperparameter Tuning"):

    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log each parameter combination as a nested run
    for idx, (params, mean_score, std_score) in enumerate(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score'])):
        with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
            # Train model with the current parameter set
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            # Model evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Print the results for verification
            print(f"Params: {params}, Mean CV Score: {mean_score}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Log the best run details in the parent run
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)
    
    print(f"Best Params: {best_params}")
    print(f"Best F1 Score: {best_score}")

    # Log the best model to MLflow
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
