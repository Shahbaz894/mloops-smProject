import mlflow
from flask import Flask, render_template, request
from preprocessing_utilis import normalize_text
import os
import pickle
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT") or os.getenv("MLFLOW_TRACKING_PASSWORD")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT or MLFLOW_TRACKING_PASSWORD environment variable is not set")

os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Shahbaz894"
repo_name = "mloops-smProject"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Load model
# model_name = 'my_model'
# model_version = 3
# model_uri = f'models:/{model_name}/{model_version}'
# model = mlflow.pyfunc.load_model(model_uri)
# import pickle

# # Load the vectorizer
# with open('models/vectorizer.pkl', 'rb') as file:
#     vectorizer = pickle.load(file)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     text = request.form['text']
    
#     # Clean
#     text = normalize_text(text)
    
#     # bow apply
#     features=vectorizer.transform([text])
    
#     # Prediction (add your prediction code here)
#     result=model.predict(features)
    
#     # Show user (return the prediction result here)
#     return 'Predicted text: {}'.format(result)  # Update with actual prediction result

# if __name__ == '__main__':
#     app.run(debug=True)

app = Flask(__name__)

# load model from model registry
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "my_model"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)

    # show
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")