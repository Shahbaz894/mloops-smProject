from flask import Flask,render_template,request
import mlflow
app=Flask(__name__)
# load model
model_name='my_model'
model_version=3
model_uri=f'models:/{model_name}/{model_version}'
model=mlflow.pyfunc._load_model(model_uri)
from preprocessing_utilis import normalize_text
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT") or os.getenv("MLFLOW_TRACKING_PASSWORD")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT or MLFLOW_TRACKING_PASSWORD environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Shahbaz894"
repo_name = "mloops-smProject"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# # fetch model version
# def get_latest_version(model_name):
#     client=MlflowClient()
#     latest_version=client.get_latest_versions(model_name,stages=["Production"])
#     if not latest_version:
#         latest_version=client.get_latest_version(model_name,stages=["None"])
#     else:
#         return latest_version[0].version if latest_version else None
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']
    # load model from model registery
    
    # clean
    text=normalize_text(text)
    
    
    # bow
    
    # prediction
    
    # show user 
    return 'text'



app.run(debug=True)