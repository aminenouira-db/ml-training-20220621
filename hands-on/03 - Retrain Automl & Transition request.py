# Databricks notebook source
# MAGIC %md
# MAGIC ## Monthly AutoML Retrain
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/mlops-end2end-flow-7.png?raw=true" width="1200">

# COMMAND ----------

# MAGIC %md ## Monthly training job
# MAGIC 
# MAGIC We can programatically schedule a job to retrain our model, or retrain it based on an event if we realize that our model doesn't behave as expected.
# MAGIC 
# MAGIC This notebook should be run as a job. It'll call the Databricks Auto-ML API, get the best model and request a transition to Staging.

# COMMAND ----------

# For multiple users working is the same workspace, we'd create a different database to store the feature store
import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = "delta_" + re.sub(r'\W+', '_', current_user)

# COMMAND ----------

# DBTITLE 1,Load Features
from databricks.feature_store import FeatureStoreClient
import pyspark.sql.functions as F
fs = FeatureStoreClient()
features = fs.read_table(f'{dbName}.telco_churn_features')

# COMMAND ----------

# DBTITLE 1,Run AutoML
import databricks.automl
model = databricks.automl.classify(features, 
                                   target_col = "Churn", 
                                   data_dir= "dbfs:/tmp/", 
                                   timeout_minutes=5) 

# COMMAND ----------

# DBTITLE 1,Register the Best Run
import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient()

run_id = model.best_trial.mlflow_run_id

model_name = "mltraining_customer_churn" + re.sub(r'\W+', '_', current_user)
model_uri = f"runs:/{run_id}/model"

client.set_tag(run_id, key='db_table', value=f'{dbName}.telco_churn_features')

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# DBTITLE 1,Add Descriptions
model_version_details = client.get_model_version(name=model_name, version=model_details.version)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using autoML and automatically getting the best model."
)

# COMMAND ----------

# DBTITLE 1,Transition model stage helper
from mlflow.utils.rest_utils import http_request
import json

client = mlflow.tracking.client.MlflowClient()

host_creds = client._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()

# COMMAND ----------

# DBTITLE 1,Request transition to Staging
# Transition request to staging
staging_request = {'name': model_name, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'true'}
mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))
