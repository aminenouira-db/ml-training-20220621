# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis & Feature Engineering
# MAGIC 
# MAGIC 
# MAGIC The goal is to produce a service that can predict in real-time whether a customer churns.
# MAGIC 
# MAGIC Our first step is to analyze the data and build the features we'll use to train our model. Let's see how this can be done.
# MAGIC 
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/mlops-end2end-flow-1.png?raw=true" width="1200">

# COMMAND ----------

# DBTITLE 1,Define variable & database
# For multiple users working is the same workspace, we'd create a different database to store the feature store
import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = "delta_" + re.sub(r'\W+', '_', current_user)
# We assume the notebook 00-Get Data was executed successfully ( upload data and create delta table)
telco_churn_table ="telco_churn"
telco_churn_dataset = "/FileStore/mltraining20220621/churn_data/"
model_name = "customer_churn_mltwfs_" + re.sub(r'\W+', '_', current_user)
spark.sql(f"""CREATE DATABASE IF NOT EXISTS {dbName}""")
spark.sql(f"""USE {dbName}""")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS telco_churn SHALLOW CLONE mltraining20220621.telco_churn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:right ;">
# MAGIC   <img src="https://raw.githubusercontent.com/databricks/koalas/master/Koalas-logo.png" width="180"/>
# MAGIC </div>
# MAGIC 
# MAGIC ### Using Pandas API on Apache Spark
# MAGIC Using Databricks, Data Scientist don't have to learn a new API to analyse data and deploy new model in production
# MAGIC 
# MAGIC * If you model is small and fit in a single node, you can use a Single Node cluster with pandas directly
# MAGIC * If your data grow, no need to re-write the code. Just switch to spark pandas
# MAGIC 
# MAGIC <div style="float:right ;">
# MAGIC   <img src="https://databricks.com/fr/wp-content/uploads/2021/09/Pandas-API-on-Upcoming-Apache-Spark-3.2-blog-img-4.png" width="300"/>
# MAGIC </div>
# MAGIC 
# MAGIC 
# MAGIC One of the known limitations in pandas is that it does not scale with your data volume linearly due to single-machine processing. For example, pandas fails with out-of-memory if it attempts to read a dataset that is larger than the memory available in a single machine.
# MAGIC 
# MAGIC 
# MAGIC Pandas API on Spark overcomes the limitation and scale beyond a single machine, enabling users to work with large datasets by leveraging Spark!
# MAGIC 
# MAGIC **Starting with spark 3.2, pandas API are directly part of the spark runtime, no need to import external library!**

# COMMAND ----------

#Just import pyspark pandas to leverage spark distributed capabilities:
import pyspark.pandas as ps
telco_ps_df = ps.read_csv(telco_churn_dataset)
display(telco_ps_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC Pandas API on spark use plotly with interactive charts. Use the standard `.plot` on a pandas dataframe to get insights.
# MAGIC 
# MAGIC **Task**: Plot the distribution of the label variable (Churn column) using plotly to check if the data is unbalanced

# COMMAND ----------

# Plot distribution of target variable - Churn column
telco_ps_df['Churn'].value_counts().plot.pie()   # To generate a bar plot

# COMMAND ----------

ps.sql("""SELECT churn, SeniorCitizen, COUNT(*) AS num_customer_per_seniority 
  FROM {telco_ps_df} GROUP BY SeniorCitizen, churn""", telco_ps_df=telco_ps_df).pivot(index="churn", columns="SeniorCitizen", values="num_customer_per_seniority").plot.bar()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Using Databricks Data Profile
# MAGIC 
# MAGIC **Task:** Use the Data Profile to answer the following question regarding `telco_ps_df`: *Can you quickly recognise what is the problem with the feature `totalCharges`?*

# COMMAND ----------

display(telco_ps_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurization
# MAGIC 
# MAGIC **Feature engineering** or **feature extraction** or **feature discovery** is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data. The motivation is to use these extra features to improve the quality of results from a machine learning process, compared with supplying only the raw data to the machine learning process. [source](https://en.wikipedia.org/wiki/Feature_engineering)
# MAGIC 
# MAGIC Next, we would make a few featurization based on the EDA we performed (TotalCharges feature) and domain knowledge (Contract feature).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Task**: Apply the following transformation using Pandas APIs or Spark APIs:
# MAGIC * `SeniorCitizen`: From Numeric to Boolean: 0 -> False , 1 -> True
# MAGIC * `Contract`: From String to Numeric (integer): Month-to-month -> 1 , One year -> 12 , Two year -> 24
# MAGIC * `TotalCharges`: From String to Numeric and fill the non-castable values (spaces only) with 0.0
# MAGIC 
# MAGIC 
# MAGIC PS: This part can be externalized and run is a scheduled job.

# COMMAND ----------

# DBTITLE 1,Using Pandas on Spark API
# 0/1 -> boolean
telco_ps_df["SeniorCitizen"] = telco_ps_df["SeniorCitizen"]==1
# Contract categorical -> duration in months
telco_ps_df.loc[telco_ps_df.Contract=='Month-to-month', 'Contract'] = 1
telco_ps_df.loc[telco_ps_df.Contract=='One year', 'Contract'] = 12
telco_ps_df.loc[telco_ps_df.Contract=='Two year', 'Contract'] = 24
telco_ps_df["Contract"] = telco_ps_df["Contract"].astype("int")
# TotalCharges Categorical to Numeric and fill the non-castable values (spaces only) with 0.0
telco_ps_df.loc[len(telco_ps_df.TotalCharges.str.strip())==0, "TotalCharges"] = 0
telco_ps_df["TotalCharges"] = telco_ps_df["TotalCharges"].astype("double")
display(telco_ps_df.describe())

# COMMAND ----------

# DBTITLE 1,Using PySpark - Mandatory to run for the next steps
import pyspark.sql.functions as F
telco_df = spark.table(f"{dbName}.{telco_churn_table}")
# 0/1 -> boolean
telco_df = telco_df.withColumn("SeniorCitizen", F.col("SeniorCitizen") == 1)
# Contract categorical -> duration in months
telco_df = telco_df.withColumn("Contract",\
    F.when(F.col("Contract") == "Month-to-month", 1).\
    when(F.col("Contract") == "One year", 12).\
    when(F.col("Contract") == "Two year", 24))
# TotalCharges Categorical to Numeric and fill the non-castable values (spaces only) with 0.0
telco_df = telco_df.withColumn("TotalCharges",\
    F.when(F.length(F.trim(F.col("TotalCharges"))) == 0, 0.0).\
    otherwise(F.col("TotalCharges").cast('double')))
# Cache DataFrame
telco_df.cache()
display(telco_df.summary())

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Write to Feature Store
# MAGIC 
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a **Delta Lake table**.
# MAGIC 
# MAGIC This will allow discoverability and reusability of our feature accross our organization, increasing team efficiency.
# MAGIC 
# MAGIC Feature store will bring traceability and governance in our deployment, knowing which model is dependent of which set of features.
# MAGIC 
# MAGIC Make sure you're using the "Machine Learning" menu to have access to your feature store using the UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Writing Feature Store Tables
# MAGIC 
# MAGIC With the tables defined by functions above, next, the feature store tables need to be written out, first as Delta tables. These are the fundamental 'offline' data stores underpinning the feature store tables. Use the client to create the feature tables, defining metadata like which database and table the feature store table will write to, and importantly, its key(s).
# MAGIC 
# MAGIC **Task**: create a feature store table and write the telco dataframe to it. Use `fs.create_table`
# MAGIC 
# MAGIC **PS**: It is recommended that we do NOT store the target column in the feature store. Instead, it should be stored outside alongside with the key then joined when training the model. With the feature store client or automl (which are what we are using in this lab), it will exclude it from the training as you define it as a target column in `fs.create_training_set`. If you wouldn't use them (FS/AutoML) during the model training, you would need to explicitely exclude or be always specific with which features to use otherwise it will cause "data leakage".

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
# Instantiate the feature store client
fs = FeatureStoreClient()
# Create the feature store based on df schema and write df to it
features_table = fs.create_table(
  name=f'{dbName}.telco_churn_features',
  primary_keys=['customerID'],
  df=telco_df,
  description='Telco churn features')

# COMMAND ----------

# Drop feature table if it exists
# fs.drop_table(name=f'{dbName}.telco_churn_features'),
help(fs.drop_table)

# COMMAND ----------

# Verify the feature store has been created and populated successfully
features_table = fs.read_table(f'{dbName}.telco_churn_features')
display(features_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Browsing the Feature Store
# MAGIC 
# MAGIC The tables are now visible and searchable in the [Feature Store](/#feature-store) 
# MAGIC 
# MAGIC **Task**: Explore the feature store UI, search by a feature value or table name

# COMMAND ----------

# MAGIC %md
# MAGIC ## BONUS: Building Models from a Feature Store
# MAGIC 
# MAGIC Next, it's time to build a simple model predicting whether the customer churns. 
# MAGIC 
# MAGIC Below, a `FeatureLookup` is created for each feature in the `telco_churn_features` table. This defines how other features are looked up (table, feature names) and based on what key -- `customerID`.

# COMMAND ----------

from databricks.feature_store import FeatureLookup
# Get metadata about the feature store
features_table = fs.get_table(f'{dbName}.telco_churn_features')
# Optional for FS but good to remember as a good practice: Drop the label & key columns from the list of features to use for the model
features = features_table.features
features.remove("Churn")
features.remove("customerID")
# Define the feature lookup - you can specify the list of features as well to use
features_table_lookup = FeatureLookup(table_name = features_table.name, 
                                      lookup_key = 'customerID',
                                      feature_names= features
                                     ) 
             
features_table_lookups  = [features_table_lookup]

# COMMAND ----------

# MAGIC %md
# MAGIC Now, modeling can proceed. `create_training_set` pulls together all the features. Its "input" is the `inference_data` table, as discussed above, and the `FeatureLookup`s define how to join in features. Modeling ignores the `customerID` of course; it's a key rather than useful data.
# MAGIC 
# MAGIC The modeling here is simplistic, and just trains a plain `sklearn` gradient boosting classifier.
# MAGIC 
# MAGIC Normally, this model would be logged (or auto-logged) by MLflow. It's necessary however to log to MLflow through the feature store client's `log_model` method instead. The Feature Store needs to package additional metadata about feature lookups and feature tables in order for the model to perform lookups and joins correctly at inference time, when deployed as a service.
# MAGIC 
# MAGIC Note that code from `read_table` through to `log_model` must be inside an MLflow run. When complete, a new run has been logged and registered in the MLflow Model Registry.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.sklearn.autolog()

# Create the training dataframe. We will feed this dataframe to our model to perform the feature lookup from the feature store and then train the model
train_data_df = telco_df.select("customerID", "Churn")

# Define a method for reuse later
def fit_model(model_feature_lookups):

  with mlflow.start_run():
    # Use a combination of Feature Store features and data residing outside Feature Store in the training set
    training_set = fs.create_training_set(train_data_df, model_feature_lookups, label="Churn", exclude_columns="customerID")

    training_pd = training_set.load_df().toPandas()
    X = training_pd.drop("Churn", axis=1)
    y = training_pd["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Not attempting to tune the model at all for purposes here
    gb_classifier = GradientBoostingClassifier(n_iter_no_change=10)
    # Need to encode categorical cols
    encoders = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), X.columns[X.dtypes == 'object'])])
    pipeline = Pipeline([("encoder", encoders), ("gb_classifier", gb_classifier)])
    pipeline_model = pipeline.fit(X_train, y_train)
    
    mlflow.log_metric('test_accuracy', pipeline_model.score(X_test, y_test))

    fs.log_model(
      pipeline_model,
      "model",
      flavor=mlflow.sklearn,
      training_set=training_set,
      registered_model_name=model_name,
      input_example=X[:100],
      signature=infer_signature(X, y))
      
fit_model(features_table_lookups)

# COMMAND ----------

# MAGIC %md
# MAGIC **Task**: Explore the feature store UI and see the downstreams field being populated by the model that was created

# COMMAND ----------

# MAGIC %md
# MAGIC It's trivial to apply a registered MLflow model to features with `score_batch`. Again its only input are `customerID`s (without the label `Churn` of course!). Everything else is looked up. However, eventually the goal is to produce a real-time service from this model and these features.
# MAGIC 
# MAGIC **Task**: Use `fs.score_batch` method

# COMMAND ----------

from pyspark.sql.functions import col
# Get the dataframe of customer ids to predict
batch_input_df = train_data_df.select("customerID")
# Generate predictions with fs.score_batch
with_predictions = fs.score_batch(f"models:/{model_name}/1", batch_input_df, result_type='string')
display(with_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Publishing Feature Store Tables
# MAGIC 
# MAGIC So far, the features exist 'offline' as Delta tables. It's been very convenient to define features and compute them with Spark and persist them for batch use. However, these features need to be available 'online' too. It would be prohibitively slow recompute features in real-time. The features are already computed, but exist in tables which may be slow to read individual rows from; the Delta tables may not be accessible, even, from the real-time serving context!
# MAGIC 
# MAGIC The Feature Store can 'publish' the data in these tables to an external store that is more suitable for fast online lookups. `publish_table` does this.
# MAGIC 
# MAGIC Please refer to [the official documentation](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/feature-store/feature-tables#publish-features-to-an-online-feature-store)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create database with the same name in the online store (Azure MySQL here).

# COMMAND ----------

scope = "online_fs"
user = dbutils.secrets.get(scope, "ofs-user")
password = dbutils.secrets.get(scope, "ofs-password")
import mysql.connector
import pandas as pd
cnx = mysql.connector.connect(user=user,
                              password=password,
                              host=<hostname>)
cursor = cnx.cursor()
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {dbName};")

# COMMAND ----------

import datetime
from databricks.feature_store.online_store_spec import AzureMySqlSpec
 
online_store = AzureMySqlSpec(
  hostname=<hostname>,
  port=3306,
  read_secret_prefix='online_fs/ofs',
  write_secret_prefix='online_fs/ofs'
)
 
fs.publish_table(
  name=f'{dbName}.demographic_features',
  online_store=online_store,
  mode='overwrite'
)
 
fs.publish_table(
  name=f'{dbName}.service_features',
  online_store=online_store,
  mode='overwrite'
)

# COMMAND ----------

# MAGIC %md
# MAGIC This becomes visible in the feature table's UI as an "Online Store" also containing the same data.
