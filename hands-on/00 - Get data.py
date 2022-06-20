# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Get Data
# MAGIC 
# MAGIC In the labs, we will assume that we the data is uploaded to this location: **/FileStore/mltraining20220621/chrun_data/** 
# MAGIC ### Using the Data UI:
# MAGIC 
# MAGIC 1. Download the [data csv file](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) from github
# MAGIC 2. Upload data to DBFS in your workspace:
# MAGIC   * In production, it is highly recommended to upload the data to an adls location and mount it to the workspace. 
# MAGIC   * For simplicity and demo purpose, we will go simple & use the UI. Please refer to [the documentation](https://docs.microsoft.com/en-us/azure/databricks/data/data) for more details on how to upload data to dbfs. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Using command line

# COMMAND ----------

# DBTITLE 1,Download the file
# MAGIC %sh mkdir -p /FileStore/mltraining20220621/churn_data; wget -O /FileStore/mltraining20220621/churn_data/Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# COMMAND ----------

# DBTITLE 1,Create a DBFS data location 
# MAGIC %fs mkdirs /FileStore/mltraining20220621/churn_data

# COMMAND ----------

# DBTITLE 1,Copy file to the DBFS data location
# MAGIC %fs cp file:/FileStore/mltraining20220621/churn_data/Telco-Customer-Churn.csv dbfs:/FileStore/mltraining20220621/churn_data/Telco-Customer-Churn.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if the file exists
# MAGIC 
# MAGIC Regardless if the UI or command line option was used, the uploaded file should appear in **dbfs:/FileStore/mltraining20220621/churn_data/Telco-Customer-Churn.csv**

# COMMAND ----------

# DBTITLE 1,Check if the file exists
# MAGIC %fs ls /FileStore/mltraining20220621/churn_data/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Read data and store it in a delta table
# MAGIC 
# MAGIC #### Define variables and create a database

# COMMAND ----------

# DBTITLE 1,Please DO NOT change the variables values
dbName = "mltraining20220621"
# We assume the data csv file is located in teleco_churn_dataset as stated in the previous section
teleco_churn_dataset = "/FileStore/mltraining20220621/churn_data/"

# COMMAND ----------

# Create the database if does not exists
spark.sql(f"""CREATE DATABASE IF NOT EXISTS {dbName}""")
spark.sql(f"""USE {dbName}""")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read data and store it in a managed delta table

# COMMAND ----------

telco_df = spark.read.csv(teleco_churn_dataset, header="true", inferSchema="true")
telco_df.write.format("delta").mode("overwrite").saveAsTable(f"{dbName}.telco_churn")

# COMMAND ----------

# DBTITLE 1,Check table is created
# MAGIC %sql
# MAGIC SELECT * FROM mltraining20220621.telco_churn
