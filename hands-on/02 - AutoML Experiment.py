# Databricks notebook source
# MAGIC %md
# MAGIC # Run AutoML experiment 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure AutoML experiment
# MAGIC 
# MAGIC Start with configuring the AutoML experiment. This is an example. 
# MAGIC 
# MAGIC Regarding the choice of the source table:
# MAGIC 
# MAGIC * If you successfully completed **01 - EDA & Featurization**, you should use **mltraining20220621.telco_churn_features**.
# MAGIC * If you did NOT complete **01 - EDA & Featurization** successfully, you should use **mltraining20220621.telco_churn**.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/automl_configure_part_1.png?raw=true" width="1200">

# COMMAND ----------

# MAGIC %md
# MAGIC Advanced configuration
# MAGIC 
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/automl_configure_part_2.png?raw=true" width="1200">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train & evalutate
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/automl_train_evaluate_part_1.png?raw=true" width="1200">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Explore best run & data exploration notebook
# MAGIC 
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/automl_explore_codes.png?raw=true" width="1200">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Explore best run in the ML Tracking server
# MAGIC 
# MAGIC <img src="https://github.com/aminenouira-db/images/blob/main/automl_explore_best_run.png?raw=true" width="1200">
