# Databricks notebook source
# MAGIC %md
# MAGIC ### Load Packages

# COMMAND ----------

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

from pyspark.ml.regression import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.tuning import *
import mlflow
import mlflow.spark

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Wrangling

# COMMAND ----------

df_results = spark.read.csv('s3://columbia-gr5069-main/raw/results.csv',header=True)
display(df_results)

# COMMAND ----------

df = df_results.select('raceId','driverId','resultId','positionOrder','points','laps','fastestLap','fastestLapTime','fastestLapSpeed','statusId')
display(df)

# COMMAND ----------

df2 = df.filter(df.fastestLapTime != '\\N' )
display(df2)

# COMMAND ----------

for col_name in df2.columns:
    df2 = df2.withColumn(col_name, col(col_name).cast("Integer"))
df2 = df2.drop("fastestLapTime")
display(df2)

# COMMAND ----------

(df2.describe().select(
                    "summary",
                    F.round("raceId", 4).alias("raceId"),
                    F.round("driverId", 4).alias("driverId"),
                    F.round("resultId", 4).alias("resultId"),
                    F.round("positionOrder", 4).alias("positionOrder"),
                    F.round("points", 4).alias("points"),
                    F.round("laps", 4).alias("laps"),
                    F.round("fastestLap", 4).alias("fastestLap"),
                    F.round("fastestLapSpeed", 4).alias("fastestLapSpeed"),
                    F.round("statusId", 4).alias("statusId"))
                    .show())

# COMMAND ----------

df2.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Extraction

# COMMAND ----------

featureCols = ["laps", "fastestLap", "fastestLapSpeed", "statusId", "driverId", "raceId"]
assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 
assembled_df = assembler.transform(df2)
display(assembled_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare training and test data

# COMMAND ----------

train, test = assembled_df.randomSplit([0.9, 0.1], seed=12345)
display(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit Model

# COMMAND ----------

lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=1,standardization=False)
model=lr.fit(train)

# COMMAND ----------

model.coefficients

# COMMAND ----------

featureCols

# COMMAND ----------

model.intercept

# COMMAND ----------

coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols, "Co-efficients": np.insert(model.coefficients.toArray(), 0, model.intercept)})
coeff_df = coeff_df[["Feature", "Co-efficients"]]
display(coeff_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Predictions

# COMMAND ----------

predictions = model.transform(test)

# COMMAND ----------

result = RegressionEvaluator(labelCol='positionOrder',predictionCol='predpositionOrder').evaluate(predictions)
result

# COMMAND ----------

predandlabels = predictions.select("positionOrder", "predpositionOrder").show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Tuning hyperparameters Individually

# COMMAND ----------

def run_model(elasticNetParaminput):
 with mlflow.start_run(run_name="Elastic-Net-Regression-Model") as run:
  ### Feature Extraction
  featureCols = ["laps", "fastestLap", "fastestLapSpeed", "statusId", "driverId", "raceId"]
  assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 
  assembled_df = assembler.transform(df2)
  ### Prepare training and test data
  train, test = assembled_df.randomSplit([0.9, 0.1], seed=12345)
  ### Fit Model
  ### Generate Predictions
  regParam=1
  fitIntercept=True
  elasticNetParam=elasticNetParaminput
  lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=elasticNetParaminput,standardization=False)
  ### Tuning hyperparameters Individually
  model1 = lr.fit(train)  
  mlflow.spark.log_model(model1, "Elastic-Net-Regression-Model")
  predtest = model1.transform(test)
  evaluator = RegressionEvaluator(labelCol='positionOrder',predictionCol='predpositionOrder')
  rmse = evaluator.evaluate(predtest, {evaluator.metricName: "rmse"})
  mlflow.log_metric("rmse", rmse)
  mlflow.log_param("regParam",regParam) 
  mlflow.log_param("fitIntercept",fitIntercept) 
  mlflow.log_param("elasticNetParam",elasticNetParam) 
  print("  rmse: {}".format(rmse))
  print("  regParam: {}".format(regParam))
  print("  elasticNetParam: {}".format(elasticNetParam))
  print("  fitIntercept: {}".format(fitIntercept))
  runID = run.info.run_uuid 
  experimentID = run.info.experiment_id
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
 return (run.info.run_uuid, run.info.experiment_id)

run_model(elasticNetParaminput=0.1)
run_model(elasticNetParaminput=0.2)
run_model(elasticNetParaminput=0.3)
run_model(elasticNetParaminput=0.4)
run_model(elasticNetParaminput=0.5)
run_model(elasticNetParaminput=0.6)
run_model(elasticNetParaminput=0.7)
run_model(elasticNetParaminput=0.8)
run_model(elasticNetParaminput=0.9)
run_model(elasticNetParaminput=1.0)
run_model(elasticNetParaminput=0)




# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Based on the result of tuning pyperparameters, we choose elastic-net models that has elasticnet parameter value = 0 or 0.1 or 0.2, since they has the lowest RMSE value.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Grid Search and Tuning hyperparameters (MLflow)

# COMMAND ----------

with mlflow.start_run(run_name="Elastic-Net-Regression-Model") as run:
  ### Feature Extraction
  featureCols = ["laps", "fastestLap", "fastestLapSpeed", "statusId", "driverId", "raceId"]
  assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 
  assembled_df = assembler.transform(df2)
  ### Prepare training and test data
  train, test = assembled_df.randomSplit([0.9, 0.1], seed=12345)
  ### Fit Model
  ### Generate Predictions
  lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=1,standardization=False)
  ### Grid Search and Tuning hyperparameters
  paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.01, 0.1, 1, 10]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])\
    .build()
  evaluator=RegressionEvaluator(labelCol='positionOrder',predictionCol='predpositionOrder')
  tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 90% of the data will be used for training, 10% for validation.
                           trainRatio=0.9,
                           collectSubModels=True)

  model1 = tvs.fit(train)
  mlflow.spark.log_model(model1, "Elastic-Net-Regression-Model")
  predtest = model1.transform(test)
  evaluator = RegressionEvaluator(labelCol='positionOrder',predictionCol='predpositionOrder')
  rmse = evaluator.evaluate(predtest, {evaluator.metricName: "rmse"})
  mlflow.log_metric("rmse", rmse)
 # print("  paramGrid: {}".format(paramGrid))
  #mlflow.log_param("paramGrid",paramGrid)
  print("  rmse: {}".format(rmse))

  runID = run.info.run_uuid # uuid: the unique Id this runID is against
  experimentID = run.info.experiment_id
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Grid Search and Tuning hyperparameters (MLflow Autolog)

# COMMAND ----------

lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=1,standardization=False)

# COMMAND ----------

paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.01, 0.1, 1, 10]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])\
    .build()

# COMMAND ----------

evaluator=RegressionEvaluator(labelCol='positionOrder',predictionCol='predpositionOrder')

# COMMAND ----------

# In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           # 90% of the data will be used for training, 10% for validation.
                           trainRatio=0.9)

# COMMAND ----------

model1 = tvs.fit(train)

# COMMAND ----------

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
model1.transform(test)\
    .select("features", "positionOrder", "predpositionOrder").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Other Codes

# COMMAND ----------

#from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.regression import LinearRegression
#from pyspark.sql.functions import datediff, current_date, avg
#from mlflow import pyfunc
#import mlflow.tensorflow

# COMMAND ----------

#df_laptimes = spark.read.csv('s3://columbia-gr5069-main/raw/lap_times.csv',header=True)
#display(df_laptimes)
#df_drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv',header=True)
#display(df_drivers)
#df_pitstops = spark.read.csv('s3://columbia-gr5069-main/raw/pit_stops.csv',header=True)
#display(df_pitstops)
#df_lap_drivers = df_drivers.select('DriverId','driverRef','code','forename','surname','nationality','age').join(df_laptimes,on=['DriverId']) 
#assembled_df.show(10, truncate=False)

# do join in pyspark, and do select in pyspark. The order is processed starting on left hand side, similar as in SQL. 

# COMMAND ----------

#df3=df2.withColumn('fastestLapTime',to_timestamp(col('fastestLapTime'))).withColumn('DiffInSeconds',col('fastestLapTime').cast("long"))
#display(df3)
#df3=df2.withColumn('fastestLapTime',from_unixtime(unix_timestamp("fastestLapTime", "m:ss.SSS"),"mm:ss.SSSSSS"))
#spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
#df3=df2.select(unix_timestamp('fastestLapTime', 'yyyy-MM-dd').alias('unix_time')).collect()
#spark.conf.unset("spark.sql.session.timeZone")
#display(df3)
#df3=df2.withColumn("fastestLapTime",to_timestamp(col("fastestLapTime"),"mm:ss.SSS"))
#df3=df2.withColumn("fastestLapTime",unix_timestamp(col("fastestLapTime"),"mm:ss.SSS"))
#df3=df2.withColumn("fastestLapTime",unix_timestamp(col("fastestLapTime"),"mm:ss.SSS"))
#df3=df2.withColumn("fastestLapTime",df2['fastestLapTime'].cast(IntegerType()))
#display(df3)
#display(df3)
#df = spark.createDataFrame(data=dates, schema=["id","input_timestamp"])

#from pyspark.sql.functions import *

#Calculate Time difference in Seconds
#df3=df2.withColumn('fastestLapTime',to_timestamp(col('fastestLapTime')))

#\
 # .withColumn('end_timestamp', current_timestamp())\
#  .withColumn('DiffInSeconds',col("end_timestamp").cast("long") - col('fastestLapTime').cast("long"))
#df2.show(truncate=False)

# COMMAND ----------

#df2 = pyspark.sql.DataFrame.dropna(df,'any')
#display(df2)
#df2.withColumn("raceId",col("raceId").cast("Integer")).withColumn("driverId",col("driverId").cast("Integer"))
#df2=df2.select(col("positionOrder").alias("label"),col("laps").alias("feature1"),col("fastestLap").alias("feature2"),col("fastestLapSpeed").alias("feature3"),col("statusId").alias("feature4"),col("driverId").alias("feature5"),col("raceId").alias("feature6"))



# COMMAND ----------

#training = df
#lr = LinearRegression(maxIter=10)
# Fit Lasso model (λ = 1, α = 1) to training data
#regression = LinearRegression(labelCol='statusId', regParam=1, elasticNetParam=1).fit(train)

# Calculate the RMSE on testing data
##rmse = RegressionEvaluator(labelCol='duration').evaluate(regression.transform(flights_test))
#print("The test RMSE is", rmse)

# Look at the model coefficients
#coeffs = regression.coefficients
#print(coeffs)

# Number of zero coefficients
#zero_coeff = sum([beta == 0 for beta in regression.coefficients])
#print("Number of coefficients equal to 0:", zero_coeff)

#lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=1)

# Fit the model
#lrModel = lr.fit(training)

# Print the weights and intercept for linear regression
#print("Weights: " + str(lrModel.weights))
#print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

#regression = LinearRegression(labelCol='positionOrder',fitIntercept=True)
#regression=regression.fit(train)
#params = ParamGridBuilder()
#params = params.addGrid(regression.fitIntercept, [True,False]).addGrid(regression.regParam,[0.001,0.01,0.1,1,10]).addGrid(regression.elasticNetParam,[0,0.25,0.5,0.75,1])
#params = params.build()
#print('Number of models to be tested',len(params))
#evaluator= RegressionEvaluator(labelCol='positionOrder',predictionCol='predpositionOrder')
#cv= CrossValidator(estimator= model,estimatorParamMaps=params,evaluator=evaluator)
#cv=cv.setNumFolds(10).setSeed(12345).fit(train)
#cv=cv.fit(train)
#cv.avgMetrics
#cv.bestModel
#predictions = cv.transform(test)
#cv.bestModel.explainParam('fitIntercept')

# COMMAND ----------

#model=lr.fit(train)
# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.

#  mlflow.spark.log_model(model, "myModel")
#lr2 = LinearRegression(maxIter=10,featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder')

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
#paramGrid = ParamGridBuilder()\
#    .addGrid(lr2.regParam, [0.1, 0.01]) \
#    .addGrid(lr2.fitIntercept, [False, True])\
#    .addGrid(lr2.elasticNetParam, [0.0, 0.5, 1.0])\
#    .build()

  # In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
#tvs = TrainValidationSplit(estimator=lr,
                           #estimatorParamMaps=paramGrid,
                           #evaluator=RegressionEvaluator(labelCol='positionOrder',#predictionCol='predpositionOrder'),
                          # 80% of the data will be used for training, 20% for validation.
#                           trainRatio=0.9)
# Run TrainValidationSplit, and choose the best set of parameters.
#model2 = tvs.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
#model.transform(test)\
 #   .select("features", "label", "prediction")\
#  .show()

# COMMAND ----------

#mlflow.spark.log_model(model1, "myModel1")

# COMMAND ----------

#paramGrid = ParamGridBuilder()\
    #.addGrid(lr.regParam, [0.01, 0.1, 1, 10]) \
    #.addGrid(lr.fitIntercept, [False, True])\
    #.addGrid(lr.elasticNetParam, [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])\
    #.build()
#print('Number of models to be tested',len(paramGrid))
#SparkSession.builder.config("spark.jars.packages", "org.mlflow.mlflow-spark")
#mlflow.spark.autolog()
#model1 = tvs.fit(train)
#paramGrid = ParamGridBuilder()\
    #.addGrid(lr.regParam, [0.01, 0.1, 1, 10]) \
   # .addGrid(lr.elasticNetParam, [0.0,0.3,0.6,0.9,1.0])\
    #.build()
#print('Number of models to be tested',len(paramGrid))

# COMMAND ----------


#with mlflow.start_run(run_name="Elastic Net Regression Model") as run:
  #vecAssembler = VectorAssembler(inputCols = ["bedrooms", "bathrooms"], outputCol = "features")
  
  #vecTrainDF = vecAssembler.transform(trainDF)
  
  #lr = LinearRegression(featuresCol = "features", labelCol = "price") # a spark model
  
  #lrModel = lr.fit(vecTrainDF)
   # Log model
  #mlflow.spark.log_model(lrModel, "linear-regression-model")
  
  #vecTestDF = vecAssembler.transform(testDF)
  #predDF = lrModel.transform(vecTestDF)

  # Instantiate metrics object
  #evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price") #evaluation on regression model
  #mlflow is a wrapper , which catches whatever what is available for the fit that can be catched
  #log in of everything you need by mlflow
  #r2 = evaluator.evaluate(predDF, {evaluator.metricName: "r2"})
  #print("  r2: {}".format(r2))
  #mlflow.log_metric("r2", r2)

 # mae = evaluator.evaluate(predDF, {evaluator.metricName: "mae"})
  #print("  mae: {}".format(mae))
  #mlflow.log_metric("mae", mae)

  #rmse = evaluator.evaluate(predDF, {evaluator.metricName: "rmse"})
  #print("  rmse: {}".format(rmse))
  #mlflow.log_metric("rmse", rmse)

  #mse = evaluator.evaluate(predDF, {evaluator.metricName: "mse"})
  #print("  mse: {}".format(mse))
  #mlflow.log_metric("mse", mse)
  
  #runID = run.info.run_uuid # uuid: the unique Id this runID is against
  #experimentID = run.info.experiment_id
  
  #print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

#vecAssembler = VectorAssembler(inputCols = ["bedrooms"], outputCol = "features")

#vecTrainDF = vecAssembler.transform(trainDF)

#lr = LinearRegression(featuresCol = "features", labelCol = "price")
#lrModel = lr.fit(vecTrainDF)
#vecAssembler = VectorAssembler(inputCols = ["bedrooms"], outputCol = "features")

#vecTrainDF = vecAssembler.transform(trainDF)
  #vecAssembler = VectorAssembler(inputCols = ["bedrooms", "bathrooms"], outputCol = "features")
  #vecTrainDF = vecAssembler.transform(train)
  #vecTrainDF = vecAssembler.transform(trainDF)
 #evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")
  #lr = LinearRegression(featuresCol = "features", labelCol = "price") # a spark model
  #lrModel = lr.fit(vecTrainDF)
    #r2 = evaluator.evaluate(predtest, {evaluator.metricName: "r2"})
  #mae = evaluator.evaluate(predtest, {evaluator.metricName: "mae"})
  #vecTestDF = vecAssembler.transform(testDF)
  #predDF = lrModel.transform(vecTestDF)
   # Log model
  #mlflow.spark.log_model(lrModel, "linear-regression-model")
  #mse = evaluator.evaluate(predDF, {evaluator.metricName: "mse"})
  #print("  mse: {}".format(mse))
  #mlflow.log_metric("mse", mse)
    #mse = evaluator.evaluate(predtest, {evaluator.metricName: "mse"})
  #r2 = evaluator.evaluate(predDF, {evaluator.metricName: "r2"})
  #print("  r2: {}".format(r2))
  #mlflow.log_metric("r2", r2)
  #mae = evaluator.evaluate(predDF, {evaluator.metricName: "mae"})
  #print("  mae: {}".format(mae))
  #mlflow.log_metric("mae", mae)
  #rmse = evaluator.evaluate(predDF, {evaluator.metricName: "rmse"})
   # Instantiate metrics object
  #evaluation on regression model
  #mlflow is a wrapper , which catches whatever what is available for the fit that can be catched
  #log in of everything you need by mlflow
       #print("  regParam: {}".format(regParam))
  #print("  fitIntercept: {}".format(fitIntercept))
  #print("  elasticNetParam: {}".format(elasticNetParam))
 # print("  mlEstimatorUid: {}".format(mlEstimatorUid))
  #print("  mlModelClass: {}".format(mlModelClass))
#mlflow.log_param("paramGrid",paramGrid) 
#mlflow.log_param("regParam",regParam) 
# #mlflow.log_param("fitIntercept",fitIntercept) 
  #mlflow.log_param("elasticNetParam",elasticNetParam) 
  #mlflow.log_param("mlEstimatorUid",mlEstimatorUid) 
  #mlflow.log_param("mlModelClass",mlModelClass) 
#lr = LinearRegression(featuresCol = "features", labelCol = "price")
#lrModel = lr.fit(vecTrainDF)
#featureCols = ["laps", "fastestLap", "fastestLapSpeed", "statusId", "driverId", "raceId"]
#assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 
#assembled_df = assembler.transform(df2)
 # print("  paramGrid: {}".format(paramGrid))
  #mlflow.log_param("paramGrid",paramGrid)
#lr = LinearRegression(featuresCol = "features", labelCol = "price")
#lrModel = lr.fit(vecTrainDF)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.1,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.2,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.3,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.4,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.5,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.6,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.7,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.8,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=0.9,standardization=False)
#lr = LinearRegression(featuresCol="features",labelCol='positionOrder',predictionCol='predpositionOrder',maxIter=10, regParam=1, elasticNetParam=1,standardization=False)

# COMMAND ----------

#with mlflow.start_run():
    #model1 = tvs.fit(train)
    #mlflow.log_param("rmse",tvs.)
    #mlflow.sklearn.log_model(tvs,"model")
    
    #test_metric = evaluator.evaluate(model1.transform(test))
    #mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric)


# COMMAND ----------

# Explicitly create a new run.
# This allows this cell to be run multiple times.
# If you omit mlflow.start_run(), then this cell could run once, but a second run would hit conflicts when attempting to overwrite the first run.

#with mlflow.start_run():
  # Run the cross validation on the training dataset. The tvs.fit() call returns the best model it found.
  #model1 = tvs.fit(train)
  
  # Evaluate the best model's performance on the test dataset and log the result.
  #test_metric = evaluator.evaluate(model1.transform(test))
  #mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric) 
  
  # Log the best model.
  #mlflow.spark.log_model(spark_model=model1.bestModel, artifact_path='best-model') 
#MLlib will automatically track trials in MLflow. After your tuning fit() call has completed, view the MLflow UI to see logged runs.


