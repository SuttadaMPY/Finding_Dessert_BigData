# Databricks notebook source
data = spark.read.csv('dbfs:/FileStore/tables/dishes.csv',header = True)

# COMMAND ----------

data.display()

# COMMAND ----------

# Standardize columns name

new_cols = [c.replace(" ", "_").replace(",", "").replace("#","").replace(".","").replace("/","_") for c in data.columns]
df = data.toDF(*new_cols)

# print the DataFrame to verify the column names have been standardized
df.display()

# COMMAND ----------

# Transform columns into their proper types
# didn't transform columns 6-680 to boolean type because it became true and false in dataframe

from pyspark.sql.functions import col
for column in df.columns[1:]:
    df = df.withColumn(column, col(column).cast("double"))

# COMMAND ----------

df.display()

# COMMAND ----------

# Check for missing values

from pyspark.sql.functions import isnan, when, count, col
missing_count = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).collect()

# Print number of missing values in each column
for col in df.columns:
    print(col + ": " + str(missing_count[0][col]))

# COMMAND ----------

# remove when row that target is null or incorrect

df = df.where("dessert is not null")
df.display()

# COMMAND ----------

# remove row that all features are null

df = df.na.drop()
df.display()

# COMMAND ----------

# Remove rare occurrence features

from pyspark.sql import functions as F
means = df.select(*[F.mean(c).alias(c) for c in df.columns]).first()
cols_to_keep = [c for c in means.asDict() if means[c]== None or means[c] >=0.001]
df2 = df[cols_to_keep]
df2.display()

# COMMAND ----------

len(df2.columns)

# COMMAND ----------

from pyspark.sql.functions import col
 
select_data = df2.where("calories is not null").select("title","calories","protein","fat","sodium","dessert")
select_data.display()

# COMMAND ----------

# Perform feature engineering

select_data = select_data.withColumn("protein_ratio", col("protein")*4 / col("calories"))
select_data = select_data.withColumn("fat_ratio", col("fat")*9 / col("calories"))
select_data = select_data.na.fill(0)
select_data.display()

# COMMAND ----------

from pyspark.sql.functions import split, trim
from pyspark.ml.feature import StopWordsRemover

split_df = select_data.withColumn("desc_array", split(trim("title")," "))
stops = StopWordsRemover(inputCol="desc_array", outputCol="no_stops")
stops_df = stops.transform(split_df)
stops_df.display()

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer(inputCol = "no_stops", outputCol="desc_vec")
cv_model = cv.fit(stops_df)

cv_df = cv_model.transform(stops_df)
cv_df.display()

# COMMAND ----------

# Assemble features
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=['calories','protein_ratio','fat_ratio','sodium','desc_vec'],outputCol='feature')
feature_df = vec_assembler.transform(cv_df)
feature_df.display()

# COMMAND ----------

# Scale features
from pyspark.ml.feature import StandardScaler

std_scaler = StandardScaler(inputCol='feature',outputCol='scaled_features')
scaled_df = std_scaler.fit(feature_df).transform(feature_df)
scaled_df.display()

# COMMAND ----------

from pyspark.ml.classification import LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

dessert_df = scaled_df.selectExpr('title','dessert','scaled_features as features')

# Random split with 80% of training
(train_df,test_df) = dessert_df.randomSplit([0.8,0.2])

string_indexer = StringIndexer(inputCol='dessert',outputCol='label',handleInvalid='skip')

# Create pipeline
svm = LinearSVC(labelCol="label", maxIter=50, regParam=0.1)
pipeline = Pipeline(stages=[string_indexer,svm])

# Train ML model
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

# COMMAND ----------

predictions.select("title","label","prediction").display()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# find accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# find Precision
evaluator.setMetricName("weightedPrecision")
precision = evaluator.evaluate(predictions)

#find recall
evaluator.setMetricName("weightedRecall")
recall = evaluator.evaluate(predictions)

# COMMAND ----------

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
