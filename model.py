import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayesModel, MultilayerPerceptronClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

data = spark.read.format('csv').option(
    "header", "True").load("./dataset/playerData.csv")


def dataCleaning(data):
    data = data.withColumn("Wage", regexp_replace(data["Wage"], "[^0-9]+", ""))
    data = data.withColumnRenamed("Wage", "Wage in Millions")
    data = data.withColumn("Value", regexp_replace(
        data["Value"], "[^0-9]+", ""))
    data = data.withColumnRenamed("Value", "Value in Thousands")

    def evaluateAttr(val):
        return eval(val)

    evaluateAttribute = udf(lambda x: evaluateAttr(x), IntegerType())

    cols = ['Age', 'overall', 'potential', 'Wage in Millions', 'Value in Thousands', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning',
            'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys']
    for column in cols:
        data = data.withColumn(column, evaluateAttribute(column))

    for column in cols:
        data = data.withColumn(column, data[column].cast("Integer"))

    data.na.fill(value=0)

    defenders = ['LWB', 'RWB', 'CB', 'LB', 'RB', 'GK']
    midfielders = ['LM', 'CM', 'RW', 'CDM']
    strikers = ['LW', 'CAM', 'CF', 'ST']

    def getPosition(pos):
        if pos in defenders:
            return "Defender"
        elif pos in midfielders:
            return "Midfielder"
        else:
            return "Striker"

    convergePosition = udf(lambda x: getPosition(x), StringType())

    data = data.withColumn("Pos", convergePosition("Position"))
    return data


print("Initiating Data Cleanup...")
data = dataCleaning(data)
print("Data Cleanup Complete")


print("Creating Data Pipeline...")


def createPipeline(cols):
    Nationality = StringIndexer(
        inputCol='Nationality', outputCol='Nationality_idx')
    label = StringIndexer(inputCol='Pos', outputCol='label')
    assembler = VectorAssembler(inputCols=cols, outputCol='features')
    pipeline = Pipeline(stages=[Nationality, label, assembler])

    return pipeline


cols = ['Nationality_idx', 'Age', 'overall', 'potential', 'Wage in Millions', 'Value in Thousands', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking',
        'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys']
pipeline = createPipeline(cols)
model = pipeline.fit(data).transform(data)
print("Data Pipeline Transform Complete...")

model = model.select(model.features, model.label)


def evaluateResults(ground_truth, pred, model):
    print(model + " Model Results: ")
    print(classification_report(ground_truth, pred,
          target_names=["Defender", "Striker", "Midfielder"]))


def predFunctions(train, test, model):
    # logistic Regression Model
    print("Initiating Model")
    if model == 'Logistic-Regression':
        ml = LogisticRegression()
    # Naive-Bayes Model
    elif model == 'Naive-Bayes':
        ml = NaiveBayes(smoothing=1)
    elif model == 'Neural-network':
        # Define the layers of the neural network
        layers = [train.toPandas().features[0].size, 30, 30, 10, 3]
        ml = MultilayerPerceptronClassifier(
            maxIter=1000, layers=layers, seed=1234, labelCol="label", solver="gd", stepSize=0.01, blockSize=100)
    else:
        return "Model not valid"

    print("Training Model")
    clf = ml.fit(train)
    clf.write().overwrite().save("./best-models/" + model)
    print("Testing Model")
    results = clf.transform(test)
    evaluateResults(test.toPandas().label,
                    results.toPandas().prediction, model)
    return clf, results


(train, test) = model.randomSplit([0.8, 0.2], seed=123)

print("Logistic Regression Model")
predFunctions(train, test, "Logistic-Regression")
print("\nNaive - Bayes Model \n")
predFunctions(train, test, "Naive-Bayes")
print("\nNeural Network Model \n")
predFunctions(train, test, "Neural-network")
