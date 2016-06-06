from pyspark import SparkContext
from pyspark.sql import SQLContext
import pyspark_csv as pycsv
import transformationDT
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.linalg import Vectors
#Creating SparkContext
sc=SparkContext()
sqlContext = SQLContext(sc)
rddUSD = sc.textFile("dataUSDuprv.csv")
rddUSD.persist()
rddUSD.take(5)
#deleting first row(header)
header=rddUSD.first()
dataLines = rddUSD.filter(lambda x: x != header)
dataLines.count()
dataLines.first()
dataLines.take(5)

#RDD to Dense vector
vectorsUSD = dataLines.map(transformationDT.transformToNumeric)
vectorsUSD.take(5)

#Perform statistical Analysis
statsUSD=Statistics.colStats(vectorsUSD)
statsUSD.mean()
statsUSD.variance()
statsUSD.min()
statsUSD.max()
Statistics.corr(vectorsUSD)

#SPARK SQL
dataframe = pycsv.csvToDataFrame(sqlContext, rddUSD, sep=",")
dataframe.registerTempTable("dataUSDuprv")
dff1=sqlContext.sql("SELECT closeJPY FROM dataUSDuprv").show()
dataframe.show()


#LabeledPoint
lpUSD = vectorsUSD.map(transformationDT.transformToLabeledPoint)
lpUSD.take(5)
dfUSD = sqlContext.createDataFrame(lpUSD, ["label", "features"])
dfUSD.select("label", "features").show(10)

#String Indexer
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
si_model = stringIndexer.fit(dfUSD)
td = si_model.transform(dfUSD)
td.collect()
td.show()

#Splitting data
(trainingData, testData) = td.randomSplit([0.6, 0.4])
trainingData.count()
testData.count()
testData.collect()

#Creating decision tree model
dtClassifer = DecisionTreeClassifier(labelCol="indexed",minInstancesPerNode=1500)
dtModel = dtClassifer.fit(trainingData)
dtModel.numNodes
dtModel.depth

#Predict on the test data
predictions = dtModel.transform(trainingData)
predictions = dtModel.transform(testData)
predictions.select("prediction","indexed","label","features").show(10)

#Evaluation
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="indexed",metricName="precision")
evaluator.evaluate(predictions)

#Draw a confusion matrix
labelList=predictions.select("indexed","label").distinct().toPandas()
predictions.groupBy("indexed","prediction").count().show()
