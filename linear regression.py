from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.stat import Statistics
from pyspark.ml.evaluation import RegressionEvaluator
import transformationLR

#Load the CSV file into a RDD
sc=SparkContext()
sqlContext = SQLContext(sc)
rddUSD = sc.textFile("../Forex DT/data/1440/USD1440.csv")
rddUSD.cache()

#Remove the first line
header=rddUSD.first()
dataLines = rddUSD.filter(lambda x: x != header)
dataLines.take(5)


usdVectors = dataLines.map(transformationLR.transformToNumeric)

#Perform statistical Analysis

usdStats=Statistics.colStats(usdVectors)
usdStats.mean()
usdStats.variance()
usdStats.min()
usdStats.max()
Statistics.corr(usdVectors)
#Transform to a Data Frame for input to Machine Learing
#Drop columns that are not required (low correlation)

    
usdLP = usdVectors.map(transformationLR.transformToLabeledPoint)
usdDF = sqlContext.createDataFrame(usdLP, ["label", "features"])
usdDF.select("label", "features").show(10)

#Split into training and testing data
(trainingData, testData) = usdDF.randomSplit([0.7, 0.3])
trainingData.count()
testData.count()

#Build the model on training data
lr = LinearRegression(maxIter=10)
lrModel = lr.fit(trainingData)
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

#Predict on the test data
predictions = lrModel.transform(testData)
predictions.select("prediction","label","features").show()

evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
evaluator.evaluate(predictions)
#Streaming data

from pyspark.streaming import StreamingContext
ssc=StreamingContext(sc,1)
inputStream=ssc.textFileStream("../Forex DT/data/1440/streaming1440.csv")
def predict(data):
    data = data.map(transformationLR.transformToNumeric)
    data = data.map(transformationLR.transformToLabeledPoint)
    prediction= lrModel.transform(testData)

    for i in range(0, prediction.count()):
        print(prediction.collect()[i])

inputStream.foreachRDD(predict)
ssc.start()
ssc.stop()