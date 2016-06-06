from pyspark.mllib.linalg import Vectors


def transformToNumeric(inputStr):
    attList = inputStr.split(",")

    # Filter out columns not wanted at this stage
    values = Vectors.dense([float(attList[28]), \
                            float(attList[4]),float(attList[5]), float(attList[6]), float(attList[7])])
    return values

def transformToLabeledPoint(inStr) :
    lp = (float(inStr[0]), Vectors.dense([inStr[1],inStr[2],inStr[3],inStr[4]]))
    return lp