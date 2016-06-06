from pyspark.mllib.linalg import Vectors

def transformToNumeric(inputStr):
    attList = inputStr.split(",")
    # Creating trend
    trendGBP = -1.0 if float(attList[0]) < float(attList[3]) else 1.0
    trendEUR = -1.0 if float(attList[4]) < float(attList[7]) else 1.0
    trendAUD = -1.0 if float(attList[8]) < float(attList[11]) else 1.0
    trendNZD = -1.0 if float(attList[12]) < float(attList[15]) else 1.0
    trendCAD = 0.0 if float(attList[16]) < float(attList[19]) else -1.0
    trendCHF = 1.0 if float(attList[20]) < float(attList[23]) else -1.0
    trendJPY = 1.0 if float(attList[24]) < float(attList[27]) else -1.0
    # Global_trend=global weakness/strength
    Global_trend=float(trendGBP+trendEUR+trendAUD+trendNZD+trendCAD+trendCHF+trendJPY)
    # Target
    USD_P = float(attList[29])


    # Filter out columns not wanted at this stage
    values = Vectors.dense([USD_P,trendGBP,trendEUR,trendAUD,trendNZD,trendCAD,trendCHF,trendJPY, \
                            Global_trend])
    return values

def transformToLabeledPoint(inStr) :
    lp = ( float(inStr[0]), \
    Vectors.dense([inStr[1],inStr[2],inStr[3], \
        inStr[4],inStr[5],inStr[6],inStr[7], \
        inStr[8]]))
    return lp