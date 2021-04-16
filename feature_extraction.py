"""
Import necessaire au fonctionnement du .ipynb
"""
import pandas as pd
from math import sqrt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import os

def addFeature(FileWithPath, saveName):
    df = pd.read_csv(FileWithPath)
    label = 1
    
    if(saveName=="features_test"):
        label = 0
    else:
        label = 1
    
    listDistFeatureDistance=[2, 48, 14, 54, 48, 54, 8, 57, 50, 58, 52, 56, 33 ,51, 18, 37, 37, 41, 23, 43, 25, 44, 44, 46, 63, 65, 61, 67, 21, 27, 22, 27]
    for feature in range(0, int(len(listDistFeatureDistance)), 2):
        Title = ("d_"+str(listDistFeatureDistance[feature])+"_"+str(listDistFeatureDistance[feature+1]))
        try:
            #print(str(listDistFeatureDistance[feature]), str(listDistFeatureDistance[feature+1]))
            df.insert((len(df.columns)-label), Title, 0)
            df[Title] = df[Title].astype(float)
        except:
            None
        for val in range(len(df)):
            x2, x1 =" x_"+str(listDistFeatureDistance[feature+1]), " x_"+str(listDistFeatureDistance[feature])
            y2, y1 =" y_"+str(listDistFeatureDistance[feature+1]), " y_"+str(listDistFeatureDistance[feature])
            df[Title][val] = sqrt(((df[x2][val] - df[x1][val])**2)+((df[y2][val] - df[y1][val])**2))


    listDistFeaturePente=[49, 48, 54, 53]
    for feature in range(0, int(len(listDistFeaturePente)), 2):
        Title = ("p_"+str(listDistFeaturePente[feature])+"_"+str(listDistFeaturePente[feature+1]))
        try:
            #print(str(listDistFeaturePente[feature]), str(listDistFeaturePente[feature+1]))
            df.insert((len(df.columns)-label), Title, 0)
            df[Title] = df[Title].astype(float)
        except:
            None
        for val in range(len(df)):
            x2, x1 =" x_"+str(listDistFeaturePente[feature+1]), " x_"+str(listDistFeaturePente[feature])
            y2, y1 =" y_"+str(listDistFeaturePente[feature+1]), " y_"+str(listDistFeaturePente[feature])
            df[Title][val] = abs(((df[y2][val] - df[y1][val]) / (df[x2][val] - df[x1][val]))*100)


    df.to_csv("./"+saveName+".csv", index=False, float_format='%.2f')
    
FileWithPath = ["./trainset/trainset.csv", "./testset/testset.csv"]
SavedNames=["trainSetWithFeatures", "features_test"]
FinalSavedNames=["features_train", "features_testUseless"]

for (File, Name, FinalName) in zip(FileWithPath, SavedNames, FinalSavedNames): 
    addFeature(File, Name)

#Suppression des fichiers temporaire (SavedNames)
#os.remove("./trainSetWithFeatures.csv")
