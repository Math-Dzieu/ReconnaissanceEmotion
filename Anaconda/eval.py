"""
Import necessaire au fonctionnement du .ipynb
"""
import pickle
from sklearn import metrics
import pandas as pd

def RecoveryFromPickle(filename):
    """
    Cette fonction va recuperer les données contenu dans un fichier .pickle et le retourner
 
    @Param1 String : filename   = Nom du fichier ou l'on récupère les informations
    """
    # Chargement du modele
    with open(filename+'.pickle', 'rb') as f:
        data = pickle.load(f)

    return data

def savePrediction(LR_Model, headerList):
    """
    Cette fonction va recuperer le modele de regression lineaire entrainer dans la partie training ainsi que la list des entêtes 
    que l'on a garde pour pouvoir faire une prediction sur la base de test
 
    @Param1 sklearn.linear_model._logistic.LogisticRegression : LR_Model   = Nom du fichier ou l'on récupère les informations
    @Param2 List : headerList                                              = Liste contenant les nom des entetes que l'on a garde au Training 
    """    
    dfTest = pd.read_csv("./features_test.csv", usecols = headerList)
    dfTest.columns.values

    array = dfTest.values
    X = array[:,1:len(dfTest.columns)]
    
    predictions = LR_Model.predict(X)
    predictions = pd.Series(predictions)    
    predictions.to_csv("./predictions.csv", index=False)
    print(predictions)

#On charge le modele linaire entraine
LR_Model= RecoveryFromPickle('LR_Model')
#On charge la liste des header recupere
listOfTitle=RecoveryFromPickle('listOfIndex')

#On sauvegarde les predictions
savePrediction(LR_Model, listOfTitle)