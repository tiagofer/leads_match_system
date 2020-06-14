import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

class RecommenderKNN:
    """
    Classe para a geração de recomendações utilizando o modelo de content based
    """    
    def __init__(self,df_train,df_test,n_neighbors):
        self.df_train = df_train
        self.df_test = df_test
        self.n_neighbors = n_neighbors

    def train_model(self,df_train):
        model_knn = NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=self.n_neighbors,n_jobs=-1)
        model_knn = model_knn.fit(self.df_train)
        return model_knn

    #função para recomendar leads
    def recommender(self,df_test,df_market,modelo):
        distance, index = modelo.kneighbors(self.df_test)
        distance = distance.flatten()
        index = index.flatten()
        recomendados = df_market.iloc[index]
        recomendados['distance'] = distance
        recomendados = recomendados.sort_values(by=['distance'],ascending=True)
        return recomendados['id']
    
    def calculate_accuracy(self,ids_true, ids_pred):
        return len(set(ids_true) & set(ids_pred)) / len(set(ids_true))