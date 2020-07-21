# Com NaiveBayes é formada uma tabela que calcula as possibilidades a partir dos dados de treino a classificacao dos dados de teste
# NaiveBayes trabalha com campos INDEPENDENTES, isto é, não autocorrelacionados

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv("/Users/Dennis/Desktop/dados/insurance.csv")
base = base.drop(columns = ['Unnamed: 0']) # Remove a coluna que nao serve pra nada

base.Accident.unique() # Mostra quais tipos de acidentes

x = base.iloc[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values # Pega as linhas
y = base.iloc[:, 7].values # Pega as linhas mas so com a colunas 7

labelencoder = LabelEncoder() # A Maioria dos algoritmos requerem dados numericos, estão em String, logo mudaremos isso

# APlicando a função que transforma as Strings para numéricos
x[:,0] = labelencoder.fit_transform(x[:,0])
x[:,1] = labelencoder.fit_transform(x[:,1])
x[:,2] = labelencoder.fit_transform(x[:,2])
x[:,3] = labelencoder.fit_transform(x[:,3])
x[:,4] = labelencoder.fit_transform(x[:,4])
x[:,5] = labelencoder.fit_transform(x[:,5])
x[:,6] = labelencoder.fit_transform(x[:,6])
x[:,7] = labelencoder.fit_transform(x[:,7])
x[:,8] = labelencoder.fit_transform(x[:,8])
x[:,9] = labelencoder.fit_transform(x[:,9])
x[:,10] = labelencoder.fit_transform(x[:,10])
x[:,11] = labelencoder.fit_transform(x[:,11])
x[:,12] = labelencoder.fit_transform(x[:,12])
x[:,13] = labelencoder.fit_transform(x[:,13])
x[:,14] = labelencoder.fit_transform(x[:,14])
x[:,15] = labelencoder.fit_transform(x[:,15])
x[:,16] = labelencoder.fit_transform(x[:,16])
x[:,17] = labelencoder.fit_transform(x[:,17])
x[:,18] = labelencoder.fit_transform(x[:,18])
x[:,19] = labelencoder.fit_transform(x[:,19])
x[:,20] = labelencoder.fit_transform(x[:,20])
x[:,21] = labelencoder.fit_transform(x[:,21])
x[:,22] = labelencoder.fit_transform(x[:,22])
x[:,23] = labelencoder.fit_transform(x[:,23])
x[:,24] = labelencoder.fit_transform(x[:,24])
x[:,25] = labelencoder.fit_transform(x[:,25])

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y,test_size=0.3,random_state= 0) # com 0 teremos sempre os mesmos registro, como no exemplo do video

modelo = GaussianNB()
modelo.fit(x_treinamento,y_treinamento) # Criando a tabela de probabilidade Naive Bayes

previsoes = modelo.predict(x_teste)