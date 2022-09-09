import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.model_selection as ms
import matplotlib.pyplot as plt # Bilitioteca utilçizada para exibir resukltados das analizes 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
dados = pd.read_csv('dataSet_folha.csv')
dados.head()

y = dados['Status']  
x = dados.drop('Status',axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.3)

scaleX = StandardScaler()  # normalizados de dados
x_train = scaleX.fit_transform(x_treino) # fit vai analisara o conjutos de dados com modelos estatisticos, media, variancia, maximo e minimo

X_test = scaleX.fit_transform(x_teste) #  o transform vai aplicar as infomrações nos dados 

modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino,y_treino)
resultado = modelo.score(x_teste, y_teste)

y_teste[10:15]
x_teste[10:15]

print('Acuracia', resultado)

previsoes = modelo.predict(x_teste[10:15]) 

previsoes

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size = 0.4, random_state = 3)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)  # metodo fit vai treina o modelo

y_pred = classifier.predict(x_test)

# criando uma previsão  do y de teste e y de previsão, a reposta é a porcentagem de cada folha em não ser doente
cm = confusion_matrix(y_test, y_pred) #  vai mostrar qual foi a probabilidade de acertos e erros

print(cm) # Imprimindo a martiz de confusão

print(accuracy_score(y_test, y_pred))#  impriminda  a porcentagem de acerto

#y_pred_prob = classifier.predict_proba(X_test)   #  proba

#y_pred_prob = y_pred_prob[:,1]

#testando valores individuais 

print(classifier.predict([[  ''' INSERIR OS DADOS ENTRE VIRGULAS '''  ]]))

print(y_pred)
print(y_pred_prob)