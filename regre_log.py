import pandas as pd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# importatno o data set

dados_reg = pd.read_csv('dataSet_folha.csv')


dados_reg.head()

#separando as colunas em x e y -> predominante da predição
y = dados_reg['Status']  #  variável alvo,  a variável que vai ser prevista se é doente ou saudável
x = dados_reg.drop('Status',axis=1)

# normalizando so dados
scaleX = StandardScaler()
x_train = scaleX.fit_transform(x_train)
x_test = scaleX.fit_transform(x_test)

#  treinamdo o modelo em treino e teste
x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size = 0.2, random_state = 3)

# treinando o modelo


classifier = LogisticRegression()
classifier.fit(x_train, y_train)  # metodo fit e oque vai computar o modelo


#testando valores individuais 
print(classifier.predict([[187.6118790645907,219.09794825758286,206.6462724580695,0.0,0.0010045281130400872,2.433656577897443e-07,3.933738813272526e-12,7.889054441482112e-12,4.356754441114767e-23,3.886861022804572e-15,-5.771077264196654e-24,1
]]))


# criando uma previsão  do y de teste e y de previsão, a reposta é a porcentagem de cada folha em não ser doente
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred) #  vai mostrar qual foi a probabilidade de acertos e erros
print(cm) # Imprimindo a martiz de confusão

'''
>>> print(cm)
[[118   6]
 [  3 113]]
>>>

>>> print(accuracy_score(y_test, y_pred))
0.49585  #  49% de de probabilidade, Olho de Rã
>>>
>>> print(accuracy_score(y_test, y_pred))
0.3798745  #  37% de probabilidade, Mancha Parda
>>>
>>> print(accuracy_score(y_test, y_pred))
0.475475  #  47% de probabilidade, Mildio
>>>



187.6118790645907,219.09794825758286,206.6462724580695,0.0,0.0010045281130400872,2.433656577897443e-07,3.933738813272526e-12,
7.889054441482112e-12,4.356754441114767e-23,3.886861022804572e-15,-5.771077264196654e-24




>>>





'''









print(accuracy_score(y_test, y_pred))#  impriminda  a porcentagem de acerto

'''
>>> print(accuracy_score(y_test, y_pred))
0.9625
>>>
'''

#testando valores individuais 

print(classifier.predict([[ 88.22039886738077,116.54425823580814,104.09659312841777,0.0,0.0015067008601716537,1.10449762157323e-07,1.1362822443429107e-11,5.62385713445496e-13,-8.92528013101697e-25,-1.2194614910835455e-16,-1.1065710983318196e-24
 ]]))


'''





melhor visualizdo no spyder,  vai mostra a probabilidade de cada imagem 

y_pred_prob = classifier.predict_proba(x_test)   #  vai prever em % de ser doente"1" ou saudáveç "0"

print(y_pred)



y_pred_prob = y_pred_prob[:,1]

print y_pred_prob


#  concatenando y  para ver os resultados

y_result = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)'''