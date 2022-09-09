import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

'''
import sklearn.model_selection as ms

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn import linear_model as lm
from sklearn import model_selection as ms

from matplotlib.axis import Axis
from matplotlib.pyplot import axis'''

dados = pd.read_csv('dataSet_folha.csv')
dados.head()


# separando as variáveis entre preditoras e variavel alvo 

y = dados['Status']  #  variável alvo,  a variável que vai ser prevista se é doente ou saudável
x = dados.drop('Status',axis=1)
y  #   0 - 1
y.shape
y.describe
#  separando so dados em treino e teste

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.3)

scaleX = StandardScaler()
x_train = scaleX.fit_transform(x_treino)
X_test = scaleX.fit_transform(x_teste)

x_treino.shape
x_teste.shape
y_treino.shape
y_teste.shape


###################################################
# criando o modelo de ML  - ExtraTreesClassifier
#################################################
modelo = ExtraTreesClassifier(n_estimators=100)
modelo.fit(x_treino,y_treino)

resultado = modelo.score(x_teste, y_teste)

print('Acuracia', resultado)

y_teste[10:15]
x_teste[10:15]



previsoes = modelo.predict(x_teste[10:15])

previsoes


 