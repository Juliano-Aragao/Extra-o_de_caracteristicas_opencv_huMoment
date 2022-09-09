import numpy as np
import os
from glob import glob  #é usado para retornar todos os caminhos de arquivo que
import  cv2
import statistics
import pandas as pd


################################################
# buscando as imagem dentro do arquivo/pasta img
################################################
img_names = glob(os.path.join(os.getcwd(),'img/*.jpg'))

########################################
# verificanco se o arquivo foi importado
########################################
img_names

######################################################
# crinado array vazios para receber as imgens tratadas
######################################################


nome = [] # Variável nome  para identificar quais são saudáveis e quais são doentes
media = [] # Variável Media vai implementar dados estáticos sobre os canais RGB de cada pixel.
hu_momentos = [] # Variável que vai receber os resultados dos cálculos matemáticos dos momentos de hu.


#############################################################
# Laço "for" criado para aplicar o tratamento das carcteristicas
# em cada array especifico 
##############################################################

for fn  in img_names: 
    print(fn)
    img = cv2.imread(fn)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    nome.append(fn)
    media.append(cv2.mean(img))
    hu_momentos.append(cv2.HuMoments(cv2.moments(img_gray)))  
    
############################################
# Transformando dados e dataframe e testando    
############################################
nome

nome = pd.DataFrame(nome, columns =[' imagens']) 
nome
media = pd.DataFrame(media, columns = ['-- R --',"-- G --","-- B --","-- Alpha  --"]) 
media


media.shape # verifincando a quantidade de linhas e colunas

# >>> (600, 4)

################################################################
# Aplicando a extração de caracteristicas pelos 7 momentos de HU
################################################################
hu_momentos  # lendo os dados

type(hu_momentos)# verificando a classe criada

# <class 'list'>

# nos momentos de HU é necessário reorganizar os dados, pelo motivos da extração ser criando em lista dentro de listas
# e não é aceito para transformação do dataframe e para transforma  os dados forma organizada,
# para isso foi criando 7 array vazios e cada momento foi realocado.
''' 
 [ 2.15887662e-24]]), array([[ 1.43305040e-03],
       [ 2.24352170e-09],
       [ 4.75864847e-12],
       [ 3.79894767e-12],
       [-1.19748877e-23],
       [-1.70687118e-16],
       [ 1.08398225e-23]]), array([[ 1.40041088e-03],
       [ 2.71726829e-10],
       [ 7.71341104e-12],
       [ 1.37172841e-11],
       [-1.41088997e-22],
       [ 1.99599192e-16],
       [ 1.72054669e-24]])]
'''

hu_momentos_1 = []
hu_momentos_2 = []
hu_momentos_3 = []
hu_momentos_4 = []
hu_momentos_5 = []
hu_momentos_6 = []
hu_momentos_7 = []


##########################################################################
# Para organizar os dados, foi criado um laço for de cada momento da lista 
# para a variável que foi crianda,  df_momentos 1,2,3,4,5,6,7
##########################################################################

for i in range(len(hu_momentos)):
    hu_momentos_1.append( hu_momentos[i][0][0])
    hu_momentos_2.append( hu_momentos[i][1][0])
    hu_momentos_3.append( hu_momentos[i][2][0])
    hu_momentos_4.append( hu_momentos[i][3][0])
    hu_momentos_5.append( hu_momentos[i][4][0])
    hu_momentos_6.append( hu_momentos[i][5][0])
    hu_momentos_7.append( hu_momentos[i][6][0])
    df_momentos_hu_1 = pd.DataFrame(hu_momentos_1,columns=['hu_1'])
    df_momentos_hu_2 = pd.DataFrame(hu_momentos_2,columns=['hu_2'])
    df_momentos_hu_3 = pd.DataFrame(hu_momentos_3,columns=['hu_3'])    
    df_momentos_hu_4 = pd.DataFrame(hu_momentos_4,columns=['hu_4'])
    df_momentos_hu_5 = pd.DataFrame(hu_momentos_5,columns=['hu_5'])
    df_momentos_hu_6 = pd.DataFrame(hu_momentos_6,columns=['hu_6'])
    df_momentos_hu_7 = pd.DataFrame(hu_momentos_7,columns=['hu_7'])
    
    df_momentos_hu_1
    
    '''  dados organizados
    >>> df_momentos_hu_1
         hu_1
0    0.001335
1    0.001262
2    0.001507
3    0.001252
4    0.001061
..        ...
595  0.001323
596  0.001304
597  0.001499
598  0.001433
599  0.001400

[600 rows x 1 columns]
    '''
    
    
 # criando do dataframe dos momentos de Hu
    
df_momentos = pd.concat([df_momentos_hu_1, df_momentos_hu_2,df_momentos_hu_3,df_momentos_hu_4,df_momentos_hu_5,df_momentos_hu_6,df_momentos_hu_7],axis=1)    

df_momentos
'''
         hu_1          hu_2          hu_3          hu_4          hu_5          hu_6          hu_7
0    0.001335  2.392589e-07  1.203822e-12  4.928146e-12 -5.276225e-24  6.462495e-16  1.078168e-23
1    0.001262  3.317979e-08  1.624090e-13  5.438196e-12  2.607152e-24  3.847247e-16  4.395777e-24
2    0.001507  1.104498e-07  1.136282e-11  5.623857e-13 -8.925280e-25 -1.219461e-16 -1.106571e-24
3    0.001252  1.503350e-07  2.768319e-12  3.191429e-12  8.529867e-24  1.144992e-15 -4.150507e-24
4    0.001061  8.331677e-08  8.839262e-13  2.905004e-12  4.549522e-24  7.706644e-16 -9.857512e-25
..        ...           ...           ...           ...           ...           ...           ...
595  0.001323  4.484459e-09  1.049397e-12  6.518515e-12  6.732490e-24  3.652889e-16 -1.566314e-23
596  0.001304  4.674095e-10  2.018809e-12  1.458214e-12  1.957459e-24 -2.318071e-17 -1.558246e-24
597  0.001499  4.114745e-09  2.091738e-12  8.421922e-12  3.528247e-23 -8.993598e-17  2.158877e-24
598  0.001433  2.243522e-09  4.758648e-12  3.798948e-12 -1.197489e-23 -1.706871e-16  1.083982e-23
599  0.001400  2.717268e-10  7.713411e-12  1.371728e-11 -1.410890e-22  1.995992e-16  1.720547e-24

[600 rows x 7 columns]
'''
###################################################################################
# Criando o dataset, os dados nome não será incluído, por não ter relevancia nas analises futura.
###################################################################################
dataSet_soja = pd.concat([media,df_momentos],axis=1)     
dataSet_soja



'''>>> dataSet_soja
        -- R --     -- G --     -- B --  -- Alpha  --      hu_1          hu_2          hu_3          hu_4          hu_5          hu_6          hu_7
0    105.651484  144.121754  135.220087           0.0  0.001335  2.392589e-07  1.203822e-12  4.928146e-12 -5.276225e-24  6.462495e-16  1.078168e-23
1    109.854948  131.510791  128.853799           0.0  0.001262  3.317979e-08  1.624090e-13  5.438196e-12  2.607152e-24  3.847247e-16  4.395777e-24
2     88.220399  116.544258  104.096593           0.0  0.001507  1.104498e-07  1.136282e-11  5.623857e-13 -8.925280e-25 -1.219461e-16 -1.106571e-24
3    151.666670  185.062793  173.914068           0.0  0.001252  1.503350e-07  2.768319e-12  3.191429e-12  8.529867e-24  1.144992e-15 -4.150507e-24
4    166.370806  192.149739  190.437610           0.0  0.001061  8.331677e-08  8.839262e-13  2.905004e-12  4.549522e-24  7.706644e-16 -9.857512e-25
..          ...         ...         ...           ...       ...           ...           ...           ...           ...           ...           ...
595  141.766830  146.533127  140.605576           0.0  0.001323  4.484459e-09  1.049397e-12  6.518515e-12  6.732490e-24  3.652889e-16 -1.566314e-23
596  144.123093  150.278061  144.385239           0.0  0.001304  4.674095e-10  2.018809e-12  1.458214e-12  1.957459e-24 -2.318071e-17 -1.558246e-24
597  116.063431  129.640900  117.458664           0.0  0.001499  4.114745e-09  2.091738e-12  8.421922e-12  3.528247e-23 -8.993598e-17  2.158877e-24
598  117.338013  131.142197  131.939514           0.0  0.001433  2.243522e-09  4.758648e-12  3.798948e-12 -1.197489e-23 -1.706871e-16  1.083982e-23
599  119.586090  136.987701  130.400696           0.0  0.001400  2.717268e-10  7.713411e-12  1.371728e-11 -1.410890e-22  1.995992e-16  1.720547e-24

[600 rows x 11 columns]
>>>

'''


type(dataSet_soja)

# >>> <class 'pandas.core.frame.DataFrame'>


####################################################
# Para utilizar o classificdor vamos impementar uma nova coluna, " Status" que será definaida com 0 ou 1
# essa coluna tera a função no classificdor para determinar se a folha é ou não doente
#########################################################################################################

dataSet_soja['Status']=0

dataSet_soja

#com a coluna Status crianda, vamos determinar que o 0 indica as folhas saudaveis e  1 as doente, 
# para isso vamos alterar as 300 primeiras linhas

dataSet_soja['Status']=0  # saudáveis
dataSet_soja['Status'][0:299]=1  # doentes

dataSet_soja
'''
>>> dataSet_soja
        -- R --     -- G --     -- B --  -- Alpha  --      hu_1          hu_2          hu_3          hu_4          hu_5          hu_6          hu_7  Status
0    105.651484  144.121754  135.220087           0.0  0.001335  2.392589e-07  1.203822e-12  4.928146e-12 -5.276225e-24  6.462495e-16  1.078168e-23       1
1    109.854948  131.510791  128.853799           0.0  0.001262  3.317979e-08  1.624090e-13  5.438196e-12  2.607152e-24  3.847247e-16  4.395777e-24       1
2     88.220399  116.544258  104.096593           0.0  0.001507  1.104498e-07  1.136282e-11  5.623857e-13 -8.925280e-25 -1.219461e-16 -1.106571e-24       1
3    151.666670  185.062793  173.914068           0.0  0.001252  1.503350e-07  2.768319e-12  3.191429e-12  8.529867e-24  1.144992e-15 -4.150507e-24       1
4    166.370806  192.149739  190.437610           0.0  0.001061  8.331677e-08  8.839262e-13  2.905004e-12  4.549522e-24  7.706644e-16 -9.857512e-25       1
..          ...         ...         ...           ...       ...           ...           ...           ...           ...           ...           ...     ...
595  141.766830  146.533127  140.605576           0.0  0.001323  4.484459e-09  1.049397e-12  6.518515e-12  6.732490e-24  3.652889e-16 -1.566314e-23       0
596  144.123093  150.278061  144.385239           0.0  0.001304  4.674095e-10  2.018809e-12  1.458214e-12  1.957459e-24 -2.318071e-17 -1.558246e-24       0
597  116.063431  129.640900  117.458664           0.0  0.001499  4.114745e-09  2.091738e-12  8.421922e-12  3.528247e-23 -8.993598e-17  2.158877e-24       0
598  117.338013  131.142197  131.939514           0.0  0.001433  2.243522e-09  4.758648e-12  3.798948e-12 -1.197489e-23 -1.706871e-16  1.083982e-23       0
599  119.586090  136.987701  130.400696           0.0  0.001400  2.717268e-10  7.713411e-12  1.371728e-11 -1.410890e-22  1.995992e-16  1.720547e-24       0

[600 rows x 12 columns]
>>>A
'''

########################################################################
# Criando o arquivo CSV  quer será utilizado no teste e no classificador
##########################################################################


dataSet_soja.to_csv('dataSet_folha.csv',index=False)


