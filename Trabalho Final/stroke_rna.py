#--------------------------------------------------
# Importanto as bibliotecas Python necessarias ao experimento
# Manipulacao matricial, matemática e visualizacao grafica
#--------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

import random

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification


#--------------------------------------------------
# Processamento dos dados
#--------------------------------------------------
from sklearn.model_selection import train_test_split
#--------------------------------------------------
# Carregando o modelo inteligente e as metricas de desempenho
#--------------------------------------------------
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix
#--------------------------------------------------
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning




# Carregando a base de dados em um objeto DataFrame pertencente a biblioteca Pandas
# Definindo as coulas e retirando a coluna id
#--------------------------------------------------
url = "https://raw.githubusercontent.com/TiagoCSoares/RNA/main/healthcare-dataset-stroke-data.csv"

cols = ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
df = pd.read_csv(url, header= 0, names=cols)
df = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']]

# Convertendo strings em inteiros

df["gender"] = df["gender"].replace({"Male": 1, "Female": 0, "Other": 2})
df["ever_married"] = df["ever_married"].replace({"Yes": 1, "No": 0})
df["work_type"] = df["work_type"].replace({"children": 1, "Govt_job": 0, "Never_worked": 2, "Private": 3, "Self-employed": 4})
df["Residence_type"] = df["Residence_type"].replace({"Rural": 1, "Urban": 0})
df["smoking_status"] = df["smoking_status"].replace({"formerly smoked": 1, "never smoked": 0, "smokes": 2, "Unknown": 3})

# Retirada das linhas que não apresentam valores inteiros

df = df[df['gender'].notna()]
df = df[df['age'].notna()]
df = df[df['hypertension'].notna()]
df = df[df['heart_disease'].notna()]
df = df[df['ever_married'].notna()]
df = df[df['work_type'].notna()]
df = df[df['Residence_type'].notna()]
df = df[df['avg_glucose_level'].notna()]
df = df[df['bmi'].notna()]
df = df[df['smoking_status'].notna()]
df = df[df['stroke'].notna()]



print("Dimensão dos dados:")
print((df).shape)

df.describe().transpose()





#--------------------------------------------------
#Criando matrizes para os recursos e a variável de resposta
#A primeira linha de código cria um objeto da variável de destino chamado 
#'target_column'. 
#--------------------------------------------------
#A segunda linha nos dá a lista de todos os recursos, excluindo a variável de 
#destino 'Outcome'
#A terceira linha normaliza os preditores. 
#A quarta linha exibe o resumo dos dados normalizados. 
#--------------------------------------------------
#Podemos ver que todas as variáveis ​​independentes agora foram dimensionadas entre 0 e 1. 
#A variável de destino permanece inalterada.
#--------------------------------------------------
target_column = ['stroke'] 
#--------------------------------------------------
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df[predictors].describe().transpose()

#Separando as variáveis independentes e dependentes
X = df[predictors]
y = df[target_column]


#Balanceando a base de dados
#Aplicando SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

#Criando o novo dataframe balanceado
df_resampled = pd.DataFrame(X_res, columns=predictors)
df_resampled[target_column] = y_res






entrada_X = df_resampled[predictors].values
saidaDesejada_y = df_resampled[target_column].values
# Aplicando a funcao train_test_split para dividir o conjunto original em 70% para treindo e 30% para teste.
X_train, X_test, y_train, y_test = train_test_split(entrada_X, saidaDesejada_y, test_size=0.3, random_state=43)
#--------------------------------------------------
print("Conjunto de treinamento:")
print(X_train.shape); 
print("Conjunto de teste:")
print(X_test.shape)






#--------------------------------------------------
# Criando duas listas que irão guardar, respectivamente, as melhores acurácias dos melhores testes e os parâmetros utilizados

melhores = [0]*10
melhores_parametros = [0]*10

# São realizados 100 conjuntos de teste e treinamento diferentes
# São definidos parâmetros aleátorios para cada conjunto de teste e treinamento

for i in range(100):
  hls = random.randint(1,15)
  ativacoes = ['identity', 'logistic', 'tanh', 'relu']
  act = random.choice(ativacoes)
  solvers = ['lbfgs', 'sgd', 'adam']
  solv = random.choice(solvers)
  maxi = random.randint(100,300)
  learning_rate = ['constant', 'invscaling', 'adaptive']
  lr = random.choice(learning_rate)
  alpha = random.uniform(0.00001,0.00020)

  warnings.simplefilter(action='ignore', category=DataConversionWarning)
  warnings.simplefilter(action='ignore', category=ConvergenceWarning)
  warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
  warnings.simplefilter(action='ignore', category=FutureWarning)
  warnings.filterwarnings('ignore')

  parametros = [hls, act, solv, maxi, lr, alpha]

  mlp = MLPClassifier(hidden_layer_sizes= hls, activation= act, solver= solv, max_iter=maxi, learning_rate=lr)
  mlp.fit(X_train,y_train)

  #--------------------------------------------------
  predict_train = mlp.predict(X_train)
  predict_test = mlp.predict(X_test)
  #--------------------------------------------------

  #Desempenho conjunto de treinamento

  cm = confusion_matrix(y_train,predict_train)

  accuracy_train = mlp.score(X_train, y_train)

  #Desempenho conjunto de Teste

  cm = confusion_matrix(y_test,predict_test)

  accuracy_test = mlp.score(X_test, y_test)

  i += 1

  # Guardando as melhores acurácias e os parâmetros utilizados
  for a in range(9):
    
    if accuracy_test > melhores[a]:
      melhores[a] = accuracy_test
      melhores_parametros[a] = parametros
      break
    
    a += 1







# As linhas de código abaixo imprimem a matriz de confusão e os 
# resultados do relatório de confusão nos dados de treinamento e teste
# com as melhores precisões.

for i in range(9):
  parametros = melhores_parametros[i]
  
  if parametros == 0:
    break

  mlp = MLPClassifier(hidden_layer_sizes= parametros[0], activation= parametros[1], solver= parametros[2], max_iter= parametros[3], learning_rate= parametros[4])
  mlp.fit(X_train,y_train)

  predict_train = mlp.predict(X_train)
  predict_test = mlp.predict(X_test)

  print("Os parametros usados foram:")
  print("hls = ",parametros[0],"\nact = ",parametros[1],"\nsolver = ", parametros[2],"\nmaxi = ", parametros[3],"\nlearning rate = ", parametros[4],"\nalpha =", parametros[5],"\n")

  #Desempenho conjunto de treinamento
  print("Desempenho do conjunto de treinamento:\n")
  cm = confusion_matrix(y_train,predict_train)
  sns.heatmap(cm,annot=True,fmt="d") 
  print(classification_report(y_train,predict_train))
  plt.show()
  print("\n")


  #Desempenho conjunto de Teste
  print("Desempenho do conjunto de Teste:\n")
  cm = confusion_matrix(y_test,predict_test)
  sns.heatmap(cm,annot=True,fmt="d")
  print(classification_report(y_test,predict_test))
  plt.show()
  accuracy_test = mlp.score(X_test, y_test)
  print("\n\n\n\n\n\n\n\n")

  i += 1