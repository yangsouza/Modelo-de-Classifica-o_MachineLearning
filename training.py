import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import joblib


from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#O seguinte projeto é dos alunos @Yang Souza e @Gabriel Consulo

#Definição do problema -> O projeto tem por finalidade fazer uma integração de um modelo de aprendizado, MACHINE LEARNING, através do Scikit Learn, com API em Flask
# e a apresentando dos dados com interface em HTML e CSS. O trabalho tem como objetivo prever se um paciente desenvolverá Diabetes ou não com base em diagnóstico médico.
#Trabalho com base no livro Jornada Python - Uma jornada imersiva na aplicabilidade de uma das linguagens de programação mais poderosas do mundo. 


#Diretório de onde está o dataset usando Pandas
url = 'http://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'

#Informa o cabeçalho das colunas 
colunas = ['preg', 'plas', 'pres','skin', 'test', 'mass', 'pedi', 'age', 'class']

# Tenta ler o arquivo usando diferentes encodings
encodings = ['utf-8', 'latin1', 'ISO-8859-1']

for encoding in encodings:
    try:
        dataset = pd.read_csv(url, names=colunas, skiprows=0, delimiter=',', encoding=encoding)
        print("Arquivo lido com sucesso usando o encoding:", encoding)
        break
    except UnicodeDecodeError:
        print(f"Falha ao tentar usar o encoding {encoding}. Tentando próximo encoding.")

#Análise dos dados
# Exibe as primeiras linhas do dataset para verificar se foi carregado corretamente
print("Linhas do dataset")
print(dataset.head())
print("---------------------------------")

#Dimensões do dataset
print("Dimensões do dataset")
print(dataset.shape)
print("---------------------------------")

#Tipos de cada atributo
print("Tipos de cada atributo do dataset")
print(dataset.dtypes)
print("---------------------------------")

#Tipos de cada atributo
print(dataset)
print("---------------------------------")
#Pré-Processamento de dados para separação em conjunto de treino e conjunto de teste
#Separação em conjuntos de treino e teste
array = dataset.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#Converte os rótulos de string para valores numéricos
le = LabelEncoder()
Y = le.fit_transform(array[:, 8])

#Modelos de classificação
#Criação e avaliação dos modelos -> Linha base será a validação cruzada 10-fold
#Parâmetros a serem utilizados
num_folds = 10
scoring = 'accuracy'

#Criação dos modelos utilizando Regressão Logística, K-vizinhos mais próximos(KNN), Árvores de Classificação (CART), Naive Bayes (NB) e Máquinas de vetores de suporte (VSM)
models = []
models.append(('LR', LogisticRegression(solver='newton-cg')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#Definindo uma semente global para a técnica de validação cruzada
np.random.seed(7)

#Avaliação dos modelos
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#Comparação de modelos
fig = plt.figure()
fig.suptitle('Comparação dos Modelos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Definição da seed global
np.random.seed(7)

#Padronização do dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression(solver='newton-cg'))])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Definição da seed global
np.random.seed(7)

#Tuning do KNN
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
joblib.dump(scaler, 'scaler.joblib')


k = [1,3,5,7,9,11,13,15,17,19,21]
distancias = ["euclidean", "manhattan", "minkowski"]
param_grid = dict(n_neighbors=k, metric=distancias)

model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Melhor: %f usando %s" %(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f(%f): %r" % (mean, stdev, param))
    
#Os resultados mostram que a melhor configuração utiliza a distância de manhattan e k = 17, o que faz com que o algoritmo faça previsões usando as 17 instâncias mais semelhantes

#Tuning do SVM

c_values = [0.1, 0.5, 1.0, 1.5, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel = kernel_values)

model = SVC()
kfold = KFold(n_splits=num_folds)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Melhor: %f usando %s" %(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f(%f): %r" % (mean, stdev, param))

#A configuração que alcançou a maior acurácia foi o modelo que utilizou o kernel linear e C = 0,1. Foi melhor que o resultado anterior só que suficiente pra Regressão Logistica inicial

#Definição da seed global
np.random.seed(7)

#Preparação do modelo
model = LogisticRegression(solver='newton-cg')
model.fit(X_train, Y_train)
joblib.dump(model, 'model.joblib')


#Estimativa da acurácia no conjunto de teste
predictions = model.predict(X_test)
print("Accuracy score = ", accuracy_score(Y_test, predictions))

# Matriz de confusão
cm = confusion_matrix(Y_test, predictions)
labels = ["Sem diabetes", "Com diabetes"]
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot(values_format="d", cmap=plt.cm.Blues, xticks_rotation='horizontal')
plt.show()
print(classification_report(Y_test, predictions, target_names=labels))
 


