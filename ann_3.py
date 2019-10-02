# Part 1: Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing Dataset
dataset = pd.read_csv('creditcard_1.csv')
X = dataset.iloc[:, 0:30].values
y = dataset.iloc[:, 30].values


# Splitting the dataset into Training Set and Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Part 2:  Make an ANN

# Import the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Initializing the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer wih Dropout
classifier.add(Dense(output_dim = round(15.5), init = 'uniform', activation = 'relu', input_dim = 30 ))
classifier.add(Dropout(p = 0.1 ))

# Adding 2nd hidden layer
classifier.add(Dense(output_dim = round(15.5), init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1 ))

# Adding the ouput layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 25)

# Part 3: Making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


new_prediction = classifier.predict(sc.transform(np.array([[172792
, -0.533412522, -0.189733337, 0.703337367, -0.50627124, -0.012545679, -0.649616686,
 1.577006254, -0.414650408, 0.486179505, -0.915426649, -1.040458335, -0.031513054,
-0.188092901, -0.08431647, 0.041333455, -0.302620086, -0.660376645, 0.167429934,
-0.256116871, 0.382948105, 0.261057331, 0.643078438, 0.376777014,0.008797379,
-0.473648704, -0.818267121, -0.002415309, 0.013648914, 217 ]])))
new_prediction = (new_prediction > 0.5)



# Making the confusion metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Part 4- Evaluating,Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = round(15.5), init = 'uniform', activation = 'relu', input_dim = 30 ))
    classifier.add(Dense(output_dim = round(15.5), init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, nb_epoch = 10)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()  



# Improving the ANN
# Dropout regularization to reduce overfitting if needed



# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = round(15.5), init = 'uniform', activation = 'relu', input_dim = 30 ))
    classifier.add(Dense(output_dim = round(15.5), init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier )
parameters = {'batch_size' : [25,32],
              'nb_epoch' : [10,20],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_

























