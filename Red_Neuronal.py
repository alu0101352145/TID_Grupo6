'''
  Grupo 6.
  Participantes:
    Juan Guillermo Zafra Fernández
    Juan Salvador Magariños Alba
    Jorge Cabrera Rodríguez
    Jorge González Delgado
  Base de datos:
    Semillas de trigo.

  Este código genera una red neuronal para hacer un estudio de las características de los cultivos de semillas de trigo.
'''

# 1. Carga de librerías

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from datetime import datetime

# Constantes

DELIMITADOR = '\n######################################################################\n'

# 2. Carga de datos
print('Cargando los datos...')
seed_stats = np.loadtxt("seeds_dataset.txt", dtype=float, delimiter='\t')
print(seed_stats)
(n_samples, n_features) = seed_stats.shape
print(f'Rows: {n_samples}')
print(f'Columns: {n_features}')

print(DELIMITADOR)

# 3. Preprocesado de datos
print('Preprocesando los datos...')
# train_size: porcentaje del conjunto de datos utilizado para el entrenamiento
# other_size: porcentaje del conjunto de datos utilizado para la validación y tests
# valid_size: porcentaje del conjunto "other" utilizado para la validación
# test_size: porcentaje del conjunto "other" utilizado para los tests

train_size = 0.5
other_size = 0.5
valid_size = 0.5
test_size = 0.5

type_column = seed_stats[:, n_features - 1]
seed_stats_normalized = preprocessing.normalize(seed_stats)
seed_stats_normalized[:, n_features - 1] = type_column

patterns_input = seed_stats_normalized[:, :n_features-1]
patterns_target = seed_stats_normalized[:, -1]

# como debemos separar el conjunto de datos total en datos de entrenamiento, validación
# y tests, separamos primero entre los que son de entrenamiento y el resto
input_train, input_others, target_train, target_others = train_test_split(
    patterns_input, patterns_target, train_size = train_size, test_size = other_size,
    random_state = 0, shuffle = True)

input_valid, input_test, target_valid, target_test = train_test_split(
    input_others, target_others, train_size = valid_size, test_size = test_size,
    random_state = 0, shuffle = True)

# 4. Resultados iniciales Perceptrón Simple

max_iter = 60
print("Learning a Perceptron with %d maximum number of iterations and ..." % max_iter)
perceptron = Perceptron(max_iter = max_iter, shuffle = False, random_state = 0, verbose = True)
perceptron.fit(input_train, target_train)

# Results
print("\n~~~ Printing Perceptron results ~~~\n")
predict_train = perceptron.predict(input_train)
predict_valid = perceptron.predict(input_valid)
print("Train accuracy: %.3f%%" % (accuracy_score(target_train, predict_train) * 100))
print("Valid accuracy: %.3f%%" % (accuracy_score(target_valid, predict_valid) * 100))

# 5. Resultados iniciales Perceptrón Multicapa

# Modelar el perceptrón multicapa

def MLP_train_valid(mlp, input_train, target_train, input_valid, target_valid, max_iter, valid_cycles, verbose):
    """
    Train and valid MLP every valid_cycles iterations
    """
    classes  = np.unique(target_train)
    loss_valid = []
    for i in range(max_iter//valid_cycles):
        for j in range(valid_cycles):
            out = mlp.partial_fit(input_train, target_train, classes)
            # Calculate loss function of valid set
        last_lost_valid = log_loss(target_valid, mlp.predict_proba(input_valid))
        loss_valid.append(last_lost_valid)
        if verbose:
            print("Iteration %d, train loss = %.8f, valid loss = %.8f" %
                  (mlp.n_iter_, mlp.loss_, last_lost_valid))
        if early_stopping and (i > 0) and (last_lost_valid > loss_valid[-2]): # Early stopping
            if verbose:
                print("Early stopping: Validation score did not improve")
            break
    if verbose: print(out)
    
    if verbose:
        # Visualizing the Cost Function Trajectory
        print("Visualizing the Cost Function Trajectory")
        plt.plot(range(1, len(mlp.loss_curve_)+1), mlp.loss_curve_, label='Train loss')
        plt.plot(range(valid_cycles,len(loss_valid)*valid_cycles+valid_cycles,valid_cycles), loss_valid, '-o', label='Valid loss')
        plt.xlabel('number of iterations')
        plt.ylabel('loss function')
        plt.legend(loc='upper right')
        plt.show()

# Multilayer Percetron wiht n_hidden hidden neurons
n_hidden = 80
max_iter = 300
learning_rate_init = 0.01
valid_cycles = 5
early_stopping = True

print("\n~~~ Learning a MLP with %d hidden neurons, %d maximum number of iterations and %.8f learning rate ... ~~~\n" % (n_hidden, max_iter, learning_rate_init))

mlp = MLPClassifier(hidden_layer_sizes = (n_hidden,), learning_rate_init = learning_rate_init, shuffle = False, random_state = 0, verbose = False)

MLP_train_valid(mlp, input_train, target_train, input_valid, target_valid, max_iter, valid_cycles, True)

# Resultados

print("\n~~~ Printing initial results~~~\n")

predict_train = mlp.predict(input_train)
predict_valid = mlp.predict(input_valid)

print("Train accuracy: %.3f%%" % (accuracy_score(target_train, predict_train) * 100))
print("Valid accuracy: %.3f%%" % (accuracy_score(target_valid, predict_valid) * 100))

print("\nTrain confusion matrix:")
print(confusion_matrix(target_train, predict_train))
print("Valid confusion matrix:")
print(confusion_matrix(target_valid, predict_valid))

print("Train classification report:")
print(classification_report(target_train, predict_train))
print("Valid classification report:")
print(classification_report(target_valid, predict_valid))

# 6. Optimización ratio de aprendizaje

print("\n~~~ Optimización del ratio de aprendizaje ~~~\n")

tests_learning_rate_init = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15]
activation = 'relu'
random_state = 0

now = datetime.now()
loss_curves =  []
for lr in tests_learning_rate_init:
    mlp = MLPClassifier(hidden_layer_sizes = (n_hidden,), learning_rate_init = lr, shuffle = False, random_state = random_state, verbose = False, activation=activation)
    MLP_train_valid(mlp, input_train, target_train, input_valid, target_valid, max_iter, valid_cycles, False)
    
    loss_curves.append(mlp.loss_curve_)

print("Number of seconds for training: %d" % (datetime.now() - now).total_seconds())

# Show results
print("Visualizing the Cost Function Trajectory with different learning rates")
for (lr, loss_curve) in zip(tests_learning_rate_init, loss_curves):
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, label = 'larning rate = ' + str(lr))

plt.xlabel('number of iterations')
plt.ylabel('loss function')
plt.legend(loc = 'upper right')
plt.show()

# 7. Optimización arquitectura

print("\n~~~ Optimización de la arquitectura ~~~\n")

# Test MLP with differents number of hidden units and several repetitions
tests_n_hidden = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]
n_reps = 10
# n_reps = 20
activation = 'relu'
# activation = 'logistic'
# learning_rate_init = 0.001
# learning_rate_init = 0.01
learning_rate_init = 0.005

now = datetime.now()
best_mlp = []
best_acc = 0.0
accs_train = []
accs_valid = []
for n_hidden in tests_n_hidden:
    max_acc_train = max_acc_valid = 0.0
    for random_state in range(n_reps):
        mlp = MLPClassifier(hidden_layer_sizes=(n_hidden,), learning_rate_init=learning_rate_init, shuffle=False, random_state=random_state, verbose=False, activation=activation)
        MLP_train_valid(mlp, input_train, target_train, input_valid, target_valid, max_iter, valid_cycles, False)
        
        acc_train = accuracy_score(target_train, mlp.predict(input_train))
        acc_valid = accuracy_score(target_valid,mlp.predict(input_valid))
        print("Seed = %d, train acc = %.8f, valid acc = %.8f, iterations = %d" % (random_state, acc_train, acc_valid, len(mlp.loss_curve_)))
        if (max_acc_valid < acc_valid):
            max_acc_valid = acc_valid
            max_acc_train = acc_train
            if (acc_valid > best_acc):
                best_acc = acc_valid
                best_mlp = mlp
    accs_train.append(max_acc_train)
    accs_valid.append(max_acc_valid)
    print("Number hidden units = %i, train acc = %.8f, max valid acc = %.8f" % (n_hidden, max_acc_train, max_acc_valid))

print("Number of seconds for training: %d" % (datetime.now() - now).total_seconds())
print("Best MLP valid accuracy: %.8f%%" % (best_acc * 100))
print("Best MLP: ", best_mlp)

# Show results

width = 4
plt.bar(np.array(tests_n_hidden) - width, 100 *(1- np.array(accs_train)), color='g', width=width, label='Train error')
plt.bar(np.array(tests_n_hidden), 100 *(1- np.array(accs_valid)), width=width, label='Min valid error')
plt.xlabel('number of hidden units')
plt.ylabel('error (%)')
plt.xticks(np.array(tests_n_hidden), tests_n_hidden)
plt.legend(loc = 'upper right')
plt.show()

# 8. Resultados finales del mejor MLP

print("\n~~~ Imprimimos los resultados finales del mejor MLP ~~~\n")

predict_train = best_mlp.predict(input_train)
predict_valid = best_mlp.predict(input_valid)
predict_test = best_mlp.predict(input_test)

print("Train accuracy: %.3f%%" % (accuracy_score(target_train, predict_train) * 100))
print("Valid accuracy: %.3f%%" % (accuracy_score(target_valid, predict_valid) * 100))
print("Test accuracy: %.3f%%" % (accuracy_score(target_test, predict_test) * 100))

print("Train confusion matrix:")
print(confusion_matrix(target_train, predict_train))
print("Valid confusion matrix:")
print(confusion_matrix(target_valid, predict_valid))
print("Test confusion matrix:")
print(confusion_matrix(target_test, predict_test))

print("Train classification report:")
print(classification_report(target_train, predict_train))
print("Valid classification report:")
print(classification_report(target_valid, predict_valid))
print("Test classification report:")
print(classification_report(target_test, predict_test))

# ROC curves of test set
mlp_probs = mlp.predict_proba(input_test)
classes  = np.unique(target_train)
mlp_auc = []
mlp_fpr = []
mlp_tpr = []
for cla in classes:
    mlp_auc.append(roc_auc_score(target_test==cla, mlp_probs[:,int(cla) - 1]))
    fpr, tpr, _ = roc_curve(target_test==cla, mlp_probs[:,int(cla) - 1])
    mlp_fpr.append(fpr)
    mlp_tpr.append(tpr)

print("\n~~~ Imprimimos las curvas ROC del conjunto de testeo ~~~\n")
# plot the roc curve for the model
for cla in classes:
    # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(mlp_fpr[int(cla) - 1], mlp_tpr[int(cla) - 1], marker = '.', label = 'Class %d (AUC: %.5f)' % (cla, mlp_auc[int (cla) - 1]))

plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
