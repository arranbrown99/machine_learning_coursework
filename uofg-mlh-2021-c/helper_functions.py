from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, roc_curve
from matplotlib.ticker import MultipleLocator

import numpy as np
from matplotlib import pyplot as plt


def evaluate(predicted_y_values, actual_y_values):
    accuracy = accuracy_score(actual_y_values, predicted_y_values)
    sensitivity = recall_score(actual_y_values, predicted_y_values)
    
    print("Accuracy Rate = " + str(accuracy))
    print("Sensitivity Rate = " + str(sensitivity))
    
    
    con_matrix = confusion_matrix(actual_y_values, predicted_y_values)
    
    return (accuracy, sensitivity, con_matrix)


def plot_roc(actual_y_values,predicted_validation_y_probs):
    roc_auc_score_value = roc_auc_score(actual_y_values, predicted_validation_y_probs)
    print("ROC AUC Score= " + str(roc_auc_score_value))
    
    roc_y, roc_x, thresholds = roc_curve(actual_y_values, predicted_validation_y_probs,pos_label = 1)
    plt.plot(roc_x, roc_y, label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def plot_confusion_matrix(con_matrix,labels,text_x=-0.2,text_y=-1):
    print("Confusion Matrix of Classes")
    print(con_matrix)
    print("---")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(con_matrix)
    plt.text(text_x,text_y, 'Confusion matrix of the classifier')
    #plt.title('Confusion matrix of the classifier',loc = 'left')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    plt.xlabel('Predicted Labels')
    plt.ylabel('Test Labels')
    plt.xticks(rotation=90)
    plt.show()
    