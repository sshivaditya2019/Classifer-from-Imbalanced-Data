'''Model evaluations'''

from sklearn.metrics import confusion_matrix


def confuseResults(
    yhat,
    ypred
):
    cm = confusion_matrix(y_true=yhat,y_pred=ypred)
    print('Confusion Matrix: \n',cm)
    total = sum(sum(cm))
    print('From Confusion Matric we calculate addtional parameters')
    print('Accuracy : ',(cm[0,0]+cm[1,1])/total)
    print('Sensitivity: ', cm[0,0]/(cm[0,0]+cm[0,1]))
    print('Specificity : ',cm[1,1]/(cm[1,0]+cm[1,1]))
