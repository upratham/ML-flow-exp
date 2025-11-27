import mlflow
from sklearn.datasets import load_wine
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns

#import wine data

wine=load_wine()
X=wine.data
y=wine.target

#Train text split 
test_size=0.2
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=21)

# Hyperparameters

max_depth=5
n_estimators=100

with mlflow.start_run():
    rfc=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,random_state=1)
    rfc.fit(X_train,y_train)
    y_pred= rfc.predict(X_test)
    accuracy= accuracy_score(y_test,y_pred)
    
    mlflow.log_metric('Accuracy', accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('Test Size',test_size)
    #create Confusion matrix plot
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title('Confusion matrix')
    #save plot
    plt.savefig('Confusionmatrix.png')
    #log artifcats using mlflow
    mlflow.log_artifact('Confusionmatrix.png')
    mlflow.log_artifact(__file__)
    print('acuuracy:',accuracy)
