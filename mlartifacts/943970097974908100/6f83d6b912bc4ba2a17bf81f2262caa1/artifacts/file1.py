import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")


wine=load_wine()
X=wine.data
y=wine.target

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20, random_state=42)

max_depth=9
n_estimators=11

# mention your experimment
mlflow.set_experiment("MLOPS-MLFLOW-EXP1")

with mlflow.start_run(): 
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    
    mlflow.log_metric('Accuracy', accuracy)
    mlflow.log_param('max depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

       # creatining a confusion metrx: 
    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel('Predicted')
    plt.title("confusion metrix")

    # save fig
    plt.savefig('Confusion-matrix.png')

    # log artifacts using mflow
    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact(__file__)

    print(accuracy)