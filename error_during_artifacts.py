import mlflow
print("printing trackinng uri scheme below")
print(mlflow.get_tracking_uri()) # this will be in file format but mlflow expect in http

print('\n')
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("printing new trackinng uri scheme below")
print(mlflow.get_tracking_uri()) 