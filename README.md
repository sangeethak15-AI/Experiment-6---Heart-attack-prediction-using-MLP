# Experiment-6---Heart-attack-prediction-using-MLP
## Aim:
To construct a  Multi-Layer Perceptron to predict heart attack using Python
## Algorithm:
1. Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<br>
2. Load the heart disease dataset from a file using pd.read_csv().<br>
3. Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<br>
4. Split the dataset into training and testing sets using train_test_split().<br>
5. Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<br>
6. Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<br>
7. Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<br>
8. Make predictions on the testing set using mlp.predict(X_test).<br>
9. Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<br>
10. Print the accuracy of the model.<br>
11. Plot the error convergence during training using plt.plot() and plt.show().<br>

## Program:
```python
Developed By: Sangeetha.K
Register No. 212221230085
import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data=pd.read_csv("heart.csv")
X=data.iloc[:, :-1].values #features 
Y=data.iloc[:, -1].values  #labels 

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

mlp=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
training_loss=mlp.fit(X_train,y_train).loss_curve_

y_pred=mlp.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Losss")
plt.show()

```
## Output:
### Loss Convergence graph
<img width="449" alt="image" src="https://github.com/Shavedha/Experiment-6---Heart-attack-prediction-using-MLP/assets/93427376/0f5ac0c2-2f62-41b1-a410-2fda3ba4f556">

### Accuracy
<img width="234" alt="image" src="https://github.com/Shavedha/Experiment-6---Heart-attack-prediction-using-MLP/assets/93427376/40a192f3-3dce-4b50-aefe-5172c816b091">

## Result:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
     
