# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SADHANA K

### Register Number: 212225240128

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Exp-1.csv")
df

x = df[["Input"]].values
y = df[["Output"]].values

x_test, x_train, y_test, y_train = train_test_split(x,y,test_size=0.4,random_state=42)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

x_train = scaler1.fit_transform(x_train)
x_test = scaler2.fit_transform(x_test)


x_train1 = torch.FloatTensor(x_train)
x_test1 = torch.FloatTensor(x_test)
y_train1 = torch.FloatTensor(y_train)
y_test1 = torch.FloatTensor(y_test)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
    def forward(self,x):
        return self.network(x)


model = neuralnet()
criterion =nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 1000
loss_history = []

for i in range(epochs):
    optimizer.zero_grad()
    pred = model(x_train1)
    loss = criterion(pred,y_train1)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if i%100==0:
        print(f"Epoch{i}:",loss.item())


with torch.no_grad():
    pred=model(x_test1)
    loss=criterion(pred,y_test1)
    print(loss.item())

model.eval()

input = torch.tensor(scaler1.transform([[22]]), dtype=torch.float32)

with torch.no_grad():
    pred = model(input)

print(pred.item())

import matplotlib.pyplot as plt

plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.grid(True)
plt.show()


```

### Dataset Information
Include screenshot of the generated data

### OUTPUT
#### Dataset Information:
<img width="285" height="771" alt="image" src="https://github.com/user-attachments/assets/19fb23a8-1d93-43f7-866e-0587fe4d10b9" />
#### Epoch:
<img width="388" height="290" alt="image" src="https://github.com/user-attachments/assets/6dd91b8d-93bb-4586-b21f-185ce4cac938" />
#### New Sample Data Prediction:
<img width="277" height="212" alt="image" src="https://github.com/user-attachments/assets/0565b313-b2b9-42b2-be3d-b7794ba9b12f" />


#### Graph:
<img width="922" height="632" alt="image" src="https://github.com/user-attachments/assets/44ccc1f1-4477-4386-9bb0-1236927de9b7" />




## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
