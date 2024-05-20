# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Set the maximum number of rows to display
pd.set_option('display.max_rows', 20)  # None means show all rows

# Set the maximum number of columns to display
pd.set_option('display.max_columns', 20)  # None means show all columns
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import os

pd.set_option('display.max_rows', 5)  # None means show all rows

# Set the maximum number of columns to display
pd.set_option('display.max_columns', 5)  # None means show all columns

# Mycomputer doesn't have gpu
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device\n----------------------------------------------------------------------------------------")
# -----------------------------------------------------------------------------------------

#  load files and convert to dataframes; we use df for data frame
# below is the command for my laptop

df = pd.read_csv('/kaggle/input/test-train-data/train.csv')
df_test = pd.read_csv('/kaggle/input/test-train-data/test.csv')



#df_test = pd.read_csv("./test.csv")
#df= pd.read_csv("./train.csv") #df will be our train data

#df_test = pd.read_csv("F:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/test.csv")
#df= pd.read_csv("F:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/train.csv") #df will be our train data

# df_test = pd.read_csv("D:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/test.csv")
# df = pd.read_csv("D:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/train.csv")

print("\nDescription of Training Dataframe\n")
print(f"{df.describe()} \n----------------------------------------------------------------------------------------")

# ----------------------------------------------------------------------------------------
print("\nALL attirbutes\n")
print(f"{df.columns}\n----------------------------------------------------------------------------------------\n")
# ----------------------------------------------------------------------------------------
# use this method for words
label = 'LotArea'
label_relation_to_sales_price = df.groupby('SalePrice')[label].mean()
print(f"{label} (column 2),\n Sale Price (column 1)\n{label_relation_to_sales_price}\n----------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------

#we now look at a specific numerical case to get a better idea of what to do
plt.scatter(x='LotArea', y='SalePrice', data=df, label='House Prices vs Lot Area')
plt.title('Example Data Distibution')  # Title
plt.xlabel('Lot Area (x)')
plt.ylabel('Sale Price (y)')
plt.legend(title="Legend", loc='lower right', fontsize='x-small')
plt.show()


# ----------------------------------------------------------------------------------------
# we will first deal with everything with numerical values
sales_column = df["SalePrice"]
# connect all the data so that with ALL the attributes, we can then conbime with the Sales price of train data, then interpolate for the remaining test through nn
train_drop = df.drop(columns=['Id', 'SalePrice'])
train_drop1 = df_test.drop(columns=['Id'])

# we will first anylize the categories with an actual integer or float value (quantitive): we use
only_numbers_train = train_drop.select_dtypes(exclude=['object'])

only_numbers_test = train_drop1.select_dtypes(exclude=['object'])

# # fill missing values with 0:
# only_numbers_train.fillna(0)

# we create a mean and std such that we train the set to recognize how many std away the test set is. we then set a binary to the valid and invalid values
# use the standard z-distribution
stat_model = only_numbers_train.apply(lambda x: (x - x.mean()) / (x.std()))  # we don't take the abs value so we can see what values are less or more than.

# Print the DataFrame
print(f"Numerical Columns: \n{only_numbers_train}\n----------------------------------------------------------------------------------------")
print(f"Z values away from expected (xavier distribution):\n{stat_model}\n----------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------
print(f"Columns with object names: \n\n{df.select_dtypes('object')}\n----------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------


# we upload the data here into a form the tensors can regonize in the training proccess
input = only_numbers_train.shape[0]
attibutes = only_numbers_train.shape[1]
print(f"Number Train_Data Samples: {input}")
print(f"Number Train_Attributes (Numerical charateristics): {attibutes}\n----------------------------------------------------------------------------------------")
print(f"Dataset affter removing object names: \n{only_numbers_train}\n----------------------------------------------------------------------------------------")

learning_rate = 1e-3
batch_size = 64

class CustomTensorDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features.float()
        self.targets = targets.float() #to insure the input to the funciton are integers

    def __len__(self):
        return len(self.features)

    def __getitem__(self, placement):  # by placement, we mean which column we want to extract features from
        sample = self.features[placement]
        target = self.targets[placement]
        return sample, target


# Convert DataFrame to tensors
train_features = torch.tensor(only_numbers_train.values).float()


trainset = torch.tensor(only_numbers_train.values).float()


test_features = torch.tensor(only_numbers_test.values).float()
target_sales = torch.tensor(sales_column.values).float()

train_features = train_features.to(device)
target_sales = target_sales.to(device)
test_features=test_features.to(device)


# Now create the dataset
train_dataset_custom_dataset = CustomTensorDataset(train_features, target_sales)
test_dataset_custom_dataset = CustomTensorDataset(test_features, target_sales)  # here, there is no sales column but we try to predict using it

train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
test_loaded_data = DataLoader(test_dataset_custom_dataset, batch_size=batch_size, shuffle=True)

print(f"Trained_data (after putting into custom dataset filter and loading data through DataLoader so python can understand): \n{train_loaded_data}\n----------------------------------------------------------------------------------------")
print(f"Tested_data (after putting into custom dataset filter and loading data through DataLoader so python can understand): \n{test_loaded_data}\n----------------------------------------------------------------------------------------")

class Train_Model(torch.nn.Module):
    def __init__(self):
        super(Train_Model, self).__init__()
        # Define layers correctly without trailing commas
        self.layer1 = nn.Linear(attibutes, 1024)
        self.layer2 = nn.Linear(1024, 1)
        #self.layer3 = nn.Linear(1024, 1)

        #we made an example of this above the class definitions
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        #nn.init.xavier_uniform_(self.layer3.weight)

        self.relu = nn.ReLU()
        #self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.layer1(x))  # Apply ReLU after linear transformation
        x = self.relu(self.layer2(x))
        #x = self.relu(self.layer3(x))
        return x.squeeze()
# ----------------------------------------------------------------------------------------

# tuple_sales = tuple(only_numbers_train[.astype(int))
model = Train_Model().to(device) # input was defined above as total number of columns (or number of characteristics)


print(f"Our neural network model:\n {model}\n----------------------------------------------------------------------------------------")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
loss_function=nn.MSELoss()

# def NONZERO_Loss(pred, target,loss_function): #we calculate the differnace between the value desired and actual value-- which gives our loss
#     optimized_pred = torch.log(pred + 1)
#     optimized_target = torch.log(target + 1)
#     loss = loss_function(optimized_pred, optimized_target)
#     return loss



def train_one_test(model, optimizer, data_loader, device):
    model.train()  # Ensure the model is in training mode
    for train_features, target_sales in data_loader:
        # Ensure features and targets are on the same device
        train_features = train_features.to(device)
        target_sales = target_sales.to(device)
        print(train_features)
        optimizer.zero_grad()  # Zero the gradients
        predictions = model(train_features)  # Forward pass
        loss = loss_function(predictions, target_sales)  # Calculate loss
        # print(f"00000000000000000000000000{predictions}")
        # print(f"00000000000000000000000000{target_sales}")

        if torch.isnan(loss):
            print("NaN loss detected")
            continue
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        return loss.item()  # Return the loss for this batch

def train_all(model, optimizer, device, number_tests):
    model.train()  # Make sure model is in training mode
    for i in range(number_tests):
        loss_after = train_one_test(model, optimizer, train_loaded_data, device)
        print(f"Test Number: {i + 1}, Loss: {loss_after}")

if __name__ == '__main__':
    print("This is our trained data after being filtered through the neural network:\n")

    for features, targets in train_loaded_data:
        outputs = model(features)
        print(outputs)
    print("----------------------------------------------------------------------------------------")

    print("Training process begins:\n")
    train_all(model, optimizer, device, 6)
    print("\nTraining completed.\n-------------------------------------------")

    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for features in test_loaded_data:
            features = features[0].to(device)  # Unpack the features and move them to the device
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())  # Collecting predictions
    print(predictions)
