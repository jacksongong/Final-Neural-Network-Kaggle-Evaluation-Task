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
pd.set_option('display.max_rows', 2000)  # None means show all rows

# Set the maximum number of columns to display
pd.set_option('display.max_columns', 2000)  # None means show all columns
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

# df = pd.read_csv('/kaggle/input/test-train-data/train.csv')
# df_test = pd.read_csv('/kaggle/input/test-train-data/test.csv')



#df_test = pd.read_csv("./test.csv")
#df= pd.read_csv("./train.csv") #df will be our train data

df_test = pd.read_csv("F:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/test.csv")
df= pd.read_csv("F:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/train.csv") #df will be our train data

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
only_numbers_test.fillna(0, inplace=True) #we can possibly use 0 for placeholders instead

# # fill missing values with 0: these mean that for that attrubiute, there is NONE of the desired value (as shown by the dataset)
only_numbers_train.fillna(0, inplace=True)


# we create a mean and std such that we train the set to recognize how many std away the test set is. we then set a binary to the valid and invalid values
# use the standard z-distribution
stat_model = only_numbers_train.apply(lambda x: (x - x.mean()) / (x.std()))  # we don't take the abs value so we can see what values are less or more than.

print(stat_model)

stat_model_test = only_numbers_test.apply(lambda x: (x - x.mean()) / (x.std()))  # we don't take the abs value so we can see what values are less or more than.


# Print the DataFrame
print(f"Numerical Columns: \n{only_numbers_train}\n----------------------------------------------------------------------------------------")
print(f"Z values away from expected (xavier distribution):\n{stat_model}\n----------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------
print(f"Columns with object names: \n\n{df.select_dtypes('object')}\n----------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------


# we upload the data here into a form the tensors can regonize in the training proccess
input = stat_model.shape[0]
attibutes = stat_model.shape[1]
print(f"Number Train_Data Samples: {input}")
print(f"Number Train_Attributes (Numerical charateristics): {attibutes}\n----------------------------------------------------------------------------------------")
print(f"Normalized Dataset after removing object names: \n{stat_model}\n----------------------------------------------------------------------------------------")

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
train_features = torch.tensor(stat_model.values).float()


#trainset = torch.tensor(only_numbers_train.values).float()


test_features = torch.tensor(stat_model_test.values).float()
sales_column_tensor = torch.tensor(sales_column.values).float()


#we normalize the sales column as well so loss can be comparable
mean=sales_column_tensor.mean()
std=sales_column_tensor.std()
normalized_sales = (sales_column_tensor-mean)/std



print(f"{normalized_sales}ffffffffffffffffffffffffffffffff")

normalized_target_sales = torch.tensor(normalized_sales).float()



train_features = train_features.to(device)
normalized_target_sales=normalized_target_sales.to(device)
normalized_sales = normalized_sales.to(device)
test_features=test_features.to(device)


# Now create the dataset
train_dataset_custom_dataset = CustomTensorDataset(train_features, normalized_target_sales)
test_dataset_custom_dataset = CustomTensorDataset(test_features, normalized_sales)  # here, there is no sales column but we try to predict using it

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

        #we made an example of this above the class definitions    #we may want to experiment with the possible weighting for the different layers.
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        #nn.init.xavier_uniform_(self.layer3.weight)

        self.model_version = nn.ReLU()  #we want loss that is as close to 1 as possible. If greater, we have a over estimate, whereas going under will have loss value less than 1

        #notes on model types:
        # sigmoid give svalues that are very close to eachother, with the deviation very minimal
        #softplus gives values that are in bunches that have a similar value but vary upon where these values are located
        #Relu is very unstable as it kills some nodes but gives GOOD VARIATION----more identical ones, but very nice variation (depends on specific runs)
        #!!!!we cannot combine leadky relu with fillna(0) since then some of the 0 values are never able to be changed from the model)

        #leakyrelu feels more stable, as the outputs are more consistantly varying and having values that are not all 0s
        #gelu gives values that are consitantly a bit small (I am assuming get values based on a lower restraint)


        #self.flatten = nn.Flatten()

    def forward(self, x):
        #print(f"uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu{x}")

        x = self.model_version(self.layer1(x))

        x = self.model_version(self.layer2(x))

        #x = self.relu(self.layer3(x))
        x = x.clone().detach().requires_grad_(True)
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
    loss_list=[]
    for train_features, normalized_target_sales in data_loader: #each iteration of the loop is only one batch in the cycle that we must implement
        # Ensure features and targets are on the same device
        train_features = train_features.to(device)
        normalized_target_sales = normalized_target_sales.to(device)

        #print(f"fffffffffffffffffffffffffffffffffffffffff{train_features}")


        optimizer.zero_grad()  # Zero the gradients---->this could possible be a mistake here
        predictions = model(train_features)  # Forward pass

        # print(f"{predictions}ffffffffffffffffffff")
        # print(f"{normalized_target_sales}ffffffffffffffffffff") # here, we have to normalize the target_sales as well

        loss = loss_function(predictions, normalized_target_sales)  # Calculate loss-----------> there must be something wrong with the loss function here
        #print(f"00000000000000000000000000{normalized_target_sales}")

        # if torch.isnan(loss):  #when this line is here, it seems to not proccedd to the rest of the code
        #     print("NaN loss detected")
        #     #loss_list.append(loss) #we CANNOT use the 'continue' function because then we will have to exit the for loop for this iteration and continue onto the next iteration.



        loss.backward()  # Backpropagation # this will change the gradients(nodes) of the network so that each time we can get a different output
        optimizer.step()  # Update model parameters

        #print(f"{loss}")  #we can use this part to see the loss for each of the individual samples (we see some are clear outliers, while others are not)
        loss_list.append(loss)


#here, the value of the loss function is wrong
    average_loss=sum(loss_list)/len(loss_list)
    return(average_loss)  # Return loss for the final updated value of loss after all the batches have run through and loss have been adjusted accordingly
#.item() ensures that we do not get a tensor as a result but istead a value that is usable

def train_all(model, optimizer, device, number_tests):
    model.train()  # Make sure model is in training mode
    for i in range(number_tests):
        loss_after = train_one_test(model, optimizer, train_loaded_data, device)
        print(f"Test Number: {i + 1}, Loss: {loss_after}")



if __name__ == '__main__':
    print("This is our trained data after being filtered through the neural network:\n")

    for features, targets in train_loaded_data:
        outputs = model(features)
        #print(outputs)
    print("----------------------------------------------------------------------------------------")

    print("Training process begins:\n")
    train_all(model, optimizer, device, 6)
    print("\nTraining completed.\n-------------------------------------------")

    model.eval()  # Set the model to evaluation mode
    predictions_ = []
    deviation=[]
    with torch.no_grad():
        for features in test_loaded_data:
            features = features[0].to(device)  # Unpack the features and move them to the device
            outputs = model(features)
            #print(outputs)
            predictions_.extend(((outputs*std)+mean).cpu().numpy())  # Collecting predictions
            deviation.extend(outputs.cpu().numpy())  # Collecting predictions

    # we must normalize the outputs of the data by reversing the steps in z-normalization in order to get the actual prices, not just the deviation from the prices

    print(f"House Predictions of Test_Data: \n{predictions_}\n")
    print(f"Z-Deviation derived from predictive model: \n{deviation}\n")
    print(f"Train Data House Prices: \n{sales_column_tensor.tolist()}")

