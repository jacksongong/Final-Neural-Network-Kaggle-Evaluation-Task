import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader #here, the pytorch ALREADY HAS pre-defined functions that can be used through the Dataset and DataLoader functions.
from sklearn.model_selection import train_test_split

test_size=0.315
random_state=42



pd.set_option('display.max_rows', 20)  # None means show all rows

# Set the maximum number of columns to display
pd.set_option('display.max_columns', 20)  # None means show all columns

#I am using GPU device
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device\n----------------------------------------------------------------------------------------")
# -----------------------------------------------------------------------------------------

df_test = pd.read_csv("F:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/test.csv")
df= pd.read_csv("F:/Cai_Wang_ResearchAssignment/House_Project_Kaggle-PyTorch/train.csv") #df will be our train data



#we need this part to do the ID match in the 'isloated attirbute' category

train_original, validation_original_train = train_test_split(df.iloc[1:], test_size=test_size, random_state=random_state)  # Approximately 1000 samples for training
# #only do for training into validation since we need all the values before manipulation
train_set_original = pd.concat([df.iloc[0:1], train_original])


test_original, validation_original_test = train_test_split(df_test.iloc[1:], test_size=test_size, random_state=random_state)  # Approximately 1000 samples for training
# #only do for training into validation since we need all the values before manipulation
train_set_test = pd.concat([df_test.iloc[0:1], test_original])


#this is for my labtop
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
print(f"{label} (column 2),\nSale Price (column 1)\n{label_relation_to_sales_price}\n----------------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------

#we now look at a specific numerical case to get a better idea of what to do
plt.scatter(x='LotArea', y='SalePrice', data=df, label='House Prices vs Lot Area')    #this line is very interesting since when are able to
plt.title('Example Data Distibution')  # Title
plt.xlabel('Lot Area (x)')
plt.ylabel('Sale Price (y)')
plt.legend(title="Legend", loc='lower right', fontsize='x-small')
plt.show()


# ----------------------------------------------------------------------------------------
# we will first deal with everything with numerical values
before_sales_column = df["SalePrice"]








sales_column, validation_sales = train_test_split(before_sales_column, test_size=test_size, random_state=random_state)  # Approximately 1000 samples for training









# connect all the data so that with ALL the attributes, we can then conbime with the Sales price of train data, then interpolate for the remaining test through nn
train_drop = df.drop(columns=['Id', 'SalePrice']) #we do not want these to affec the training
test_drop = df_test.drop(columns=['Id'])

# we will first anylize the categories with an actual integer or float value (quantitive): we use
#here, we want to filter such that we only have a random number of selections that work for for the TRAINING set, then we will validate on the validate set
before_numbers_train = train_drop.select_dtypes(exclude=['object'])






only_numbers_train, validation_train_set = train_test_split(before_numbers_train, test_size=test_size, random_state=random_state)  # Approximately 1000 samples for training








print(f"iiiiiiiiiiiiiiiiiiiiiiiiiiiiiii{before_numbers_train}")
print(f"iiiiiiiiiiiiiiiiiiiiiiiiiiiiiii{only_numbers_train}")
print(f"iiiiiiiiiiiiiiiiiiiiiiiiiiiiiii{validation_train_set}")



only_numbers_test = test_drop.select_dtypes(exclude=['object'])


#only_numbers_test, validation_test_set = train_test_split(before_numbers_test, test_size=test_size, random_state=random_state)  # Approximately 1000 samples for training


only_numbers_test.fillna(0, inplace=True) #we can possibly use 0 for placeholders instead, since these values are 'none'
only_numbers_train.fillna(0, inplace=True)

# we create a mean and std such that we train the set to recognize how many std away the test set is. we then set a binary to the valid and invalid values
# use the standard z-distribution
stat_model = only_numbers_train.apply(lambda x: (x - x.mean()) / (x.std()))  # we don't take the abs value so we can see what values are less or more than.
stat_model_test = only_numbers_test.apply(lambda x: (x - x.mean()) / (x.std()))  # we don't take the abs value so we can see what values are less or more than.
stat_model_validation = validation_train_set.apply(lambda x: (x - x.mean()) / (x.std()))  # we don't take the abs value so we can see what values are less or more than.


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

    def __getitem__(self, placement):  # by placement, this is like
        sample = self.features[placement]
        target = self.targets[placement]
        return sample, target

    #this part is unnessseary
    def __listtargets__ (self):
        list=self.targets.tolist()
        return list


# Convert DataFrame to tensors
train_features = torch.tensor(stat_model.values).float()
test_features = torch.tensor(stat_model_test.values).float()
validation_features = torch.tensor(stat_model_validation.values).float()



sales_column_tensor = torch.tensor(sales_column.values).float()
validation_sales_tensor=torch.tensor(validation_sales.values).float()


#we normalize the sales column as well so loss can be comparable
mean=validation_sales_tensor.mean()
std=validation_sales_tensor.std()
validation_normalized_sales = (validation_sales_tensor-mean)/std   #this is already a tensor, since we used tensors for the mean and std


#we normalize the sales column as well so loss can be comparable
mean=sales_column_tensor.mean()
std=sales_column_tensor.std()
normalized_sales = (sales_column_tensor-mean)/std   #this is already a tensor, since we used tensors for the mean and std


normalized_sales = normalized_sales.to(device)
normalized_tensor_sales = torch.tensor(normalized_sales).float()
#
train_features = train_features.to(device)
validation_features=validation_features.to(device)
normalized_tensor_sales=normalized_tensor_sales.to(device)
test_features=test_features.to(device)


# Now create the dataset #
print((train_features).shape)
print((normalized_tensor_sales).shape)

train_dataset_custom_dataset = CustomTensorDataset(train_features, normalized_tensor_sales)

print((test_features).shape)
print((normalized_sales).shape)

test_dataset_custom_dataset = CustomTensorDataset(test_features, normalized_sales)  # here, there is no sales column but we try to predict using it

print((validation_features).shape)
print((validation_normalized_sales).shape)

validation_dataset_custom_dataset = CustomTensorDataset(validation_features, validation_normalized_sales)  # here, there is no sales column but we try to predict using it



train_loaded_data = DataLoader(train_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
validation_loaded_data = DataLoader(validation_dataset_custom_dataset, batch_size=batch_size, shuffle=True)
test_loaded_data = DataLoader(test_dataset_custom_dataset, batch_size=batch_size, shuffle=True)

# print(f"Trained_data (after putting into custom dataset filter and loading data through DataLoader so python can understand): \n{train_loaded_data}\n----------------------------------------------------------------------------------------")
# print(f"Tested_data (after putting into custom dataset filter and loading data through DataLoader so python can understand): \n{test_loaded_data}\n----------------------------------------------------------------------------------------")

class Train_Model(torch.nn.Module):
    def __init__(self):
        super(Train_Model, self).__init__()
        # Define layers correctly without trailing commas
        self.layer1 = nn.Linear(attibutes, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1)


#the issue with regular relu is that it does not recognize the negative values of the normalized set

        self.model_version = nn.PReLU()  #we want loss that is as close to 1 as possible. If greater, we have a over estimate, whereas going under will have loss value less than 1

        #we made an example of this above the class definitions    #we may want to experiment with the possible weighting for the different layers.
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)

        #notes on model types:
        # sigmoid give svalues that are very close to eachother, with the deviation very minimal
        #softplus gives values that are in bunches that have a similar value but vary upon where these values are located
        #Relu is very unstable as it kills some nodes but gives GOOD VARIATION----more identical ones, but very nice variation (depends on specific runs)
        #!!!!we cannot combine leadky relu with fillna(0) since then some of the 0 values are never able to be changed from the model)

        #leakyrelu feels more stable, as the outputs are more consistantly varying and having values that are not all 0s
        #gelu gives values that are consitantly a bit small (I am assuming get values based on a lower restraint)

        #self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.model_version(self.layer1(x))
        x = self.model_version(self.layer2(x))
        x = self.model_version(self.layer3(x))
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


all_loss=[]

def train_one_test(model, optimizer, data_loader, device):
    loss_list = []

    model.train()  # Ensure the model is in training mode
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
        loss_list.append(loss) #no need abs value, since the MSE will square the result.


#here, the value of the loss function is wrong
    #print(loss_list)
    average_loss=sum(loss_list)/len(loss_list)
    all_loss.append(loss_list)
    return(average_loss)  # Return loss for the final updated value of loss after all the batches have run through and loss have been adjusted accordingly
#.item() ensures that we do not get a tensor as a result but istead a value that is usable

def train_all(model, optimizer, loaded_data, device, epochs):
    model.train()  # Make sure model is in training mode
    for i in range(epochs):
        loss_after = train_one_test(model, optimizer, loaded_data, device)
        print(f"Test Number: {i + 1}, Loss: {loss_after}")



if __name__ == '__main__':
    print("This is our trained data after being filtered through the neural network:\n")

    isolated_attribut='LotArea'
    trained_outputs=[]
    for features_, targets in train_loaded_data:
        outputs = model(features_)
        trained_outputs.append(outputs)
        #print(outputs)   #this function here doesn't do anything. it simply iterates over the customdataset class, which simply will iterate over the placement values over the entire set
    #print(f"{trained_outputs}\n----------------------------------------------------------------------------------------")

    print("Training process begins:\n")

# this is for the training of the original train data

    train_all(model, optimizer, train_loaded_data,device, 6)
    print("\nTraining completed.\n-------------------------------------------")



    model.eval()  # Set the model to evaluation mode
    predictions_ = []
    deviation=[]
    with torch.no_grad():
        for features1 in test_loaded_data:
            features1 = features1[0].to(device)  # Unpack the features and move them to the device
            outputs = model(features1)
            #print(outputs)
            predictions_.extend(((outputs*std)+mean).cpu().numpy())  # Collecting predictions
            deviation.extend(outputs.cpu().numpy())  # this part collects the deviation as the z-values in actual-mean/std since it is the result from squeezed value after putting into the Train_Model class

    # we must normalize the outputs of the data by reversing the steps in z-normalization in order to get the actual prices, not just the deviation from the prices
    x_length_predictions=[]
    for i in range(0,len(predictions_)):
        x_length_predictions.append(i+1)











# this is for the training of the validation train data
    train_all(model, optimizer, validation_loaded_data, device, 6)
    print("\nTraining of Validation set completed.\n-------------------------------------------")

    model.eval()  # Set the model to evaluation mode
    predictions_validation = []
    deviation_validate = []
    with torch.no_grad():
        for features2 in validation_loaded_data:
            features2 = features2[0].to(device)  # Unpack the features and move them to the device
            outputs = model(features2)
            # print(outputs)
            predictions_validation.extend(((outputs * std) + mean).cpu().numpy())  # Collecting predictions
            deviation_validate.extend(outputs.cpu().numpy())  # this part collects the deviation as the z-values in actual-mean/std since it is the result from squeezed value after putting into the Train_Model class

    # we must normalize the outputs of the data by reversing the steps in z-normalization in order to get the actual prices, not just the deviation from the prices
    x_length_validate_predictions = []
    for i in range(0, len(predictions_validation)):
        x_length_validate_predictions.append(i + 1)

    loss_validation=[]
    for i in range (0, len(predictions_validation)):
        loss_validation_one_case=predictions_validation[i]-validation_sales[i]
        loss_validation.append(loss_validation_one_case)


    print(f"The differance in price for validation set is: \n{loss_validation}")
    plt.plot(x_length_validate_predictions,loss_validation,lw=0.5, label='Validation Prediction Differance From Actual Sales')
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()











    print(f"House Predictions of Test_Data: \n{predictions_}\n")
    plt.figure(figsize=(10, 5))
    plt.plot(x_length_predictions, predictions_,lw=0.5, label='Predictions Sales Values')
    print(f"Z-Deviation derived from predictive model: \n{deviation}\n")
    print(f"Train Data House Prices: \n{sales_column_tensor.tolist()}")

    x_length_train_target = []
    for i in range(0, len(sales_column_tensor.tolist())):
        x_length_train_target.append(i + 1)
    plt.plot(x_length_train_target, sales_column_tensor.tolist(),lw=0.5, label='Training Sales Values')
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')

#we can go one step furhter and anylize the impact of one feature on the sales prive by having the x value as say 'lot area' and the price afterwards and see if the relationship is somewhat linear.
#realize that no one feature will be LINEAR---it is quite difficult to find a direct relationship between x and y values
    plt.show()


    plt.plot(isolated_attribut, 'SalePrice', data=train_set_original, label=f'Orginial {isolated_attribut}',lw=0.2)  # this line is very interesting since when are able to
    #print(df_test['LotArea'].dtype())
    #(df_test['LotArea'].max())
    plt.plot(isolated_attribut,tuple(predictions_), data=train_set_test, label=f'Predictive {isolated_attribut}',lw=0.2)
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')

    plt.show()


    #print(type(x_length_train_target))

    # x_length_train_target_cpu = x_length_train_target.numpy()
    # loss_list_array = np.array(loss_list)

    # print(x_length_train_target)
    # print(all_loss[1])


    for i in range (0, len(all_loss)):
        scalar_list = [individual_element.item() for individual_element in all_loss[i]]
        length_scalar = []

        for t in range(0, len(scalar_list)):
            length_scalar.append(t + 1)
        plt.plot(length_scalar,scalar_list, label=f'Loss of Each Batch Training Data, Epoch Number {i+1}, Batches Per Trial: {len(length_scalar)}')
    plt.legend(title="Legend", loc='upper right', fontsize='x-small')
    plt.show()



    #now, we plot the verification of the train data by putting through prediction, giving prediction, then matching to the actual value that is ALSO GIVEN by the train data






