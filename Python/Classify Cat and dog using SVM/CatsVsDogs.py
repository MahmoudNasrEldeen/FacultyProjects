from HelperFunctions import TrainingProcess, TestingProcess
import pandas as pd
# Step 1 : Importing Images Features and save it into file
path = 'data/train'
''' Note
First I used the next function to Get features of each image then
save it into File to use it again easly .. you 'll find this file
in attached to this .py file so its not neccessary to uncomment
the following function its already executed and saved data to file 
But if you want Run it import it From Assignment_3_HelperFunctions
File then uncomment the next Function
'''
#ReadImages(path)

# Step 2: From Generated File Get Features
print("Extract the data from file .. Please wait !!")
dataset = pd.read_csv('catsVsDogs.csv')
x = dataset.iloc[:, 0:3780]
y = dataset.iloc[:, 3780]

# Step 3: Splitting Data into Trained and Test
''' Note
I feel that iam confused on how to read iamages and classify it in test file
so i split th e trained dataset and work with some of them as test
'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, shuffle = True)

# Step 4: Training SVM Model
print("Training SVM Model data .. Please wait !!")
svc = TrainingProcess(X_train, Y_train)

# Step 5: Testing SVM Model
print("Testing SVM Model the data .. Please wait !!")
TestingProcess(svc, X_train, X_test, Y_train, Y_test)
