# Imported Modules
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from Preprocessing import RunTestCase
from train_models import LinearRegressionModel, PolynomialRegressionModel
import pickle

# Ignore Warning [It Doesn't effect on the efficiency of Code]
import warnings
warnings.filterwarnings("ignore")

# Get File From user
print("You are now testing Modules By loading Them from File ..")
file_path = 'Dataset/AppleStore_training.csv'
x_test, y_test = RunTestCase(file_path)

# Then Load The Models
again = True
while again:
    print("Which Model you want to Test")
    print("1- Linear Regression Model")
    print("2- Polynomial Regression Model")
    choice = int(input("your choice: "))

    if choice == 1:
        # ---------------- LR MODEL ------------------
        print("You are now testing LR Model")
        file_path = "Models/LinearRegressionModel.txt"
        scale = StandardScaler()
        x_test = scale.fit_transform(x_test)
        lr_model = pickle.load(open(file_path, 'rb'))
        y_predict_test = lr_model.predict(x_test)
        MSE = metrics.mean_squared_error(y_test, y_predict_test)
        print('Accuracy for LR Model: ', 1 - round(MSE, 2))

    elif choice == 2:
        # ---------------- PR MODEL ------------------
        print("You are now testing PR Model")
        file_path = "Models/PolynomialRegressionModel.txt"
        pr_model = pickle.load(open(file_path, 'rb'))
        y_predict_test = pr_model.predict(x_test)
        MSE = metrics.mean_squared_error(y_test, y_predict_test)
        print('Accuracy for PR Model: ', 1 - round(MSE, 2))

    print("Do you want try test another model?")
    print("1- Yes, I want try another model")
    print("2- No, Close Whole Testing")
    test_again = int(input("Your Choice: "))
    if test_again == 2:
        again = False
