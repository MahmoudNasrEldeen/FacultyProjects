# Imported Modules
from Preprocessing import FeatureScaling, PlottingData
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import time
import pickle

# Modules Usage
'''
LinearRegression used in LinearRegressionModel()
PolynomialFeatures, np used in PolynomialRegressionModel()
metrics, time used in Both
FeatureScaling used in RunAllModels()
'''

def LinearRegressionModel(x_train, x_test, y_train, y_test):
    lin_model = LinearRegression()
    start = time.time()
    lin_model.fit(x_train, y_train)
    print('Training Time : ', time.time() - start)

    # Save Linear Regression Model
    file_path = "Models/LinearRegressionModel.txt"
    with open(file_path, 'wb') as fh:
        pickle.dump(lin_model, fh)

    print("--- For Training ----")
    y_predict_train = lin_model.predict(x_train)
    MSE = metrics.mean_squared_error(y_train, y_predict_train)
    print('MSE : ', round(MSE, 2))
    print('Accuracy : ', 1 - round(MSE, 2))

    print("--- For Testing ----")
    y_predict_test = lin_model.predict(x_test)
    MSE = metrics.mean_squared_error(y_test, y_predict_test)
    print('MSE : ', round(MSE, 2))
    print('Accuracy : ', 1 - round(MSE, 2))

    return y_predict_test

def PolynomialRegressionModel(x_train, x_test, y_train, y_test):
    # Get Features of Polynomial
    polynomial_features = PolynomialFeatures(degree=4)
    x_train_poly = polynomial_features.fit_transform(x_train)
    x_test_poly = polynomial_features.fit_transform(x_test)

    model = LinearRegression()
    start = time.time()
    model.fit(x_train_poly, y_train)
    print('Training Time : ', time.time() - start)

    # Save Polynomial Regression Model
    file_path = "Models/PolynomialRegressionModel.txt"
    with open(file_path, 'wb') as fh:
        pickle.dump(model, fh)


    print("--- For Training ----")
    y_poly_predict_train = model.predict(x_train_poly)
    MSE_train = metrics.mean_squared_error(y_train, y_poly_predict_train)
    print('MSE : ', round(MSE_train, 2))
    print('Accuracy : ', 1 - round(MSE_train, 2))

    print("--- For Testing ----")
    y_poly_predict_test = model.predict(x_test_poly)
    MSE_test = metrics.mean_squared_error(y_test, y_poly_predict_test)
    print('MSE : ', round(MSE_test, 2))
    print('Accuracy : ', 1 - round(MSE_test, 2))

    return y_poly_predict_test

def RunAllModels(dataset, x_train, x_test, y_train, y_test, most_correlated_feature):
    want_again = True
    while want_again:
        print("Which Model you prefer to Run !!")
        print("1- Linear Regression Model.")
        print("2- Polynomial Regression Model.")
        user_choice = int(input("Your choice: "))
        if user_choice == 1:
            # Make Feature Scaling
            x_train, x_test = FeatureScaling(x_train, x_test)
            # Then Run Model
            print("Linear Regression Model Start Running..")
            y_pred = LinearRegressionModel(x_train, x_test, y_train, y_test)
            PlottingData(dataset, x_test, y_test, y_pred, most_correlated_feature)

        elif user_choice == 2:
            # Run Polynomial Model
            print("Polynomial Regression Model Start Running..")
            y_pred = PolynomialRegressionModel(x_train, x_test, y_train, y_test)
            PlottingData(dataset, x_test, y_test, y_pred, most_correlated_feature)
        else:
            print("La2 Hwa Aakherna Two Models bs.")

        another_model = input("Do You wanna try another Model (y/n): ")
        if another_model != 'y':
            want_again = False
