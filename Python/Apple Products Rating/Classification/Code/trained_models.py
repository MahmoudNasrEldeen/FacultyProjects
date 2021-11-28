# Imported Modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from preprocessing import PlottingData
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle


class SVM:
    def __init__(self, database, x_train, x_test, y_train, y_test, feature, c=8, gamma=0.9):
        self.C = c
        self.Gamma = gamma

        self.database = database
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.most_correlated_feature = feature

        self.accuracy_test = None
        self.accuracy_train = None
        self.training_time = None
        self.time_test = None
        self.time_train = None

    def TrainModel(self, x_train, y_train):
        start = time.time()
        svm_model = SVC(kernel='rbf', gamma=self.Gamma, C=self.C).fit(x_train, y_train)
        end = time.time()

        # Save Model in File
        file_path = "Models/svm_model.txt"
        with open(file_path, 'wb') as fh:
            pickle.dump(svm_model, fh)

        self.training_time = end - start
        print("SVM Training Time: ", end - start)
        return svm_model

    def TestModel(self, svm_model, x, y, test_for):
        print("------- For", test_for, "Data --------")
        start = time.time()
        svm_pred = svm_model.predict(x)
        end = time.time()
        print("- Test Time for", test_for, "Data : ", end - start)
        print('- Mean Square Error: ', metrics.mean_squared_error(y, svm_pred))
        print('- Accuracy: ', svm_model.score(x, y))

        if test_for == "Trained":
            self.accuracy_train = svm_model.score(x, y)
            self.time_train = end - start
        else:
            self.accuracy_test = svm_model.score(x, y)
            self.time_test = end - start

        return svm_pred

    def RunOnlyTestandPlotting(self, svm_model):
        svm_pred_train = self.TestModel(svm_model, self.x_train, self.y_train, "Trained")
        PlottingData(self.database, self.x_train, self.y_train, svm_pred_train, self.most_correlated_feature, "Trained")

        svm_pred_test = self.TestModel(svm_model, self.x_test, self.y_test, "Tested")
        PlottingData(self.database, self.x_test, self.y_test, svm_pred_test, self.most_correlated_feature, "Tested")

    def RunAllSVMModel(self):
        svm_model = self.TrainModel(self.x_train, self.y_train)
        self.RunOnlyTestandPlotting(svm_model)

    def PlottingBarsFor_Accuarcy(self):
        measurements = ['Training Accuracy', 'Testing Accuracy']
        performance = [self.accuracy_train, self.accuracy_test]

        y_pos = np.arange(len(measurements))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, measurements)
        plt.ylabel('Value in %')
        plt.title('Accuracy Measurement Of Model')
        plt.show()

    def PlottingBarsFor_Train_Test_Time(self):
        measurements = ['Training set Time', 'Testing set Time', 'Model Training Time']
        performance = [self.training_time, self.time_test, self.time_train]

        y_pos = np.arange(len(measurements))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, measurements)
        plt.ylabel('Value in Seconds')
        plt.title('Time Measurement Of Model')
        plt.show()


class RFC:

    def __init__(self, database, x_train, x_test, y_train, y_test, feature, n_estimators=50, max_depth=20):
        self.n_estimator = n_estimators
        self.max_depth = max_depth

        self.database = database
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.most_correlated_feature = feature

        self.accuracy_test = None
        self.accuracy_train = None
        self.training_time = None
        self.time_test = None
        self.time_train = None

    def TrainModel(self, x_train, y_train):
        start = time.time()
        rfc_model = RandomForestClassifier(n_estimators=self.n_estimator, max_depth=self.max_depth)
        rfc_model.fit(x_train, y_train)
        end = time.time()

        # Save Model in File
        file_path = "Models/rfc_model.txt"
        with open(file_path, 'wb') as fh:
            pickle.dump(rfc_model, fh)

        self.training_time = end - start
        print("RFC model Training Time: ", end - start)
        return rfc_model

    def TestModel(self, rfc_model, x, y, test_for):
        print("------- For", test_for, "Data --------")
        start = time.time()
        rfc_pred = rfc_model.predict(x)
        end = time.time()
        print("- Test Time for", test_for, "Data : ", end - start)
        print('- Mean Square Error: ', metrics.mean_squared_error(y, rfc_pred))
        print('- Accuracy: ', rfc_model.score(x, y))

        if test_for == "Trained":
            self.accuracy_train = rfc_model.score(x, y)
            self.time_train = end - start
        else:
            self.accuracy_test = rfc_model.score(x, y)
            self.time_test = end - start

        return rfc_pred

    def RunOnlyTestandPlotting(self, rfc_model):
        rfc_pred_train = self.TestModel(rfc_model, self.x_train, self.y_train, "Trained")
        PlottingData(self.database, self.x_train, self.y_train, rfc_pred_train, self.most_correlated_feature, "Trained")

        rfc_pred_test = self.TestModel(rfc_model, self.x_test, self.y_test, "Tested")
        PlottingData(self.database, self.x_test, self.y_test, rfc_pred_test, self.most_correlated_feature, "Tested")

    def RunAllRCFModel(self):
        rfc_model = self.TrainModel(self.x_train, self.y_train)
        self.RunOnlyTestandPlotting(rfc_model)

    def PlottingBarsFor_Accuarcy(self):
        measurements = ['Training Accuracy', 'Testing Accuracy']
        performance = [self.accuracy_train, self.accuracy_test]

        y_pos = np.arange(len(measurements))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, measurements)
        plt.ylabel('Value in %')
        plt.title('Accuracy Measurement Of Model')
        plt.show()

    def PlottingBarsFor_Train_Test_Time(self):
        measurements = ['Training set Time', 'Testing set Time', 'Model Training Time']
        performance = [self.training_time, self.time_test, self.time_train]

        y_pos = np.arange(len(measurements))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, measurements)
        plt.ylabel('Value in Seconds')
        plt.title('Time Measurement Of Model')
        plt.show()


class KNN:
    def __init__(self, database, x_train, x_test, y_train, y_test, feature, k=3, algorithm='auto'):
        self.K = k
        self.Algorithm = algorithm

        self.database = database
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.most_correlated_feature = feature

        self.accuracy_test = None
        self.accuracy_train = None
        self.training_time = None
        self.time_test = None
        self.time_train = None

    def TrainModel(self, x_train, y_train):
        start = time.time()
        knn_model = KNeighborsClassifier(n_neighbors=self.K, algorithm=self.Algorithm)
        knn_model.fit(x_train, y_train)
        end = time.time()

        # Save Model in File
        file_path = "Models/knn_model.txt"
        with open(file_path, 'wb') as fh:
            pickle.dump(knn_model, fh)

        self.training_time = end - start
        print("KNN Training Time: ", end - start)
        return knn_model

    def TestModel(self, knn_model, x, y, test_for):
        print("------- For", test_for, "Data --------")
        start = time.time()
        knn_pred = knn_model.predict(x)
        end = time.time()
        print("- Test Time for", test_for, "Data : ", end - start)
        print('- Mean Square Error: ', metrics.mean_squared_error(y, knn_pred))
        print('- Accuracy: ', knn_model.score(x, y))

        if test_for == "Trained":
            self.accuracy_train = knn_model.score(x, y)
            self.time_train = end - start
        else:
            self.accuracy_test = knn_model.score(x, y)
            self.time_test = end - start

        return knn_pred

    def RunOnlyTestandPlotting(self, knn_model):
        knn_pred_train = self.TestModel(knn_model, self.x_train, self.y_train, "Trained")
        PlottingData(self.database, self.x_train, self.y_train, knn_pred_train, self.most_correlated_feature, "Trained")

        knn_pred_test = self.TestModel(knn_model, self.x_test, self.y_test, "Tested")
        PlottingData(self.database, self.x_test, self.y_test, knn_pred_test, self.most_correlated_feature, "Tested")

    def RunAllKNNModel(self):
        knn_model = self.TrainModel(self.x_train, self.y_train)
        self.RunOnlyTestandPlotting(knn_model)

    def PlottingBarsFor_Accuarcy(self):
        measurements = ['Training Accuracy', 'Testing Accuracy']
        performance = [self.accuracy_train, self.accuracy_test]

        y_pos = np.arange(len(measurements))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, measurements)
        plt.ylabel('Value in %')
        plt.title('Accuracy Measurement Of Model')
        plt.show()

    def PlottingBarsFor_Train_Test_Time(self):
        measurements = ['Training set Time', 'Testing set Time', 'Model Training Time']
        performance = [self.training_time, self.time_test, self.time_train]

        y_pos = np.arange(len(measurements))
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, measurements)
        plt.ylabel('Value in Seconds')
        plt.title('Time Measurement Of Model')
        plt.show()


def RunWholeModels(dataset, x_train, x_test, y_train, y_test, most_correlated_feature):
    Whole_model_Again = True
    while Whole_model_Again:
        print("Hey .. What Model You want To Work With?")
        print("1- SVM Model.")
        print("2- RFC Model.")
        print("3- KNN Model.")
        print("4- No I wanna Exit Whole Program")
        model = int(input("Choice: "))
        if model == 1:
            # ---------- SVM MODEL -------------
            print("Ok .. You now working with SVM Model")
            print("For C and Gamma Values?")
            print("1- Run With Default C and Gamma[Student Work]")
            print("2- test Another C and Gamma")
            param_choice = int(input("Choice: "))
            if param_choice == 1:
                obj = SVM(dataset, x_train, x_test, y_train, y_test, most_correlated_feature)
                obj.RunAllSVMModel()
                obj.PlottingBarsFor_Accuarcy()
                obj.PlottingBarsFor_Train_Test_Time()
            else:
                Again = True
                while Again:
                    new_c = int(input("Enter C Value: "))
                    new_gamma = float(input("Enter Gamma Value: "))
                    obj = SVM(dataset, x_train, x_test, y_train, y_test, most_correlated_feature,
                              c=new_c, gamma=new_gamma)
                    obj.RunAllSVMModel()

                    print("Do you wanna another Test for C and Gamma")
                    print("1- Yes I want")
                    print("2- No I want To Try another Model")
                    another_test = int(input("Choice: "))
                    if another_test == 1:
                        continue
                    elif another_test == 2:
                        Again = False

        elif model == 2:
            # ---------- RCF MODEL -------------
            print("Ok .. You now working with RCF Model")
            print("For n_estimator and max_depth Values?")
            print("1- Run With Default estimator and max depth[Student Work]")
            print("2- test Another estimator and max depth")
            param_choice = int(input("Choice: "))
            if param_choice == 1:
                obj = RFC(dataset, x_train, x_test, y_train, y_test, most_correlated_feature)
                obj.RunAllRCFModel()
                obj.PlottingBarsFor_Accuarcy()
                obj.PlottingBarsFor_Train_Test_Time()
            else:
                Again = True
                while Again:
                    new_estimator = int(input("Enter Estimator Value: "))
                    new_depth = input("Enter max Depth Value: ")
                    obj = RFC(dataset, x_train, x_test, y_train, y_test, most_correlated_feature,
                              n_estimators=new_estimator, max_depth=int(new_depth))
                    obj.RunAllRCFModel()

                    print("Do you wanna another Test for Estimator and Max Depth")
                    print("1- Yes I want")
                    print("2- No I want To Try another Model")
                    another_test = int(input("Choice: "))
                    if another_test == 1:
                        continue
                    elif another_test == 2:
                        Again = False
        elif model == 3:
            # ---------- KNN MODEL -------------
            print("Ok .. You now working with KNN Model")
            print("For n Neighbors and Algorithm Values?")
            print("1- Run With Default N Neighbors and Algorithm[Student Work]")
            print("2- test Another N Neighbors and Algorithm")
            param_choice = int(input("Choice: "))
            if param_choice == 1:
                obj = KNN(dataset, x_train, x_test, y_train, y_test, most_correlated_feature)
                obj.RunAllKNNModel()
                obj.PlottingBarsFor_Accuarcy()
                obj.PlottingBarsFor_Train_Test_Time()
            else:
                Again = True
                while Again:
                    new_neighbor = int(input("Enter N Neighbors Value: "))
                    new_algorithm = input("Enter Algorithm Value: ")
                    obj = KNN(dataset, x_train, x_test, y_train, y_test, most_correlated_feature,
                              k=new_neighbor, algorithm=new_algorithm)
                    obj.RunAllKNNModel()

                    print("Do you wanna another Test for N Neighbors and Algorithm")
                    print("1- Yes I want")
                    print("2- No I want To Try another Model")
                    another_test = int(input("Choice: "))
                    if another_test == 1:
                        continue
                    if another_test == 2:
                        Again = False
        else:
            Whole_model_Again = False
