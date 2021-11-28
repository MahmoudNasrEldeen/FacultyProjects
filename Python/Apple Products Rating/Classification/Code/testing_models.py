# Imported Modules
from preprocessing import RunTestFile
import pickle

# Ignore Warning [It Doesn't effect on the efficiency of Code]
import warnings
warnings.filterwarnings("ignore")

# Get File From user
print("You are now testing Modules By loading Them from File ..")
file_path = 'Dataset/AppleStore_training_classification.csv'
x_test, y_test = RunTestFile(file_path)

# Then Load The Models
again = True
while again:
    print("Which Model you want to Test")
    print("1- SVM Model")
    print("2- RFC Model")
    print("3- KNN Model")
    choice = int(input("your choice: "))

    if choice == 1:
        # ---------------- SVM MODEL ------------------
        print("You are now testing SVM Model")
        file_path = "Models/SVM_Model.txt"
        svm_model = pickle.load(open(file_path, 'rb'))
        print("Prediction for SVM Model: {}".format(svm_model.score(x_test, y_test)))

    elif choice == 2:
        # ---------------- RFC MODEL ------------------
        print("You are now testing RCF Model")
        file_path = "Models/rfc_Model.txt"
        rcf_model = pickle.load(open(file_path, 'rb'))
        print("Prediction for RFC Model: {}".format(rcf_model.score(x_test, y_test)))

    elif choice == 3:
        # ---------------- KNN MODEL ------------------
        print("You are now testing KNN Model")
        file_path = "Models/knn_Model.txt"
        knn_model = pickle.load(open(file_path, 'rb'))
        print("Prediction for KNN Model: {}".format(knn_model.score(x_test, y_test)))

    print("Do you want try test another model?")
    print("1- Yes, I want try another model")
    print("2- No, Close Whole Testing")
    test_again = int(input("Your Choice: "))
    if test_again == 2:
        again = False
