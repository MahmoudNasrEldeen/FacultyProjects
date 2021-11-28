# Imported Modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

# Modules usage
'''
np used in TopkPrecentage()
sns, plt used in DrawHeatmap()
LabelEncoder used in CategoricalToNumerical()
ColumnTransformer, pd, OneHotEncoder used in MakeDummyVariable()
StandardScaler used in FeatureScaling()
'''

def DrawHeatmap(dataset):
    corr = dataset.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

def FixNanValues(dataset):
    # Get output variable
    output_variable = dataset.columns[len(dataset.columns) - 1]

    # Then Get Categorical and Numerical Features from dataset
    categorical_features = list(GetCategoricalFeatures(dataset))
    numerical_features = list(dataset.drop(categorical_features, axis=1).columns)
    categorical_features.remove(output_variable)

    # Remove Records that have 8 Empty Values
    dataset = dataset.dropna(axis=0, thresh=8)

    # Replace null values with mod value(the most repeated value in column) if it is categorical feature
    dataset = ReplaceCategoricalNan(dataset, categorical_features)

    # Replace null values with mean if it is numerical feature
    dataset = ReplaceNumericalNan(dataset, numerical_features)

    # Finally Return Dataset and output_variable
    return dataset, output_variable

def TopKPercentage(dataset, output_var, percentage, isTrain=True):
    # Defined Variables
    TopKValues=[]
    TopKIndex=[]

    # Make Correlation according To output variable
    corr = dataset.corr()
    abs_corr = abs(corr[output_var])
    abs_corr = abs_corr.drop(labels=output_var)
    sorted_corr = abs_corr.sort_values(ascending=False)
    
    # According to given number of features add to list to plot them
    for i in range(sorted_corr.size):
        if sorted_corr.values[i] >= percentage:
            TopKValues.append(sorted_corr.values[i])
            TopKIndex.append(sorted_corr.index[i])
        else:
            break

    # Finally plotting it
    if isTrain is True:
        y_pos = np.arange(len(TopKIndex))
        plt.barh(y_pos, TopKValues, align='center', alpha=0.5)
        plt.yticks(y_pos, TopKIndex)
        title_to_show = "Top Features for " + str(percentage) + " Percentage"
        plt.title(title_to_show)
        plt.show()

    # Then return The top features WITH Wanted output variable
    TopKIndex.append(output_var)
    return TopKIndex, sorted_corr

def ReplaceCategoricalNan(dataset, features):
    for feature in features:
        dataset.loc[:, feature] = dataset[feature].fillna(dataset[feature].value_counts().idxmax())
    return dataset

def ReplaceNumericalNan(dataset, features):
    for feature in features:
        dataset.loc[:, feature] = dataset[feature].fillna(round(dataset[feature].mean(), 2))
    return dataset

def GetCategoricalFeatures(dataset):
    return dataset.select_dtypes(exclude=['float64']).columns

def CategoricalToNumerical(dataset):
    # Call the Function that get categorical features
    categorical_features = GetCategoricalFeatures(dataset)

    # Encode these Categorical Features using Label Encoder
    label_encoder = LabelEncoder()
    for col in categorical_features:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    # Here check for Fixed Feature
    for feature in dataset.columns:
        if dataset[feature].nunique() == 1:
            dataset = dataset.drop(axis=1, labels=feature)

    # Return Dataset
    return dataset

def MakeDummyVariable(dataset, features):
    # Here Means i don't want to make dummy variable
    if features == -1:
        return dataset

    # First make Dummy Variable
    dummy_to_drop = list()
    for feature in features:
        ct = ColumnTransformer([(feature, OneHotEncoder(), [dataset.columns.get_loc(feature)])],
                               remainder='passthrough')
        dataset = ct.fit_transform(dataset)
        dataset = pd.DataFrame(dataset, columns=ct.get_feature_names())
        dummy_to_drop.append(ct.get_feature_names()[0])

    # Then Remove First Columns of Each one
    dataset = dataset.drop(dummy_to_drop, axis=1)
    return dataset

def FeatureSelection(dataset, features):
    main_dataset = dataset
    features_to_remove = dataset.drop(features, axis=1)
    return main_dataset.drop(features_to_remove, axis=1)

def FeatureScaling(x_train, x_test):
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.fit_transform(x_test)
    return x_train, x_test

def DummyVariableAcquiring(filename):
    # Get Categorical Feature
    dummies_file = "Dummies/dummies_selected.txt"
    dummy_variables_count = list()
    dataset = pd.read_csv(filename)
    categorical_features = list(GetCategoricalFeatures(dataset))
    categorical_features.remove(categorical_features[len(categorical_features) - 1])

    # Get Values of each Feature and save it into dictionary
    for feature in categorical_features:
        dummy_variables_count.append(len(dataset[feature].value_counts()))
    dummy_dict = {categorical_features[i]: dummy_variables_count[i] for i in range(len(categorical_features))}

    # Just for visualization let user choose the dummy variables
    NodummyAvailable = True
    for feature, count in dummy_dict.items():
        if 30 > count > 2:
            NodummyAvailable = False
            break

    if NodummyAvailable:
        print("Ops, Theres is No suitable Features to make dummy so we continue with current features.")
        with open(dummies_file, 'w') as file:
            file.write('not used any dummy')
        return -1

    print("Here the Recommended Features To make dummy")
    feature_no = 1
    dummies_recommended = list()
    for feature, count in dummy_dict.items():
        if count > 30 or count < 2:
            continue
        dummies_recommended.append(feature)
        print(feature_no, "-", feature, "Have", count, "Different value")
        feature_no += 1

    # Take dummy choices from user as numbers
    print("So what is/are variable(s) you wanna make dummy")
    dummies_choice_str = input("choice: ")
    dummies_choice = dummies_choice_str.split()

    # get the feature that represent that number and add it to list
    str_to_write = ""
    chosen_dummies = list()
    for dummy in dummies_choice:
        chosen_dummies.append(dummies_recommended[int(dummy) - 1])
        str_to_write += dummies_recommended[int(dummy) - 1] + " "

    # return The list
    with open(dummies_file, 'w') as file:
        file.write(str_to_write)
    return chosen_dummies

def PlottingData(dataset, x, y, y_pred, most_corrolated_param, type):
    x_feature = x[:, dataset.columns.get_loc(most_corrolated_param)]
    y_feature = y[:]
    y_feature = y_feature.reshape(-1, 1)

    plt.scatter(x_feature, y_feature, color='blue')
    plt.xlabel(most_corrolated_param)
    plt.ylabel('Predicted Output Values')
    plt.plot(x_feature, y_pred, color='red')
    title_to_show = "The " + type + " Data Plotting"
    plt.title(title_to_show)
    plt.show()

def GetSelectedDummies(file_name):
    with open(file_name) as f:
        dummies = f.read()

    if dummies == 'not used any dummy':
        return -1
    return dummies.split()

def RunTestFile(file_name):
    dummies_file = "Dummies/dummies_selected.txt"
    dataset = pd.read_csv(file_name)

    dataset, output_var = FixNanValues(dataset)
    dataset = CategoricalToNumerical(dataset)
    selective_features_to_dummy = GetSelectedDummies(dummies_file)
    dataset = MakeDummyVariable(dataset, selective_features_to_dummy)

    included_features, _ = TopKPercentage(dataset, output_var, 0.02, isTrain=False)
    dataset = FeatureSelection(dataset, included_features)

    x_test = dataset.drop(output_var, axis=1).values
    y_test = dataset[output_var].values

    scale = StandardScaler()
    x_test = scale.fit_transform(x_test)

    return x_test, y_test



