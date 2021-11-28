# Imported Modules
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

# Modules usage
'''
sns, plt used in DrawHeatmap()
LabelEncoder used in CategoricalToNumerical()
ColumnTransformer, pd, OneHotEncoder used in MakeDummyVariable()
StandardScaler used in FeatureScaling()
'''

def DrawHeatmap(dataset):
    dataset.corr()
    sns.heatmap(dataset.corr(), annot=True)
    plt.show()


def ReplaceCategoricalNan(dataset, features):
    for feature in features:
        dataset.loc[:, feature] = dataset[feature].fillna(dataset[feature].value_counts().idxmax())
    return dataset


def ReplaceNumericalNan(dataset, features):
    for feature in features:
        dataset.loc[:, feature] = dataset[feature].fillna(round(dataset[feature].mean(), 2))
    return dataset


def FixNanValues(dataset):

    # Get output variable
    output_variable = dataset.columns[len(dataset.columns) - 1]

    # Then Get Categorical and Numerical Features from dataset
    categorical_features = list(GetCategoricalFeatures(dataset))
    numerical_features = list(dataset.drop(categorical_features, axis=1).columns)
    numerical_features.remove(output_variable)

    # Remove Records that have 8 Empty Values
    dataset = dataset.dropna(axis=0, thresh=8)

    # Replace null values with mod value(the most repeated value in column) if it is categorical feature
    dataset = ReplaceCategoricalNan(dataset, categorical_features)

    # Replace null values with mean if it is numerical feature
    dataset = ReplaceNumericalNan(dataset, numerical_features)

    # Finally Return Dataset and output_variable
    return dataset, output_variable



def GetCategoricalFeatures(dataset):
    return dataset.select_dtypes(exclude=['float64']).columns


def CategoricalToNumerical(dataset):
    # Call the Function that get categorical features
    categorical_features = GetCategoricalFeatures(dataset)

    # Encode these Categorical Features using Label Encoder
    label_encoder = LabelEncoder()
    for col in categorical_features:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    # Return the Data
    return dataset


def MakeDummyVariable(dataset, features):
    # if not working with any dummies
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

    # Then Remove First Columns of Both
    dataset = dataset.drop(dummy_to_drop , axis=1)
    return dataset


def FeatureSelection(dataset, features):
    return dataset.drop(features, axis=1)

def FeatureScaling(x_train, x_test):
    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.fit_transform(x_test)
    return x_train, x_test

def PlottingData(dataset, x_test, y_test, y_pred, most_correlated_feature):
    user_rating_feature = x_test[:1584, dataset.columns.get_loc(most_correlated_feature)]
    yf = y_test[:1584]
    yf = yf.reshape(-1, 1)

    plt.scatter(user_rating_feature, yf, color='blue')
    plt.xlabel(most_correlated_feature)
    plt.ylabel('Output Variable')
    plt.plot(user_rating_feature, y_pred, color='blue')
    plt.show()

def DummyVariableAcquiring(filename):
    # Get Categorical Feature
    dummies_file = "Testing/dummies_selected.txt"
    dummy_variables_count = list()
    dataset = pd.read_csv(filename)
    categorical_features = list(GetCategoricalFeatures(dataset))

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

def FeatureSelectionAcquring(dataset, dummies, output_var):
    # Get the correlation for MAIN Dataset
    features_file = "Testing/features_selected.txt"
    corr = dataset.corr()
    corr = corr.fillna(0)
    abs_corr = abs(corr[output_var])
    abs_corr = abs_corr.drop(labels=output_var)
    sorted_corr = abs_corr.sort_values(ascending=True)

    # if dummies exists Check if any converted to dummy
    if dummies != -1:
        for dummy in dummies:
            if dummy in sorted_corr:
                sorted_corr = sorted_corr.drop(index=dummy, axis=1)

    # Ask user for removing not needed features
    print("Do you want to use Recommended features or using your own ones?")
    print("1- use recommended features.")
    print("2- use my own features.")
    feat_sel = int(input("your choice: "))

    if feat_sel == 1:
        removed_features_size = 5
    else:
        print("Here is The features dataset and its correlation with {}".format(output_var))
        print(sorted_corr)
        removed_features_size = int(input("so what is the number of features you need to remove: "))

    # Then add not needed to list to remove
    str_to_write = ""
    features_to_remove = list()
    for i in range(removed_features_size):
        features_to_remove.append(sorted_corr.index[i])
        str_to_write += sorted_corr.index[i] + ' '

    # Show deleted features then return this list
    print("Fine .. selected features to delete: {}".format(features_to_remove))
    with open(features_file, 'w') as file:
        file.write(str_to_write)
    return features_to_remove, sorted_corr.index[len(sorted_corr)-1]


def GetSelectedDummy(dummies_file):
    with open(dummies_file) as f:
        dummies = f.read()
    if dummies == 'not used any dummy':
        return -1
    return dummies.split()

def GetSelectedFeatures(features_file):
    with open(features_file) as f:
        dummies = f.read()
    return dummies.split()

def RunTestCase(file_name):
    dummies_file = "Testing/dummies_selected.txt"
    features_file = "Testing/features_selected.txt"

    dataset = pd.read_csv(file_name)
    dataset, output_var = FixNanValues(dataset)
    dataset = CategoricalToNumerical(dataset)

    dummies_features = GetSelectedDummy(dummies_file)
    dataset = MakeDummyVariable(dataset, dummies_features)
    feature_selected = GetSelectedFeatures(features_file)
    dataset = FeatureSelection(dataset, feature_selected)

    x_test = dataset.drop(output_var, axis=1).values
    y_test = dataset[output_var].values

    return x_test, y_test

