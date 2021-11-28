# Imported Modules
import pandas as pd
from Preprocessing import DrawHeatmap, FixNanValues, CategoricalToNumerical, MakeDummyVariable,\
                          FeatureSelection, DummyVariableAcquiring, FeatureSelectionAcquring
from sklearn.model_selection import train_test_split
from train_models import RunAllModels

# Ignore Warning [It Doesn't effect on the efficiency of Code]
import warnings
warnings.filterwarnings("ignore")

# Step 1: Reading Dataset From File
file_path = 'Dataset/AppleStore_training.csv'
dataset = pd.read_csv(file_path)

# Step 2: Draw Heatmap
DrawHeatmap(dataset)

# Step 3: Make Preprocessing For Dataset
dataset, output_var = FixNanValues(dataset)
temp_dataset = dataset
dataset = CategoricalToNumerical(dataset)
dummies_features = DummyVariableAcquiring(file_path)
dataset = MakeDummyVariable(dataset, dummies_features)
feature_selected, most_correlated_feature = FeatureSelectionAcquring(temp_dataset, dummies_features, output_var)
dataset = FeatureSelection(dataset, feature_selected)

# Step 4: Split dataset into Train and Test
X = dataset.drop(output_var, axis=1).values
Y = dataset[output_var].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)

# Step 5: Train Models
RunAllModels(dataset, x_train, x_test, y_train, y_test, most_correlated_feature)
