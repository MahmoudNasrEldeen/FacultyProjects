# Imported Modules
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import FeatureScaling, DrawHeatmap, FixNanValues, CategoricalToNumerical, \
                          DummyVariableAcquiring, MakeDummyVariable, TopKPercentage, FeatureSelection
from trained_models import RunWholeModels

# Ignore Warning [It Doesn't effect on the efficiency of Code]
import warnings
warnings.filterwarnings("ignore")

# Step 1: Reading Dataset From File
file_name = 'Dataset/AppleStore_training_classification.csv'
dataset = pd.read_csv(file_name)

# Step 3: Make Preprocessing For Dataset + Draw Heatmap for Data
dataset, output_var = FixNanValues(dataset)
dataset = CategoricalToNumerical(dataset)
DrawHeatmap(dataset)
Selective_features_to_dummy = DummyVariableAcquiring(file_name)
dataset = MakeDummyVariable(dataset, Selective_features_to_dummy)

# Step 4: Feature Selection
included_features, sorted_corr = TopKPercentage(dataset, output_var, 0.02)
dataset = FeatureSelection(dataset, included_features)
most_correlated_feature = sorted_corr.index[0]

# Step 5: Split dataset into Train and Test
X = dataset.drop(output_var, axis=1).values
Y = dataset[output_var].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Step 6: Make Feature Scaling
x_train, x_test = FeatureScaling(x_train, x_test)

# Step 7: Train Models
RunWholeModels(dataset, x_train, x_test, y_train, y_test, most_correlated_feature)
