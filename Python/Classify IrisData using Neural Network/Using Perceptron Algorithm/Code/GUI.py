# Imported Modules
from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
from NeuralNetwork import *

# Defined Functions
def CreateForm(size, title):
    form = Tk()
    form.geometry(size)
    form.title(title)
    form.resizable(False, False)
    return form

def CreateLabel(text):
    lbl_var = StringVar()
    lbl_var.set(text)
    lbl_features = Label(master=master, textvariable=lbl_var, font=fontStyle)
    lbl_features.pack(fill='x', padx=5, pady=5)

def CreateComboBox(data):
    cmb_obj = ttk.Combobox(master=master, textvariable=StringVar(), font=fontStyle)
    cmb_obj['values'] = data
    cmb_obj['state'] = 'readonly'
    cmb_obj.pack(fill='x', padx=5, pady=5)
    return cmb_obj

def CreateTextbox():
    txt_entry = Entry(master, font=fontStyle)
    txt_entry.pack(fill='x', padx=5, pady=5)
    return txt_entry

def CreateCheckbox(text):
    cb_var = IntVar()
    cb_obj = Checkbutton(master=master, text=text, variable=cb_var, onvalue=1, offvalue=0, font=fontStyle)
    cb_obj.pack()
    return cb_var

def GetSelectedFeatures():
    selected_option = features_cmb.get()
    selected_features = selected_option.split()
    selected_features.remove('and')
    return selected_features

def GetSelectedClasses():
    selected_option = classes_cmb.get()
    selected_classes = selected_option.split()
    selected_classes.remove('and')
    return selected_classes

def GetLearningRate():
    return float(txt_LearningRate.get())

def GetEpochNumber():
    return int(txt_Epochs.get())

def GetBiasDesicion():
    return int(use_bias.get())

def RunWholeProgram():
    # Defined Variables
    filename = "Dataset/IrisData.txt"
    dataset = pd.read_csv(filename)
    output_header = dataset.columns[-1]

    # Get Results From Form
    user_features = GetSelectedFeatures()
    user_classes = GetSelectedClasses()
    user_lr = GetLearningRate()
    user_epochs = GetEpochNumber()
    user_bias = GetBiasDesicion()

    # Remove un-needed flower Class
    flowers = {'C1': 'Iris-setosa', 'C2': 'Iris-versicolor', 'C3': 'Iris-virginica'}
    needed_flowers = list()
    for flower_class, flower_name in flowers.items():
        if flower_class not in user_classes:
            dataset = dataset[dataset[output_header] != flower_name]
        else:
            needed_flowers.append(flower_name)

    # Replace the flower name with 1 and -1 respectively
    numerical_output = [1, -1]
    dataset[output_header] = dataset[output_header].replace(needed_flowers, numerical_output)

    # Filter the dataset to selected features
    all_features = ['X1', 'X2', 'X3', 'X4']
    not_needed_features = list()
    for feature in all_features:
        if feature not in user_features:
            not_needed_features.append(feature)
    dataset = dataset.drop(not_needed_features, axis=1)

    # Get Classes Values then Shuffle it
    class_a, class_b = dataset[:50], dataset[50:]
    class_a, class_b = class_a.sample(frac=1), class_b.sample(frac=1)

    # Then Split The data into 30 Train and 20 Test
    class_a_train, class_a_test = class_a[:30], class_a[30:]
    class_b_train, class_b_test = class_b[:30], class_b[30:]

    # Merge trained data together and tested data together
    trained_data = pd.concat([class_a_train, class_b_train])
    tested_data = pd.concat([class_a_test, class_b_test])

    # Preparing Parameters
    whole_trained_data = trained_data.to_numpy()
    inputs_vector, actual_output = SplitInputsAndOutput(whole_trained_data)
    weights_vector = np.array(np.random.randn(2))
    bias = 0 if user_bias == 0 else np.random.randn()

    # Training Scenario
    weights_vector, bias = PerceptronLearningAlgorithm(input=inputs_vector,
                                                       actual_output=actual_output,
                                                       weight=weights_vector,
                                                       bias=bias,
                                                       epochs=user_epochs,
                                                       alpha=user_lr)

    # Then Draw Line
    point_a_x = trained_data[user_features[0]].min()
    point_b_y = trained_data[user_features[0]].max()
    point_a, point_b = GetPointsForLine(point_a_x, point_b_y, weights_vector, bias)
    DrawLine(trained_data, user_features, point_a, point_b, actual_output)

    # Testing Scenario
    whole_tested_data = tested_data.to_numpy()
    inputs_vector_test, actual_output_test = SplitInputsAndOutput(whole_tested_data)
    prediction = MakePrediction(inputs_vector_test, weights_vector, bias)

    # Building up Confusion Matrix
    test_accuracy = CreateConfuionMatrix(actual_output_test, prediction)
    print("accuracy: {}".format(test_accuracy))


####################### Start of Main ###########################
# Create main form
master = CreateForm(size="350x380", title="TASK 1 SOLUTION")
fontStyle = tkFont.Font(family="JetBrains Mono", size=10)

# Create Label and Combobox for features
features = ('X1 and X2', 'X1 and X3', 'X1 and X4', 'X2 and X3', 'X2 and X4', 'X3 and X4')
CreateLabel(text="Select The Features")
features_cmb = CreateComboBox(data=features)

# Create Label and Combobox for Classes
classes = ('C1 and C2', 'C1 and C3', 'C2 and C3')
CreateLabel(text="Select The Classes")
classes_cmb = CreateComboBox(data=classes)

# Create Label and TextBox for Learning Rate
CreateLabel(text="Enter Learning rate Value")
txt_LearningRate = CreateTextbox()

# Create Label and TextBox for Epochs
CreateLabel(text="Enter Epochs Number")
txt_Epochs = CreateTextbox()

# Create Label and Checkbox for bias
CreateLabel(text="Please Check this for Selecting Bias")
use_bias = CreateCheckbox(text="Use Bias")

# Create Button
btn_submit = Button(master=master, text="Submit", width=15, command=RunWholeProgram, font=fontStyle)
btn_submit.pack(pady=15)

# Run the Form
master.mainloop()


