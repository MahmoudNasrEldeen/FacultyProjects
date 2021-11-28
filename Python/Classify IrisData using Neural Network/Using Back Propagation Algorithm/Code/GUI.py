# Imported Modules
from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont

import pandas as pd
    
from NN_BackPropagation import *

# functions used on creating GUI
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

# functions used to get data from GUI
def GetHiddenLayers():
    return int(txt_HiddenLayers.get())
def GetNeuronsNumber():
    neurons = txt_NeuronsNumber.get()
    neurons_list = neurons.split()
    neurons_num = [int(neuron) for neuron in neurons_list]
    return neurons_num
def GetLearningRate():
    return float(txt_LearningRate.get())
def GetEpochNumber():
    return int(txt_Epochs.get())
def GetBiasDesicion():
    return int(cb_bias.get())
def GetSelectedFunction():
    selected_function = cmb_functions.get()
    return selected_function

# The Core of program function
def RunWholeProgram():
    # Read Dataset
    dataset = pd.read_csv('Dataset/IrisData.txt')
    output_header = dataset.columns[-1]

    # Get Results From Form
    hidden_layers = GetHiddenLayers()
    neurons_number = GetNeuronsNumber()
    eta_value = GetLearningRate()
    epochs_num = GetEpochNumber()
    bias_decision = GetBiasDesicion()
    function_used = GetSelectedFunction()

    # Make One Hot Encoder for three classes
    dataset = MakeHotEncoderForOutput(dataset, output_header)

    # Here add bias with value 1 or zero based on user choice
    dataset = WorkwithBias(dataset, bias_decision)

    # Make Preprocessing for input data
    trained_data, tested_data = inputDataPreprocessing(dataset)

    # Preparing Parameters used in training
    whole_trained_data = trained_data.to_numpy()
    data_input, actual_output = SplitInputsAndOutput(whole_trained_data)
    weights = GetSuitableWeight(hidden_layers, neurons_number)

    # Apply Back Propagation Algorithm
    optimal_weights = BackPropagationAlgorithm(inputs=data_input,
                                               output=actual_output,
                                               weight=weights,
                                               activation=function_used,
                                               epochs=epochs_num,
                                               eta=eta_value,
                                               layers=hidden_layers,
                                               neurons=neurons_number,
                                               useBias=bias_decision)

    # Preparing parameters used in testing
    total_neurons_values = CreateNeuronsMat(neurons_number)
    whole_tested_data = tested_data.to_numpy()
    inputs_test, output_test = SplitInputsAndOutput(whole_tested_data)

    # The Apply Testing Scenario by Calculate prediction for all network inputs
    network_output = ApplyTestScenario(inputs=inputs_test,
                                       weights=optimal_weights,
                                       activ_function=function_used,
                                       hidden=hidden_layers,
                                       neurons_output=total_neurons_values,
                                       useBias=bias_decision)

    # Building up Confusion Matrix and print Accuracy
    CreateConfusionMatrix(output_test, network_output)


# ------------------------------- START OF MAIN -----------------------------------
if __name__ == "__main__":
    # Create main form
    master = CreateForm(size="400x450", title="TASK 3 SOLUTION")
    fontStyle = tkFont.Font(family="JetBrains Mono", size=10)

    # Create Label and TextBox for Learning Rate
    CreateLabel(text="Enter Number of Hidden Layers")
    txt_HiddenLayers = CreateTextbox()

    # Create Label and TextBox for Learning Rate
    CreateLabel(text="Enter Number of Neurons in Each Hidden Layer")
    txt_NeuronsNumber = CreateTextbox()

    # Create Label and TextBox for Learning Rate
    CreateLabel(text="Enter Learning rate Value")
    txt_LearningRate = CreateTextbox()

    # Create Label and TextBox for Epochs
    CreateLabel(text="Enter Epochs Number")
    txt_Epochs = CreateTextbox()

    # Create Label and Checkbox for bias
    CreateLabel(text="Please Check this for Selecting Bias")
    cb_bias = CreateCheckbox(text="Add Bias")

    # Create Label and Combobox for Activation function
    activation_functions = ('Sigmoid', 'Hyperbolic Tangent Sigmoid')
    CreateLabel(text="Select The activation function to use")
    cmb_functions = CreateComboBox(data=activation_functions)

    # Create Button
    btn_submit = Button(master=master, text="Submit", width=15, command=RunWholeProgram, font=fontStyle)
    btn_submit.pack(pady=15)

    # Run the Form
    master.mainloop()
