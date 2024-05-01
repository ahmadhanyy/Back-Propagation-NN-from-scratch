import re
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import font
from Models import Models
from Models import Data

x_train = None
y_train = None
x = None
y = None
weights = None

def get_text():
    global x_train, y_train, x, y, weights
    layers_num = int(layers.get())
    neurons_num = neurons.get()
    NumberOfEpochs = int(epochs.get())
    ETALearningRate = float(learnrate.get())
    activation_function = var.get()
    data = Data()
    x_train, x_test, y_train, y_test, x, y = data.data_preprocessing()

    # Define a pattern to match numeric values
    pattern = r'\b\d+\b'
    # Use re.findall() to find all numeric values in the string
    neuron = re.findall(pattern, neurons_num)
    # Convert the string numbers to integers
    neuron = [int(num) for num in neuron]

    if activation_function == 1:
        activation = True
    else:
        activation = False

    if bias.get() == 1:
        bias_value = True
    else:
        bias_value = False

    model = Models(ETALearningRate, NumberOfEpochs)
    weights, bias_list = model.train_multiperceptron(x_train, y_train, bias=bias_value, layers_num=layers_num, neurons_num=neuron, use_sigmoid=activation)
    accuracy, confusion_matrix = model.test(x_test, y_test, weights, bias=bias_value, layers_num=layers_num, neurons_num=neuron, use_sigmoid=activation, bias_list=bias_list)

    # Display the accuracy and the confusion matrix in the GUI window
    accuracy_label.config(text=f"Accuracy: {accuracy*100}%")
    confusion_matrix_label.config(text=f"Confusion Matrix: \n{confusion_matrix}")

def show_graph():

    global x_train, y_train, x, y, weights
    # Generating the decision boundary
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Flattening the mesh grid and predicting the labels
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    Z = np.dot(mesh_data, weights)
    Z = np.where(Z >= 0, 1, -1)
    Z = Z.reshape(xx.shape)

    # Plotting the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Scatter the data points for class 1 (label 1)
    plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], label='class1', marker='o', color='blue')

    # Scatter the data points for class -1 (label -1)
    plt.scatter(x_train[y_train == -1][:, 0], x_train[y_train == -1][:, 1], label='class2', marker='x', color='red')
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend()
    plt.title('Decision Boundary and Data Points')
    plt.show()

m = tk.Tk()
m.title("Multi Layer Perceptron")
m.geometry("500x500")
bg_color = 'gray'
m.configure(bg=bg_color)

# Labels and Entry fields
label_font = font.Font(family='Helvetica', size=10, weight='bold', slant='italic')
entry_font = font.Font(family='Helvetica', size=10)

tk.Label(m, text='Number of Layers:', bg=bg_color, font=label_font).grid(row=0, column=0, pady=5, padx=10, sticky=tk.W)
layers = tk.Entry(m, font=entry_font)
layers.grid(row=0, column=1, pady=5, padx=10, sticky=tk.W)

tk.Label(m, text='Number of Neurons:', bg=bg_color, font=label_font).grid(row=1, column=0, pady=5, padx=10, sticky=tk.W)
neurons = tk.Entry(m, font=entry_font)
neurons.grid(row=1, column=1, pady=5, padx=10, sticky=tk.W)

tk.Label(m, text='Number of Epochs:', bg=bg_color, font=label_font).grid(row=2, column=0, pady=5, padx=10, sticky=tk.W)
epochs = tk.Entry(m, font=entry_font)
epochs.grid(row=2, column=1, pady=5, padx=10, sticky=tk.W)

tk.Label(m, text='Learning Rate:', bg=bg_color, font=label_font).grid(row=3, column=0, pady=5, padx=10, sticky=tk.W)
learnrate = tk.Entry(m, font=entry_font)
learnrate.grid(row=3, column=1, pady=5, padx=10, sticky=tk.W)

# Radio Buttons
var = tk.IntVar()
tk.Radiobutton(m, text="Sigmoid Function", variable=var, value=1, bg=bg_color, font=label_font).grid(row=4, column=0, pady=5, padx=10, sticky=tk.W)
tk.Radiobutton(m, text="Hyperbolic Tangent Function", variable=var, value=2, bg=bg_color, font=label_font).grid(row=4, column=1, pady=5, padx=10, sticky=tk.W)

# Check Button
bias = tk.IntVar()
tk.Checkbutton(m, text="Add Bias", variable=bias, onvalue=1, offvalue=0, bg=bg_color, font=label_font).grid(row=5, column=0, pady=5, padx=10, sticky=tk.W)

# Action Button
button_font = font.Font(family='Helvetica', size=10, weight='bold', slant='italic')

execute_button = tk.Button(m, text="Execute", width=15, command=get_text, bg='darkgreen', font=button_font)
execute_button.grid(row=6, column=0, pady=15, padx=10, sticky=tk.W)

graph_button = tk.Button(m, text="Show Graph", width=15, command=show_graph, bg='navy', font=button_font)
graph_button.grid(row=6, column=1, pady=15, padx=10, sticky=tk.W)

# Label widgets to display accuracy and confusion matrix
result_label_font = font.Font(family='Helvetica', size=10, weight='bold', slant='italic')

accuracy_label = tk.Label(m, text="Accuracy: ", bg=bg_color, font=result_label_font)
accuracy_label.grid(row=7, column=0, pady=10, padx=10, sticky=tk.W)

confusion_matrix_label = tk.Label(m, text="Confusion Matrix: ", bg=bg_color, font=result_label_font)
confusion_matrix_label.grid(row=7, column=1, pady=10, padx=10, sticky=tk.W)

m.mainloop()
