import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
class Data:

    def data_preprocessing(self):
        # Load the dataset from the Excel file
        file_path = 'Dry_Bean_Dataset.csv'
        df = pd.read_csv(file_path)

        # Map the classes (BOMBAY -> 100) (CALI -> 010) (SIRA -> 001)
        # Remove ['Class'] column and add ['Class_BOMBAY'] , ['Class_CALI'] , ['Class_SIRA'] columns
        # Convert class to one-hot encoding
        one_hot_encoded = pd.get_dummies(df['Class'], prefix='Class', columns=['BOMBAY', 'CALI', 'SIRA'])

        # Concatenate the new columns to the original dataframe
        df = pd.concat([df, one_hot_encoded], axis=1)

        # Drop the original 'Class' column
        df = df.drop('Class', axis=1)

        # Fill NAN values
        df.interpolate(method='linear', inplace=True)

        # Extracting the first 5 columns as features
        features = df.iloc[:, :5]

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit and transform the scaler on the features
        normalized_features = scaler.fit_transform(features)

        # Replace the original features with the normalized ones
        df.iloc[:, :5] = normalized_features

        # Separate the dataset into 3 classes after mapping
        class1_df = df.head(50)
        class2_df = df.iloc[50:100]
        class3_df = df.tail(50)

        # Set a random seed
        np.random.seed(42)

        # Shuffle each class rows
        class1_df = class1_df.sample(frac=1, random_state=42).reset_index(drop=True)
        class2_df = class2_df.sample(frac=1, random_state=42).reset_index(drop=True)
        class3_df = class3_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Divide data to train (30 rows) & test (20 rows) for each class
        class1_df_train = class1_df.iloc[:30]
        class1_df_test = class1_df.iloc[30:50]
        class2_df_train = class2_df.iloc[:30]
        class2_df_test = class2_df.iloc[30:50]
        class3_df_train = class3_df.iloc[:30]
        class3_df_test = class3_df.iloc[30:50]

        # Combine train rows
        train = pd.concat([class1_df_train, class2_df_train, class3_df_train], axis=0, ignore_index=True)

        # Combine test rows
        test = pd.concat([class1_df_test, class2_df_test, class3_df_test], axis=0, ignore_index=True)

        # Set a random seed
        #np.random.seed(40)

        # Shuffle each class rows
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        test = test.sample(frac=1, random_state=42).reset_index(drop=True)

        # Extract features and target separately for each class
        X_train = train[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']].values
        y_train = train[['Class_BOMBAY', 'Class_CALI', 'Class_SIRA']].values
        X_test = test[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']].values
        y_test = test[['Class_BOMBAY', 'Class_CALI', 'Class_SIRA']].values

        return X_train, X_test, y_train, y_test, train, test

class Models:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def train_multiperceptron(self, x, y, bias, layers_num, neurons_num, use_sigmoid):

        # Add the number of neurons of input & output layers to the array of number of neurons
        neurons_num.insert(0, 5)
        neurons_num.insert(len(neurons_num), 3)

        # Initialize bias for each layer
        bias_list = []
        for i in range(layers_num + 1):
            bias_values = np.random.rand(neurons_num[i + 1])
            bias_list.append(bias_values)

        # Check for bias
        if bias:
            bias_flag = 1
        else:
            bias_flag = 0

        # Initialize weights for each layer
        weights_list = []
        for i in range(layers_num + 1):
            weights = np.random.rand(neurons_num[i + 1], neurons_num[i])
            weights_list.append(weights)

        for _ in range(self.n_iterations):
            for Xi, Yi in zip(x, y):

                # Store the prediction & gradiant values for all neurons
                pred_list = []
                pred_list.append(Xi)
                grad_list = []
                # ****Feed Forward****
                for i in range(layers_num + 1):  # Loop on every hidden layer & the output layer
                    # Store the prediction values for neurons of each layer
                    prediction_list = []
                    for j in range(neurons_num[i + 1]):  # Loop on each neuron
                        # Calculate net value for each neuron
                        net_value = np.dot(pred_list[len(pred_list) - 1], weights_list[i][j].T) + (bias_list[i][j] * bias_flag)
                        # Calculate prediction value after using Activation function
                        if use_sigmoid:  # Using Sigmoid function (where a = 1)
                            prediction = 1 / (1 + math.exp(-net_value))
                        else:  # Using Hyperbolic tangent function (where a = 1)
                            prediction = (1 - math.exp(-net_value)) / (1 + math.exp(-net_value))

                        # Store the prediction value for each neuron
                        prediction_list.append(prediction)

                    # Store the neurons value for each layer
                    pred_list.append(prediction_list)
                pred_list.pop(0)

                # ****Back Propagation***
                # Calculate gradiant for the output layer only
                gradiant_list = []
                for j in range(neurons_num[len(neurons_num) - 1]):  # Loop on each neuron on output layer
                    if use_sigmoid:  # Using Sigmoid function (where a = 1)
                        gradiant = (Yi[j] - pred_list[layers_num][j]) * (pred_list[layers_num][j]) * (1 - pred_list[layers_num][j])
                    else:  # Using Hyperbolic tangent function (where a = 1 & b = 1)
                        gradiant = (Yi[j] - pred_list[layers_num][j]) * (1 + pred_list[layers_num][j]) * (1 - pred_list[layers_num][j])
                    # Store gradiant value for each neuron
                    gradiant_list.append(gradiant)
                # Store the neurons value for each layer
                grad_list.append(gradiant_list)

                # Calculate gradiant for the hidden layers
                current_layer = layers_num - 1
                for i in range(layers_num, 0, -1):  # Loop backward on every hidden layer
                    gradiantlist = []
                    for j in range(neurons_num[i]):  # Loop on each neuron
                        # Calculate the sum between weight & gradiant of previous layer for each neuron
                        backward_weights = []
                        for k in range(len(weights_list[i])):
                            backward_weights.append(weights_list[i][k][j])
                        sum_value = np.dot(grad_list[0], backward_weights)
                        if use_sigmoid:  # Using Sigmoid function (where a = 1)
                            gradiant = (pred_list[current_layer][j]) * (1 - pred_list[current_layer][j]) * sum_value
                        else:  # Using Hyperbolic tangent function (where a = 1 & b = 1)
                            gradiant = (1 - pred_list[current_layer][j]) * (1 + pred_list[current_layer][j]) * sum_value
                        # Store gradiant value for each neuron
                        gradiantlist.append(gradiant)
                    # Store the neurons value for each layer
                    grad_list.insert(0, gradiantlist)
                    current_layer = current_layer - 1

                # ****Update Weights****
                for i in range(layers_num + 1):  # Loop on every layer
                    for j in range(neurons_num[i + 1]):  # Loop on each neuron
                        for k in range(len(weights_list[i][j])):  # Loop on all weights
                            if i == 0:  # Weights between input layer and first hidden layer
                                weights_list[i][j][k] = weights_list[i][j][k] + (self.learning_rate * grad_list[i][j] * Xi[k])
                                bias_list[i][j] = bias_list[i][j] + (self.learning_rate * grad_list[i][j] * bias_flag)
                            else:
                                weights_list[i][j][k] = weights_list[i][j][k] + (self.learning_rate * grad_list[i][j] * pred_list[i][j])
                                bias_list[i][j] = bias_list[i][j] + (self.learning_rate * grad_list[i][j] * bias_flag)
        return weights_list, bias_list

    def test(self, x, y, weights, bias, layers_num, neurons_num, use_sigmoid, bias_list):

        # Check for bias
        if bias:
            bias_flag = 1
        else:
            bias_flag = 0

        final_pred = []
        for Xi, Yi in zip(x, y):
            pred_list = []
            pred_list.append(Xi)
            for i in range(layers_num + 1):  # Loop on every hidden layer & the output layer
                # Store the prediction values for neurons of each layer
                prediction_list = []
                for j in range(neurons_num[i + 1]):  # Loop on each neuron
                    # Calculate net value for each neuron
                    net_value = np.dot(pred_list[len(pred_list) - 1], weights[i][j].T) + (bias_list[i][j] * bias_flag)
                    # Calculate prediction value after using Activation function
                    if use_sigmoid:  # Using Sigmoid function (where a = 1)
                        prediction = 1 / (1 + math.exp(-net_value))
                    else:  # Using Hyperbolic tangent function (where a = 1)
                        prediction = (1 - math.exp(-net_value)) / (1 + math.exp(-net_value))
                    # Store the prediction value for each neuron
                    prediction_list.append(prediction)
                # Store the neurons value for each layer
                pred_list.append(prediction_list)
                # Store the output values
                if i == layers_num:
                    final_pred.append(prediction_list)

        #for i in range(len(y)):
        #    print('desired output')
        #    print(y[i][0], 'and', y[i][1], 'and', y[i][2])
        #    print('the output')
        #    print(final_pred[i][0], 'and', final_pred[i][1], 'and', final_pred[i][2])
        #    print('-----------------------------------------------------------------------')

        # Map all outputs into 0 or 1
        for i in range(len(y)):
            if final_pred[i][0] == max(final_pred[i]):
                final_pred[i][0] = 1
                final_pred[i][1] = 0
                final_pred[i][2] = 0
            elif final_pred[i][1] == max(final_pred[i]):
                final_pred[i][0] = 0
                final_pred[i][1] = 1
                final_pred[i][2] = 0
            elif final_pred[i][2] == max(final_pred[i]):
                final_pred[i][0] = 0
                final_pred[i][1] = 0
                final_pred[i][2] = 1

        # for i in range(len(y)):
        #     print('desired output')
        #     print(y[i][0], 'and', y[i][1], 'and', y[i][2])
        #     print('the output')
        #     print(final_pred[i][0], 'and', final_pred[i][1], 'and', final_pred[i][2])
        #     print('-----------------------------------------------------------------------')

        # Calculate the correct predictions
        correct = 0
        for i in range(len(y)):
            if y[i][0] == 1 & y[i][1] == 0 & y[i][2] == 0:
                if final_pred[i] == [1, 0, 0]:
                    correct += 1
            elif y[i][0] == 0 & y[i][1] == 1 & y[i][2] == 0:
                if final_pred[i] == [0, 1, 0]:
                    correct += 1
            else:
                if final_pred[i] == [0, 0, 1]:
                    correct += 1

        accuracy = correct / len(y)
        if use_sigmoid:
            print(f"Accuracy of Multy Layer Perceptron using Sigmoid: {accuracy * 100:.2f}%")
        else:
            print(f"Accuracy of Multy Layer Perceptron using Hyperbolic Tangent: {accuracy * 100:.2f}%")


        confusion_matrix = np.zeros((3, 3))
        for i in range(len(y)):
            if y[i][0] == 1 & y[i][1] == 0 & y[i][2] == 0:
                if final_pred[i] == [1, 0, 0]:
                    confusion_matrix[0, 0] += 1  # True Positive
                elif final_pred[i] == [0, 1, 0]:
                    confusion_matrix[0, 1] += 1  # False Negative
                else:
                    confusion_matrix[0, 2] += 1  # False Negative

            elif y[i][0] == 0 & y[i][1] == 1 & y[i][2] == 0:
                if final_pred[i] == [1, 0, 0]:
                    confusion_matrix[1, 0] += 1  # False Positive
                elif final_pred[i] == [0, 1, 0]:
                    confusion_matrix[1, 1] += 1  # True Negative
                else:
                    confusion_matrix[1, 2] += 1  # True Negative

            else:
                if final_pred[i] == [1, 0, 0]:
                    confusion_matrix[2, 0] += 1  # False Positive
                elif final_pred[i] == [0, 1, 0]:
                    confusion_matrix[2, 1] += 1  # True Negative
                else:
                    confusion_matrix[2, 2] += 1  # True Negative

        print("Confusion Matrix:")
        print(confusion_matrix)
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        return accuracy, confusion_matrix

# Running The Model
'''
data = Data()

X_train, X_test, y_train, y_test, x, y = data.data_preprocessing()

model = Models(learning_rate=0.1, n_iterations=100)

neu = [10, 5, 4]
# Train the Perceptron model
weights, biaslist = model.train_multiperceptron(X_train, y_train, bias=False, layers_num=3, neurons_num=neu, use_sigmoid=True)

# Testing the model
model.test(X_test, y_test, weights, bias=False, layers_num=3, neurons_num=neu, use_sigmoid=True, bias_list=biaslist)
'''