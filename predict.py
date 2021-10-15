import numpy as np
import csv
import sys


from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = list(np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1))
    train_X = list(np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64, skip_header=1))
    train_Y = list(np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.int))
    return test_X, train_X, train_Y


def calculate_accuracy(predicted_Y, actual_Y):
    correct = 0
    for i in range(len(predicted_Y)):
        if predicted_Y[i] == actual_Y[i]:
            correct += 1
    accuracy = correct/len(predicted_Y)
    return accuracy


def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n):
    import math
    total_num_of_observations = len(train_X)
    train_length = math.floor((float(100 - validation_split_percent))/100 * total_num_of_observations )
    validation_X = train_X[train_length:]
    validation_Y = train_Y[train_length:]
    train_X = train_X[0:train_length]
    train_Y = train_Y[0:train_length]

    best_k = -1
    best_accuracy = 0
    for k in range(1, train_length+1):
        predicted_Y = classify_points_using_knn(train_X,train_Y,validation_X, k, n)
        accuracy = calculate_accuracy(predicted_Y,validation_Y)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return best_k


def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    distance = 0
    for i in range(0, vector_len):
      diff = (abs(vector1[i] - vector2[i]))**n
      distance += diff
    distance = distance ** (1.0/n)
    return distance


def find_k_nearest_neighbors(train_X, test_example, k, n):
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
      distance = compute_ln_norm_distance(train_elem_x, test_example, n)
      indices_dist_pairs.append([index, distance])
      index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices


def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_elem_x in test_X:
        top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k, n)
        top_knn_labels = []

        for i in top_k_nn_indices:
            top_knn_labels.append(train_Y[i])
        Y_values = sorted(list(set(top_knn_labels)))

        max_count = 0
        most_frequent_label = -1
        for y in Y_values:
            count = top_knn_labels.count(y)
            if (count > max_count):
                max_count = count
                most_frequent_label = y

        test_Y.append(most_frequent_label)
    return test_Y


def predict_target_values(test_X, train_X, train_Y):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    k = get_best_k_using_validation_set(train_X, train_Y, 30, 2)
    test_Y = classify_points_using_knn(train_X, train_Y, test_X, k, 2)
    return test_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, train_X, train_Y = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X, train_X, train_Y)
    pred_Y = np.array(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv")