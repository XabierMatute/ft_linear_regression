# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: xmatute- <xmatute-@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/08/14 16:52:59 by xmatute-          #+#    #+#              #
#    Updated: 2025/08/14 17:10:56 by xmatute-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import sys
import os

DEFAULT_DATA_FILE = 'data/data.csv'
DEFAULT_MODEL_FILE = 'model.csv'

DEFAULT_THETA0 = 0.0
DEFAULT_THETA1 = 0.0

LEARNING_RATE = 0.1
MAX_ITERATIONS = 100000
MIN_ERROR = 0.0001

def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def denormalize(normalized_data, original_data):
    return normalized_data * (original_data.max() - original_data.min()) + original_data.min()

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    data_file = DEFAULT_DATA_FILE
    model_file = DEFAULT_MODEL_FILE

    if sys.argv[1:]:
        if len(sys.argv) > 2:
            print("Usage: python train.py [data_file] [model_file]")
            return
        data_file = sys.argv[1]
        if len(sys.argv) == 3:
            model_file = sys.argv[2]

    if os.path.exists(data_file):
        try:
            data = pd.read_csv(data_file)
            if 'km' not in data.columns or 'price' not in data.columns:
                raise KeyError("Required columns 'km' and 'price' not found.")
        except (FileNotFoundError, KeyError) as e:
            print(f"Error: Could not read data from {data_file}. {e}")
            return
    else:
        print(f"Error: Data file {data_file} does not exist.")
        return
    
    km = data['km']
    price = data['price']

    km_normalized = normalize(km)
    price_normalized = normalize(price)

    learning_rate = LEARNING_RATE
    max_iterations = MAX_ITERATIONS
    min_error = MIN_ERROR
    min_error_normalized = min_error / (price.max() - price.min())
    m = len(km_normalized)

    # theta0_denormalized = DEFAULT_THETA0
    # theta1_denormalized = DEFAULT_THETA1

    # if os.path.exists(model_file):
    #     try:
    #         with open(model_file, 'r') as f:
    #             line = f.readline()
    #             theta0_denormalized, theta1_denormalized = map(float, line.split(','))
    #             print(f"Loaded existing model: theta0={theta0}, theta1={theta1}")
                
    #     except (IOError, ValueError):
    #         print(f"Warning: Could not read {model_file}. Using default values ({theta0}, {theta1}).")

    # theta1 = theta1_denormalized * (price.max() - price.min()) / (km.max() - km.min())
    # theta0 = (price.max() - price.min()) * theta0_denormalized + price.min() - theta1 * km.min()

    theta0 = DEFAULT_THETA0
    theta1 = DEFAULT_THETA1

    for _ in range(max_iterations):
        price_prediction_normalized = predict_price(km_normalized, theta0, theta1)

        error = price_prediction_normalized - price_normalized
        
        theta0 -= learning_rate * (1/m) * sum(error)
        theta1 -= learning_rate * (1/m) * sum(error * km_normalized)

        if error.abs().mean() < min_error_normalized:
            print(f"Converged after {_} iterations.")
            break


    theta1_denormalized = theta1 * (price.max() - price.min()) / (km.max() - km.min())
    theta0_denormalized = (price.max() - price.min()) * theta0 + price.min() - theta1_denormalized * km.min()

    try:
        with open(model_file, 'w') as f:
            f.write(f"{theta0_denormalized},{theta1_denormalized}")
        print(f"Training complete. Model saved to {model_file}.")
    except IOError:
        print(f"Error: Could not save the model to {model_file}. Please check file permissions.")

if __name__ == "__main__":
    main()