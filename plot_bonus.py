# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_bonus.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: xmatute- <xmatute-@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/08/14 16:53:05 by xmatute-          #+#    #+#              #
#    Updated: 2025/08/14 17:26:02 by xmatute-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_DATA_FILE = 'data/data.csv'
DEFAULT_MODEL_FILE = 'model.csv'

FIGSIZE = (10, 6)

def calculate_MSE(predictions, price):
    mse = sum((p - a) ** 2 for p, a in zip(predictions, price)) / len(price)
    return mse

def calculate_R2(predictions, price):
    mean_price = sum(price) / len(price)
    ss_total = sum((a - mean_price) ** 2 for a in price)
    ss_residual = sum((a - p) ** 2 for p, a in zip(predictions, price))
    r2 = 1 - (ss_residual / ss_total)

    return r2

def main():
    print("Plotting car price data...")
    data_file = DEFAULT_DATA_FILE
    model_file = DEFAULT_MODEL_FILE

    if sys.argv[1:]:
        if len(sys.argv) > 2:
            print("Usage: python plot_bonus.py [data_file] [model_file]")
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

    plt.figure(figsize=FIGSIZE)

    plt.scatter(km, price, color='blue', alpha=0.7, label='Data points')

    if os.path.exists(model_file):
        try:
            with open(model_file, 'r') as f:
                theta0, theta1 = map(float, f.readline().split(','))
            x_line = [0, - theta0 / theta1] if theta1 != 0 else [km.min(), km.max()]
            y_line = [theta0 + theta1 * x for x in x_line]
            plt.plot(x_line, y_line, color='red', label='Regression Line')


            predictions = [theta0 + theta1 * x for x in km]

            mse = calculate_MSE(predictions, price)
            r2 = calculate_R2(predictions, price)

            print(f"Model Metrics:\nMSE: {mse:.2f}\nR²: {r2:.2f}")

            metrics_text = f"MSE: {mse:.2f}\nR²: {r2:.2f}"
            plt.text(0.05, 0.05, metrics_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
            
        except (IOError, ValueError):
            print(f"Warning: Could not read model from {model_file}. Skipping regression line.")
    else:
        print(f"{model_file} not found. Skipping regression line.")

    plt.title('Car Price vs Mileage')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()