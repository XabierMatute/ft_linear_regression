import os
import sys

DEFAULT_MODEL_FILE = 'model.csv'
DEFAULT_THETA0 = 0.0
DEFAULT_THETA1 = 0.0

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    theta0 = 0.0
    theta1 = 0.0
    model_file = DEFAULT_MODEL_FILE

    if sys.argv[1:]:
        if len(sys.argv) > 2:
            print("Usage: python predict.py [model_file]")
            return
        model_file = sys.argv[1]

    if os.path.exists(model_file):
        try:
            with open(model_file, 'r') as f:
                line = f.readline()
                t0_str, t1_str = line.split(',')
                theta0 = float(t0_str)
                theta1 = float(t1_str)
        except (IOError, ValueError):
            print(f"Warning: Could not read {model_file}. Using default values ({DEFAULT_THETA0}, {DEFAULT_THETA1}).")
            print("Please ensure the model file is formatted correctly.")
            print("Or run train.py to generate the model file.")
    else:
        print(f"Warning: {model_file} not found. Using default values ({DEFAULT_THETA0}, {DEFAULT_THETA1}).")
        print("Please run train.py to generate the model file.")

    try:
        mileage = float(input("Please enter a mileage in km: "))
        if mileage < 0:
            print("Mileage cannot be negative.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    estimated_price = predict_price(mileage, theta0, theta1)
    print(f"For a mileage of {mileage} km, the estimated price is: {estimated_price:.2f}€")

if __name__ == "__main__":
    main()