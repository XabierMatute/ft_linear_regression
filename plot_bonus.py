import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_DATA_FILE = 'data/data.csv'
MODEL_FILE = 'model.csv'

def calculate_metrics(km, price, theta0, theta1):
    """Calcula el MSE y R² para evaluar la precisión del modelo."""
    # Predicciones
    predictions = [theta0 + theta1 * x for x in km]

    # Error Cuadrático Medio (MSE)
    mse = sum((p - a) ** 2 for p, a in zip(predictions, price)) / len(price)

    # Coeficiente de Determinación (R²)
    mean_price = sum(price) / len(price)
    ss_total = sum((a - mean_price) ** 2 for a in price)
    ss_residual = sum((a - p) ** 2 for p, a in zip(predictions, price))
    r2 = 1 - (ss_residual / ss_total)

    return mse, r2

def main():
    print("Plotting car price data...")
    data_file = os.getenv('DATA_FILE', DEFAULT_DATA_FILE)

    # Check arguments
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    print(f"Data file: {data_file}")

    # Load data
    try:
        data = pd.read_csv(data_file)
        km = data['km']
        price = data['price']
    except (FileNotFoundError, KeyError):
        print(f"Error: Could not read data from {data_file}.")
        return

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(km, price, color='blue', alpha=0.7, label='Data points')

    # Load model and plot regression line
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'r') as f:
                theta0, theta1 = map(float, f.readline().split(','))
            # Generate regression line
            x_line = [km.min(), km.max()]
            y_line = [theta0 + theta1 * x for x in x_line]
            plt.plot(x_line, y_line, color='red', label='Regression Line')

            # Calculate and print metrics
            mse, r2 = calculate_metrics(km, price, theta0, theta1)
            print(f"Mean Squared Error (MSE): {mse:.2f}")
            print(f"R² (Coefficient of Determination): {r2:.2f}")

            # Add metrics to the plot (bottom-left corner)
            metrics_text = f"MSE: {mse:.2f}\nR²: {r2:.2f}"
            plt.text(0.05, 0.05, metrics_text, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        except (IOError, ValueError):
            print(f"Warning: Could not read model from {MODEL_FILE}. Skipping regression line.")
    else:
        print(f"Warning: {MODEL_FILE} not found. Skipping regression line.")

    # Finalize plot
    plt.title('Car Price vs Mileage')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (€)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()