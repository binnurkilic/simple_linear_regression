import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function to generate synthetic data
def generate_data(size=200, noise_scale=0.15):
    """
    Generate random data for linear regression.
    Args:
        size (int): Number of data points to generate.
        noise_scale (float): Standard deviation of noise added to the data.
    Returns:
        x (ndarray): Input feature values.
        y (ndarray): Corresponding labels with some added noise.
    """
    x = np.random.uniform(0, 1, size=size)
    y = 2 * x + 5 + np.random.normal(scale=noise_scale, size=size)
    return x, y

# Function to split data into training and testing sets
def split_data(x, y, train_size=0.8):
    """
    Split the dataset into training and testing sets.
    Args:
        x (ndarray): Input feature values.
        y (ndarray): Corresponding labels.
        train_size (float): Fraction of data to be used for training.
    Returns:
        X_train, y_train, X_test, y_test: Split data sets.
    """
    split_idx = int(len(x) * train_size)
    X_train, y_train = x[:split_idx], y[:split_idx]
    X_test, y_test = x[split_idx:], y[split_idx:]
    return X_train, y_train, X_test, y_test

# Function to train a linear regression model
def train_model(X_train, y_train):
    """
    Train a linear regression model on the training data.
    Args:
        X_train (ndarray): Training feature values.
        y_train (ndarray): Training labels.
    Returns:
        model (LinearRegression): Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train.reshape(-1, 1), y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using R-squared metric.
    Args:
        model (LinearRegression): Trained model.
        X_test (ndarray): Test feature values.
        y_test (ndarray): Test labels.
    Returns:
        r2 (float): R-squared score of the model.
    """
    y_predict = model.predict(X_test.reshape(-1, 1))
    r2 = r2_score(y_test, y_predict)
    return r2

# Function to plot results
def plot_results(X_train, y_train, X_test, y_test, model):
    """
    Plot the training and testing data points, and the regression line.
    Args:
        X_train (ndarray): Training feature values.
        y_train (ndarray): Training labels.
        X_test (ndarray): Test feature values.
        y_test (ndarray): Test labels.
        model (LinearRegression): Trained linear regression model.
    """
    # Generate points for plotting the regression line
    x_range = np.arange(0, 1, 0.01)
    y_predict_range = model.predict(x_range.reshape(-1, 1))

    # Plotting
    plt.plot(x_range, y_predict_range, color='black', ls='--', alpha=0.9)
    plt.scatter(X_train, y_train, color='red', label='Training data')
    plt.scatter(X_test, y_test, color='blue', label='Testing data')

    plt.xlabel('Feature $x$')
    plt.ylabel('Label $y$')
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.show()

# Main function to run the linear regression workflow
def main():
    # Generate data
    x, y = generate_data()

    # Split data into training and testing sets
    X_train, y_train, X_test, y_test = split_data(x, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    r2 = evaluate_model(model, X_test, y_test)
    print(f'R-squared score of the model: {r2:.4f}')

    # Plot the results
    plot_results(X_train, y_train, X_test, y_test, model)

# Run the main function
if __name__ == "__main__":
    main()
