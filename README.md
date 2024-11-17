
# Simple Linear Regression Model

This project implements a basic linear regression model using synthetic data. It demonstrates the steps of generating data, splitting it into training and testing sets, training a model, evaluating its performance, and visualizing the results.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the required dependencies using `pip`:

pip install numpy matplotlib scikit-learn


## Functions
generate_data(size=100, noise_scale=0.15): Generates synthetic data with a linear relationship and some added noise.
split_data(x, y, train_size=0.8): Splits the dataset into training and testing sets.
train_model(X_train, y_train): Trains a simple linear regression model using the training data.
evaluate_model(model, X_test, y_test): Evaluates the model's performance using the R-squared score.
plot_results(X_train, y_train, X_test, y_test, model): Plots the training data, testing data, and the regression line.


## Usage
Run the script: Simply execute the Python script, and it will generate synthetic data, train a linear regression model, evaluate its performance, and display a plot.

## Usage
Run the script: Simply execute the Python script, and it will generate synthetic data, train a linear regression model, evaluate its performance, and display a plot.

## Output:
The R-squared score of the trained model will be printed to the console.
A plot will display the training data, testing data, and the regression line.

## Example Output

R-squared score of the model: 0.9786


# License

### Explanation:

- **Purpose**: The `README.md` gives a concise summary of the project and the steps involved in using it.
- **Functions**: I’ve briefly described the main functions in your script to help users understand the workflow.
- **Usage**: A short section explaining how to run the script and what to expect as output.

Let me know if you need any further adjustments!
