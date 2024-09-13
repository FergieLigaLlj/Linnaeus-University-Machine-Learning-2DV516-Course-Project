import numpy as np
from MachineLearningModel import _polynomial_features, RegressionModelNormalEquation, RegressionModelGradientDescent

def test_polynomial_features():
    # Initialize parameters
    degree = 2
    X_train = np.array([[1], [2], [3]])
    # Instantiate the model
    model = RegressionModelNormalEquation(degree)

    # Generate polynomial features
    X_poly = _polynomial_features(model, X_train)
    
    # Check if the polynomial features have the correct shape
    assert X_poly.shape[1] == degree + 1, "Incorrect shape for polynomial features"
    X_poly_expected = np.array([[1., 1., 1.], [1., 2., 4.], [1., 3., 9.]])
    assert np.array_equal(X_poly, X_poly_expected), "Polynomial features are not as expected"

def test_fit_predict():
    # Initialize parameters
    degree = 2
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([2, 4, 9])
    X_test = np.array([[4], [5]])

    # Instantiate the model
    model = RegressionModelNormalEquation(degree)
    modelGD = RegressionModelGradientDescent(degree, learning_rate=0.02, num_iterations=15000)

    # Train the model
    model.fit(X_train, y_train)
    modelGD.fit(X_train, y_train)
    
    # Make predictions
    y_pred_ne = model.predict(X_test)
    y_pred_gd = modelGD.predict(X_test)

    # Check if predictions have correct length
    assert len(y_pred_ne) == len(X_test), "Incorrect number of predictions for Normal Equation Implementation"
    assert len(y_pred_gd) == len(X_test), "Incorrect number of predictions for Gradient Descent Implementation"
    # Define the expected array
    expected_array = np.array([3., -2.5, 1.5])
    
    # Compare your variable with the expected array
    #### in my implementation I saved the betas in a variable called theta ####
    #### change here if you used a variable with a different name, such as beta or betas ####

    # Print the actual values of both arrays
    print("Values of expected_array:", expected_array)
    print("Values of your_variable for Normal Equation:", model.beta)
    print("Values of your_variable for Gradient Descent:", modelGD.beta)    
    
    # Define a tolerance for floating-point comparison
    tolerance = 0.1  # Adjust tolerance as needed

    # Perform a tolerance-based comparison
    comparison_ne = np.abs(model.beta - expected_array) < tolerance
    comparison_gd = np.abs(modelGD.beta - expected_array) < tolerance


    # Check if all elements are within the tolerance for fit and predict
    assert np.all(comparison_ne), "Your variable is not equal to the expected array within the tolerance for the Normal Equation Implementation"
    assert np.all(comparison_gd), "Your variable is not equal to the expected array within the tolerance for the Gradient Descent Implementation"


def test_evaluate():
    # Initialize parameters
    degree = 2
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([2, 4, 9])

    # Instantiate the models
    model = RegressionModelNormalEquation(degree)
    modelGD = RegressionModelGradientDescent(degree, learning_rate=0.02, num_iterations=15000)

    # Train the model
    model.fit(X_train, y_train)
    modelGD.fit(X_train, y_train)        
    
    mse_ne = model.evaluate(X_train, y_train)
    mse_gd = modelGD.evaluate(X_train, y_train)    

    expected_mse = 9.035941500445311e-27
    tolerance = 0.1  # Adjust tolerance as needed
    # Check if MSE is a float
    assert isinstance(mse_ne, float), "MSE is not a float"
    # Perform a tolerance-based comparison
    if abs(mse_ne - expected_mse) < tolerance:
        print("Your MSE is within the tolerance for the Normal Equation.")
    else:
        print("Your MSE is not within the tolerance for the Normal Equation.")
    if abs(mse_gd - expected_mse) < tolerance:
        print("Your MSE is within the tolerance for the Gradient Descent.")
    else:
        print("Your MSE is not within the tolerance for the Gradient Descent.")

if __name__ == "__main__":
    # Run tests
    test_polynomial_features()
    test_fit_predict()
    test_evaluate()

