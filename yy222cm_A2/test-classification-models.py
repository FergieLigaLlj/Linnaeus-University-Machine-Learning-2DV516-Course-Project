import numpy as np
from MachineLearningModel import LogisticRegression, NonLinearLogisticRegression

def test_fit_predict():
    # Initialize parameters
    degree = 2
    X_train = np.array([[1,2], [2, 3], [3,4]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[1,3], [2,5]])

    # Instantiate the model
    lr = 0.02
    itrs = 2000
    modelLR = LogisticRegression()
    modelNLR = NonLinearLogisticRegression(degree, learning_rate=lr, num_iterations=itrs)

    # Train the model
    modelLR.fit(X_train, y_train)
    modelNLR.fit(X_train, y_train)
    
    # Make predictions
    y_pred_lr = modelLR.predict(X_test)
    y_pred_nlr = modelNLR.predict(X_test)

    # Check if predictions have correct length
    assert len(y_pred_lr) == len(X_test), "Incorrect number of predictions for Logistic Regression"
    assert len(y_pred_nlr) == len(X_test), "Incorrect number of predictions for Non-Linear Logistic Regression"
    # Define the expected array
    expected_array_lr = np.array([-0.15065123, -0.01045665, -0.16110788])
    expected_array_nlr = np.array([-1.08781537,  0.9897509,  -0.09806447, -0.7823303,   0.20742059,  0.10935612])
    
    # Compare your variable with the expected array
    #### in my implementation I saved the betas in a variable called theta ####
    #### change here if you used a variable with a different name, such as beta or betas ####

    # Print the actual values of both arrays
    print("Values of expected_array for Logistic Regression:", expected_array_lr)
    print("Values of your_variable for Logistic Regression:", modelLR.Beta)
    print("Values of expected_array for Non Linear Logistic Regression:", expected_array_nlr)
    print("Values of your_variable for Non Linear Logistic Regression:", modelNLR.Beta)    
    
    # Define a tolerance for floating-point comparison
    tolerance = 0.1  # Adjust tolerance as needed

    # Perform a tolerance-based comparison
    comparison_lr = np.abs(modelLR.Beta - expected_array_lr) < tolerance
    comparison_nlr = np.abs(modelNLR.Beta - expected_array_nlr) < tolerance


    # Check if all elements are within the tolerance for fit and predict
    assert np.all(comparison_lr), "Your variable is not equal to the expected array within the tolerance for the Logistic Regression Implementation"
    assert np.all(comparison_nlr), "Your variable is not equal to the expected array within the tolerance for the Non Linear Logistic Regression Implementation"


def test_evaluate():
    # Initialize parameters
    degree = 2
    X_train = np.array([[1,2], [2, 3], [3,4]])
    y_train = np.array([0, 1, 0])    

    # Instantiate the model
    lr = 0.02
    itrs = 2000
    modelLR = LogisticRegression()
    modelNLR = NonLinearLogisticRegression(degree, learning_rate=lr, num_iterations=itrs)

    # Train the model
    modelLR.fit(X_train, y_train)
    modelNLR.fit(X_train, y_train)        
    
    acc_lr = modelLR.evaluate(X_train, y_train)
    cost_nlr = modelNLR.evaluate(X_train, y_train)        

    expected_acc = 0.6666666666
    expected_cost = 0.5605
    tolerance = 0.1  # Adjust tolerance as needed
    # Check if MSE is a float
    assert isinstance(acc_lr, float), "ACC is not a float"
    assert isinstance(cost_nlr, float), "Cost is not a float"
    # Perform a tolerance-based comparison
    if abs(acc_lr - expected_acc) < tolerance:
        print("Your Accuracy is within the tolerance for the Logistic Regression.")
    else:
        print("Your Accuracy is not within the tolerance for the Normal Equation.")
    if abs(cost_nlr - expected_cost) < tolerance:
        print("Your Cost is within the tolerance for the Non Linear Logistic Regression.")
    else:
        print("Your Cost is not within the tolerance for the Non Linear Logistic Regression.")

if __name__ == "__main__":
    # Run tests
    test_fit_predict()
    test_evaluate()

