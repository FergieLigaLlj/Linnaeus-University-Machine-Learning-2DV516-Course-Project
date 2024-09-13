from abc import ABC, abstractmethod
import numpy as np

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.
        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.
        Returns:
        None
        """
        pass
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass
    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass
def _polynomial_features(self, X):
    """
        Generate polynomial features from the input features.
        Check the slides for hints on how to implement this one. 
        This method is used by the regression models and must work
        for any degree polynomial
        Parameters:
        X (array-like): Features of the data.
        Returns:
        X_poly (array-like): Polynomial features.
    """
    if self.degree > 1:
        X_poly = np.c_[np.ones((np.shape(X)[0],1)),X]
        for i in range(2,self.degree+1):
            X_poly = np.column_stack((X_poly,X**i))
        return X_poly
    elif self.degree == 1:
        return np.c_[np.ones((np.shape(X)[0],1)),X]
class RegressionModelNormalEquation(MachineLearningModel):
    """
    Class for regression models using the Normal Equation for polynomial regression.
    """
    def __init__(self, degree):
        """
        Initialize the model with the specified polynomial degree.

        Parameters:
        degree (int): Degree of the polynomial features.
        """
        #--- Write your code here ---#
        self.degree = degree
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        Xe = _polynomial_features(self,X)
        beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
        self.beta = beta
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        Xe = _polynomial_features(self,X)
        y_predict = np.array(np.dot(Xe,self.beta))
        return y_predict
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        Xe = _polynomial_features(self,X)
        j = np.array(np.dot(Xe,self.beta)-y)
        MSE = (j.T.dot(j))/(np.shape(Xe)[0])
        return MSE
class RegressionModelGradientDescent(MachineLearningModel):
    """
    Class for regression models using gradient descent optimization.
    """

    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the model with the specified parameters.

        Parameters:
        degree (int): Degree of the polynomial features.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        Xe = _polynomial_features(self,X)
        num_of_feature = int(np.shape(Xe)[1])
        beta = np.array([0]*num_of_feature)
        cost_J = []
        n = int(np.shape(Xe)[0])
        alpha = (2*self.learning_rate)/n
        for i in range(self.num_iterations):
            beta = beta-alpha*Xe.T.dot((Xe.dot(beta)-y))
            j = (np.dot(Xe,beta)-y)
            J = (j.T.dot(j))/n
            cost_J.append(J)
        beta = np.array(beta)
        cost_J = np.array(cost_J)
        self.beta = beta
        self.cost_J = cost_J
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
        Xe = _polynomial_features(self,X)
        y_predict = np.array(np.dot(Xe,self.beta))
        return y_predict

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (MSE).
        """
        #--- Write your code here ---#
        Xe = _polynomial_features(self,X)
        j = np.array(np.dot(Xe,self.beta)-y)
        MSE = (j.T.dot(j))/(np.shape(Xe)[0])
        return MSE
class LogisticRegression:
    """
    Logistic Regression model using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        epsilon = 0.000000000000001
        Xe=np.c_[np.ones((np.shape(X)[0],1)),X]
        num_of_feature = int(np.shape(Xe)[1])
        Beta = np.array([0]*num_of_feature)
        Cost_J = []
        n=int(np.shape(Xe)[0])
        alpha = self.learning_rate/n
        for i in range(self.num_iterations):
            Beta = Beta-alpha*Xe.T.dot(self._sigmoid(Xe.dot(Beta))-y)
            J = ((-1)/n)*(y.T.dot(np.log(self._sigmoid(Xe.dot(Beta))+epsilon))+(1-y).T.dot(np.log(1-self._sigmoid(Xe.dot(Beta))+epsilon)))
            Cost_J.append(J)
        Beta = np.array(Beta)
        Cost_J=np.array(Cost_J)
        self.Beta = Beta
        self.Cost_J = Cost_J
    def predict(self, X):
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        Xe=np.c_[np.ones((np.shape(X)[0],1)),X]
        y_predict = self._sigmoid(Xe.dot(self.Beta))
        return y_predict


    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        #--- Write your code here ---#
        Xe=np.c_[np.ones((np.shape(X)[0],1)),X]
        _y_predict = self._sigmoid(Xe.dot(self.Beta))
        _y_predict = np.round(_y_predict)
        num_correct = 0
        for i in range(len(y)):
            if _y_predict[i]==y[i]:
                num_correct = num_correct+1
        accuracy = num_correct / len(y)
        return accuracy



    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        g_z = (1+np.exp(-z))**(-1)
        return g_z
    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        epsilon = 0.000000000000001
        Xe=np.c_[np.ones((np.shape(X)[0],1)),X]
        n = int(np.shape(Xe)[0])
        J = ((-1)/n)*(y.T.dot(np.log(self._sigmoid(Xe.dot(self.Beta))+epsilon))+(1-y).T.dot(np.log(1-self._sigmoid(Xe.dot(self.Beta))+epsilon)))
        return J


    
class NonLinearLogisticRegression:
    """
    Nonlinear Logistic Regression model using gradient descent optimization.
    It works for 2 features (when creating the variable interactions)
    """

    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        epsilon = 0.000000000000001
        X1 = X[:,0]
        X2 = X[:,1]
        Xe = self.mapFeature(X1,X2,self.degree)
        num_of_feature = int(np.shape(Xe)[1])
        Beta = np.array([0]*num_of_feature)
        Cost_J = []
        n = int(np.shape(Xe)[0])
        alpha = self.learning_rate/n
        for i in range(self.num_iterations):
            Beta = Beta-alpha*Xe.T.dot(self._sigmoid(Xe.dot(Beta))-y)
            J = ((-1)/n)*(y.T.dot(np.log(self._sigmoid(Xe.dot(Beta))+epsilon))+(1-y).T.dot(np.log(1-self._sigmoid(Xe.dot(Beta))+epsilon)))
            Cost_J.append(J)
        Beta = np.array(Beta)
        Cost_J=np.array(Cost_J)
        self.Beta = Beta
        self.Cost_J = Cost_J


    def predict(self, X):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        X1 = X[:,0]
        X2 = X[:,1]
        Xe = self.mapFeature(X1,X2,self.degree)
        y_predict = self._sigmoid(Xe.dot(self.Beta))
        return y_predict
    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        X1 = X[:,0]
        X2 = X[:,1]
        Xe = self.mapFeature(X1,X2,self.degree)
        _y_predict = self._sigmoid(Xe.dot(self.Beta))
        _y_predict = np.round(_y_predict)
        num_correct = 0
        for i in range(len(y)):
            if _y_predict[i]==y[i]:
                num_correct = num_correct+1
        accuracy = num_correct / len(y)
        return accuracy
    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        g_z = (1+np.exp(-z))**(-1)
        return g_z

    def mapFeature(self, X1, X2, D):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        #--- Write your code here ---#
        one = np.ones([len(X1),1])
        Xe = np.c_[one,X1,X2]
        for i in range(2,D+1):
            for j in range(0,i+1):
                Xnew = X1**(i-j)*X2**j
                Xnew = Xnew.reshape(-1,1)
                Xe = np.append(Xe,Xnew,1)
        return Xe

    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        epsilon = 0.000000000000001
        X1 = X[:,0]
        X2 = X[:,1]
        Xe = self.mapFeature(X1,X2,self.degree)
        n = int(np.shape(Xe)[0])
        J = ((-1)/n)*(y.T.dot(np.log(self._sigmoid(Xe.dot(self.Beta))+epsilon))+(1-y).T.dot(np.log(1-self._sigmoid(Xe.dot(self.Beta))+epsilon)))
        return J