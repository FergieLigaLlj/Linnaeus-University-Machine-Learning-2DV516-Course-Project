import numpy as np
from ROCAnalysis import ROCAnalysis

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
        """
        #--- Write your code here ---#
        self.X=np.array(X)
        self.y=np.array(y)
        self.model=model

    def create_split(self, X, y):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        #--- Write your code here ---#
        c = list(zip(X, y))
        np.random.shuffle(c)
        X, y = zip(*c)
        X_train = X[:int(len(X)*0.8)]
        X_test = X[int(len(X)*0.8):]
        y_train = y[:int(len(X)*0.8)]
        y_test = y[int(len(X)*0.8):]
        X_train=np.array(X_train)
        X_test=np.array(X_test)
        y_train=np.array(y_train)
        y_test=np.array(y_test)
        return X_train,X_test,y_train,y_test
    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        #--- Write your code here ---#
        Xe=self.X[:,features]
        model = self.model
        X_train,X_test,y_train,y_test=self.create_split(Xe,self.y)
        model.fit(X_train,y_train)
        cost = model._cost_function(X_train,y_train)
        self.cost=cost
        y_predict = np.round(model.predict(X_test))
        ROC_model=ROCAnalysis(y_predict,y_test)
        f_score = ROC_model.f_score(1)
        return f_score
    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        #--- Write your code here ---#
        model = list([[0]])
        model_matrix=list([0])
        all_indice=list(np.arange(1,np.shape(self.X)[1],1))
        f_score=self.train_model_with_features(model_matrix)
        f_score=np.nan_to_num(f_score)
        print("The selected feature matrix is",model_matrix,",which has 1 feature. And its f_score is:",f_score)
        f_score_matrix=[]
        f_score_matrix.append(f_score)
        for k in range(1,np.shape(self.X)[1]-1,1):
            index_cost=[]
            for a in all_indice:
                model_matrix=list(model_matrix)
                model_matrix.append(int(a))
                model_matrix=np.array(model_matrix)
                self.train_model_with_features(model_matrix)
                cost=self.cost
                index_cost.append((a,cost))
                model_matrix=list(model_matrix)
                model_matrix.remove(int(a))
            index_cost=np.array(index_cost)
            index=np.argmin(index_cost[1])
            if int(index_cost[index][0])<=12:
                model_matrix.append(int(index_cost[index][0]))
                model.append(model_matrix)
                model_matrix=np.array(model_matrix)
                f_score=self.train_model_with_features(model_matrix)
                f_score=np.nan_to_num(f_score)
                print("The selected feature matrix is",model_matrix,",which has",k+1,"features. And its f_score is:",f_score)
                f_score_matrix.append(f_score)
                all_indice.remove(index_cost[index][0])
        f_score_matrix=np.array(f_score_matrix)
        Index=np.argmax(f_score_matrix)
        model_choose=model[Index]
        max_f_score=f_score_matrix[Index]
        self.model_to_choose=np.array(model_choose)
        self.max_f_score=max_f_score

    def fit(self):
        """
        Fits the model using the selected features.
        """
        #--- Write your code here ---#
        Xe=self.X[:,self.model_to_choose]
        X_Train,X_Test,y_Train,y_Test=self.create_split(Xe,self.y)
        model = self.model
        model.fit(X_Train,y_Train)
        beta = model.Beta
        self.beta=beta
    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        #--- Write your code here ---#
        model=self.model
        Xe=X_test[:,self.model_to_choose]
        y_predict=model.predict(Xe)
        y_predict=np.round(y_predict)
        return y_predict
