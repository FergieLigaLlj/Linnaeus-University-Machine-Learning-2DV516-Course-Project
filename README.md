### Attribution of different jupyter notebooks to the corresponding assignments and their learning expectation:

Linear and Polynomial Regression (Lecture 2):
1. Classes Implementations: **RegressionModelNormalEquation** and **RegressionModelGradientDescent** classes in abstract class **MachineLearningModel.py**
2. Use implementations in Multivariate Regression Model (d=1): effect of normalization in **NormalEquation** and **GradientDescent** is evaluated by **predicted beta** and **cost function versus iterations**
3. Use implementations in Multivariate Regression Model (d>1): randomly divide the training and test set several times and fit the polynomial from 1 to 6, motivating the best **degree optimization**.

Logistic Regression (Lecture 3):
1. Classes Implementations: **LogisticRegressionModel** and **NonLinearLogisticRegressionModel** classes in abstract class **MachineLearningModel.py**
2. Use Implementations for the **LogisticRegressionModel** and the **NonLinearLogisticRegressionModel**: optimal **learning rate** and **number of iterations**, **cost function J(β)** as a function over **iterations**, **model comparison**, **DecisionBoundary.py** class for plot.

 Model Selection and Regularization (Lecture 4):
1. Classes Implementations: **ROCAnalysis.py** and **ForwardSelection.py**: optimal regarding **TP-rate** metric
2. Use implementations of ROCAnalysis and ForwardSelection: model selection with **f-score**, **validation** confirmation and conclusion
