from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


def plotDecisionBoundary(X1, X2, y, model,X3,X4):
    """
    Plots the decision boundary for a binary classification model along with the training data points.

    Parameters:
        X1 (array-like): Feature values for the first feature.
        X2 (array-like): Feature values for the second feature.
        y (array-like): Target labels.
        model (object): Trained binary classification model with a `predict` method.

    Returns:
        None
    """
    #--- Write your code here ---#
    model = model
    h = .01 # step size in the mesh
    x_min,x_max = X1.min()-0.1,X1.max()+0.1
    y_min,y_max = X2.min()-0.2,X2.max()+0.2
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h)) # Mesh Grid
    x1,x2 = xx.ravel(),yy.ravel() # Turn to two Nx1 arrays
    XXe = model.mapFeature(x1,x2,2) # Extend matrix for degree 2
    p = model._sigmoid(np.dot(XXe,model.Beta))  # classify mesh ==> probabilities
    classes = p>0.5 # round off probabilities
    clz_mesh = classes.reshape(xx.shape)  # return to mesh format
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"]) # mesh plot
    cmap_bold = ListedColormap(["#FF0000", "#00FF00"]) # colors
    plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
    plt.scatter(X3,X4,c=y,marker="1",cmap = cmap_bold)


    
    
    