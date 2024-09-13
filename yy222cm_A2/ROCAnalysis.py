class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        #--- Write your code here ---#
        self.y_predicted = y_predicted
        self.y_true = y_true

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        #--- Write your code here ---#
        tp = sum((self.y_predicted+self.y_true)==2)
        fn = sum((self.y_predicted-self.y_true)==-1)
        tp_rate = tp/(tp+fn)
        return tp_rate


    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        #--- Write your code here ---#
        fp = sum((self.y_predicted-self.y_true)==1)
        tn = sum((self.y_predicted+self.y_true)==0)
        fp_rate = fp/(fp+tn)
        return fp_rate


    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        #--- Write your code here ---#
        tp = sum((self.y_predicted+self.y_true)==2)
        fp = sum((self.y_predicted-self.y_true)==1)
        precision = tp/(tp+fp)
        return precision
  
    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        #--- Write your code here ---#
        tp = sum((self.y_predicted+self.y_true)==2)
        fp = sum((self.y_predicted-self.y_true)==1)
        fn = sum((self.y_predicted-self.y_true)==-1)
        precision = tp/(tp+fp)
        tp_rate = tp/(tp+fn)
        f_score = 2*((precision*tp_rate)/(precision+tp_rate))
        return f_score
