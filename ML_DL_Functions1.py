import numpy as np

def LeastSquares(X,y):

  #X_with_intercept = np.c_[np.ones(X.shape[0]), X]
  Xt = X.T
  theta = np.linalg.inv(Xt @ X) @ Xt @ y

  return theta

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: numpy input matrix, size [N,m]
    :param s: numpy input vector of ground truth labels, size [N]
    :return: accuracy of the model = (correct classifications)/(total classifications) type float
    N is the number of samples and m is the number of features=28
  '''
  predictions = model.predict(X)
    
  correct = (predictions == s).sum()
    
  accuracy = correct / len(s)
    
  return accuracy
  

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem. length 28
  '''
  return [[-3.93996222e-02,5.93035721e-02,-4.64445152e-02,2.34440489e-02,
 -4.83178153e-02,-1.35336929e-02,8.05724680e-02,-1.82937646e-02,
 -5.84449866e-04,-4.71317563e-02,5.92245375e-02 ,7.92348225e-03,
  6.59969392e-02, 1.74013552e-01 , 7.53972040e-01 , 1.57419350e-02,
  1.83970685e-02 , 2.10923918e-03  ,1.48961853e-02, -3.43862391e-03,
  3.88713559e-02  ,3.14717497e-02 , 6.08192259e-03 ,-3.12351287e-02,
 -1.76478363e-02 ,-8.94175418e-03, -9.17396165e-03 ,-3.04002576e-02]]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value. type float
  '''
  return -1.392735894391803e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of list of coefficiants for the classification problem.  length 28
  '''
  return [[-2.34677176e-01,-1.22039408e-01,2.53873203e-01,-3.43023416e-02,
  -3.71785922e-01 ,-5.74515263e-01  ,1.64294613e-02  ,1.63875633e-02,
   4.94028213e-02  ,1.31275838e-01 ,-5.37540172e-01  ,3.17557480e-02,
  -1.86789091e-01  ,1.15474475e+00  ,2.97220668e+00 ,-5.90707220e-01,
  -5.82430811e-02 ,-1.25763231e-03 ,-3.51843404e-01 ,-1.26430585e-01,
   2.04034493e-01 ,-1.16642459e-01  ,9.12551818e-03 ,-2.81492039e-01,
  -3.99560706e-01  ,8.22814643e-02  ,1.38067233e-01  ,3.40590514e-01]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: list with the intercept value. length 1
  '''
  return [0.23635986]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem. length 2.
  '''
  return [0, 1]
