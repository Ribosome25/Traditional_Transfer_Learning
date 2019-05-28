# Traditional Transfer Learning

  This is an demo for Traditional transfer learning techniques.

## 1. TrAdaboost. 
Ref. *Boosting for Transfer Learning,* ICML 2007.<br>

## 2. Regression Tradaboost. 

  In the classification case, the weights are multiplied with `coef.^(0 or 1).` <br>
  While in the regression Tradaboost, the abs error is used as the power term. 

## 3. Instance weighting kernel ridge regression
Instance-weighted kernel ridge regression<br>

Ref: *Jochen Garcke, Importance Weighted Inductive Transfer Learning for Regression*<br>

In this scenario, the source domain data are all labeled, and a small portion from the target domain is labeled too. Here we call this part “Auxiliary data”. The rest of the target domain data is unlabeled and called “Test data”.<br>
In this method, the weights (alphas) of source instances are calculated based on [source data + auxiliary data] and applied on the source instances. <br>
The source set has n instances and auxiliary set has m. <br>
The method contains 3 steps:<br>
1)	A kernel ridge regression (rbf kernel) model is trained and test on the Source data. The dual ecoefficiency a (n*1) is obtained. <br>
2)	This a is used to calculate the weights alphas. Instead of scalars applied on each of the instances, here the author uses a form of rbf distances: <br>
 ![alt text](https://github.com/Ribosome25/Traditional_Transfer_Learning/blob/master/imgs/kRR_fig1.png)<br>
And *alpha* is the variable, instead of w(x,y)<br>
And the cost function is weighted error with a regulating term on *alpha*. <br>
 ![alt text](https://github.com/Ribosome25/Traditional_Transfer_Learning/blob/master/imgs/kRR_fig2.png)<br>
*Alpha* is supposed to be >0, therefore a library of convex optimize, a function like quadprog in Matlab is used here. <br>
In this step, the sample weights of the source data are obtained from [X_source, X_auxiliary, Y_source, and Y_auxiliary]. <br>
3)	Then a weighted kernel ridge regression is performed. The weights of Source data are normalized to \[0,1], and the weights for Auxiliary data are all 1. Test data are predicted using this model.  <br>
This model has 5 hyper parameters: the sigma for the rbf kernel; the lambda in the 1st ridge regression; the eta in the weight-alpha relationship; the gamma in the 2nd step for regularizing the alpha; and the lambda in the 3rd step.<br>

## 4. Correlation alignment. (CORAL)

  The assumption is in different tasks, the correlation between features should be similar. e.g. A hat should always have high correlation to the head. <br>
  (Does that means CORAL is more efficient in top layers?)<br>
  The correlation descrepancy between mapped source data and the target data is minimized.
  
