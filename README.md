# Traditional Transfer Learning

This is an demo for Traditional transfer learning techniques.

1. Adaboost. Ref. Boosting for Transfer Learning, ICML 2007.
  
2. Regression Tradaboost. 
  In the classification case, the weights are multiplied with coef.^(0 or 1). 
  While in the regression Tradaboost, the abs error is used as the power term. 
  
3. Correlation alignment. (CORAL)
  The assumption is in different tasks, the correlation between features should be similar. e.g. A hat should always have high correlation to the head. 
  (Does that means CORAL is more efficient in top layers?)
  The correlation descrepancy between mapped source data and the target data is minimized.
  
