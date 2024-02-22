[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="1100" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **QReg_engel_curve** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'QReg_engel_curve'

Published in: 'QReg_engel_curve'

Description: 'Household expenditure on food versus annual income. Different estimated slope of Engel curve based on Quantile Regression.'

Submitted: '21 Feb 2024'

Keywords: 
	'-Quantile Regression 
	 -Food Expenditure
	 -Income
	 -Estimation 
	 -Engel Curve'

Author: 'Alexandra Conda'
```

### PYTHON Code
```python

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#add the csv file in the same folder as .py file
engel = pd.read_csv('engel.csv')
print(engel)

engel['income'] = np.log10(engel['income'])
engel['foodexp'] = np.log10(engel['foodexp'])

X = np.column_stack([
  np.ones((engel.shape[0],1)),
  np.asarray(engel['income'])
  ])
  
y = np.asarray(engel['foodexp'])


def quantile_reg(par, X, y, tau):
  lp = np.dot(X, par)
  res = y - lp
  loss = np.where(res < 0 , -(1 - tau)*res, tau*res)
  
  return(np.sum(loss))


qs = [.15, .25, .5, .75, .95, .99]
fit = []

for tau in qs:
  init = minimize(
    fun = quantile_reg,
    x0  = [0, 0],
    args    = (X, y, tau),
    method  = 'BFGS', 
    tol     = 1e-12, 
    options = {'maxiter': 500}
  )
  
  fit.append(
    pd.DataFrame(init.x.reshape(1,2),  columns = ['int', 'slope'])
  )
    
result = pd.concat(fit)

print(result)


plt.figure(5) 
plt.scatter(engel['income'], engel['foodexp'],  color='black') 
y_pred0 = result['int'].iloc[0] + result['slope'].iloc[0] * engel['income']
plt.plot(engel['income'], y_pred0, color='r',   linewidth=1, label='tau = 0.15')  
y_pred1 = result['int'].iloc[1] + result['slope'].iloc[1] * engel['income']
plt.plot(engel['income'], y_pred1, color='g',   linewidth=1, label='tau = 0.25') 
y_pred2 = result['int'].iloc[2] + result['slope'].iloc[2] * engel['income']
plt.plot(engel['income'], y_pred2, color='b',   linewidth=1, label='tau = 0.5') 
y_pred3 = result['int'].iloc[3] + result['slope'].iloc[3] * engel['income']
plt.plot(engel['income'], y_pred3, color='c',   linewidth=1, label='tau = 0.75') 
y_pred4 = result['int'].iloc[4] + result['slope'].iloc[4] * engel['income']
plt.plot(engel['income'], y_pred4, color='tab:pink',   linewidth=1, label='tau = 0.95') 
y_pred5 = result['int'].iloc[5] + result['slope'].iloc[5] * engel['income']
plt.plot(engel['income'], y_pred5, color='yellow',   linewidth=1, label='tau = 0.99') 
plt.xticks(()) 
plt.yticks(()) 
plt.xlabel("log10(income)") 
plt.ylabel("log10(foodexp)") 
plt.savefig("graph.pdf", format = "pdf")
plt.show()






















```

automatically created on 2024-02-22