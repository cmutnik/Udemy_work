# Machine Learning, Data Science and Deep Learning with Python
This is a Udemy class for Machine Learning.  Link to class [here.](https://bah.udemy.com/course/data-science-and-machine-learning-with-python-hands-on)

Course materials can be found [here.](https://sundog-education.com/machine-learning/).  There you can get the zip file, `MLCourse.zip`, that contains images, `*.ipynb` files, etc.

<!--
Instructor's github repo [here.]()

My wiki page for this class.
My work for this class can be found on github, [here.]()
-->

----
## Class Requirements
- anaconda
- pydotplus: `conda install pydotplus`
  - Used to visualize decision trees
- tensorflow: `conda install tensorflow`
  - Used to make deep neural networks

----
## Regression
Steps for linear regression are shown in [`linreg.ipynb`](./notebooks/linreg.ipynb)

Instructor provided steps for polynomial regression are shown in [`PolynomialRegression.ipynb`](./notebooks/PolynomialRegression.ipynb)


### Calculating R^2
Using randomly generated data arrays, x and y:
```py
import numpy as np
# choose degree of polynomial, here we used 4
predicted_values = np.poly1d(np.polyfit(x, y, 4))
```

We can use `scipy.stats`'s linear regression tool to calculate an r^2 value:
```py
from scipy import stats
# retrieve linear regression values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
# calculate r^2 value
r2 = r_value**2
print(r2)
```

You can also calculate an r^2 score by using `sklearn`'s tools:
```py
from sklearn.metrics import r2_score
r2 = r2_score(y, predicted_values(x))

print(r2)
```