# Machine Learning, Data Science and Deep Learning with Python
This is a Udemy class for Machine Learning.  Link to class [here.](https://bah.udemy.com/course/data-science-and-machine-learning-with-python-hands-on)

Course materials can be found [here.](https://sundog-education.com/machine-learning/).  There you can get the zip file, `MLCourse.zip`, that contains images, `*.ipynb` files, etc.

This [`README`](./README.md) will serve as my wiki page, for this class.
<!--  My work for this class will be contained in this github repo.-->

----
## Class Requirements
- anaconda
- pydotplus: `conda install pydotplus`
  - Used to visualize decision trees
- tensorflow: `conda install tensorflow`
  - Used to make deep neural networks

----
## Regression
Steps for linear regression are shown in [`linreg.ipynb`](./notebooks/made/linreg.ipynb)

Instructor provided steps for polynomial regression are shown in [`PolynomialRegression.ipynb`](./notebooks/provided/PolynomialRegression.ipynb)


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

----
## Naive Bayes Classifier
Instructor provided steps for making a Naive Bays (NB) Classifier are shown in [`NaiveBayes.ipynb`.](./notebooks/provided/NaiveBayes.ipynb) 
In this example, we are given two sets of email:
1. - emails known to be `spam`
2. - emails known to be `ham` (not spam)
The data sets are provided in the course materials.  Spam emails are located in `MLCourse/emails/spam`.  Similarly, non-spam emails be found in `MLCourse/emails/spam`.<br>
Most of the code is used to merely clean and separate the data.  The actual NB code works by counting all identifying words and using Bayes Theorem to assign weights to each word.  This is then used to determine the overall probability of an email (or any body of text) being spam, and subsequently classifying each email.

First, we import the needed packages:
```py
#import os
#import io
#import numpy
#from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
```

Next, we use `sklearn`'s `CountVectorizer` feature, to count the number of times a particular word occurs in an email; and apply it to out data.
```py
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
```

This is followed by applying `sklearn.naive_bayes.MultinomialNB`, to classify out transformed data.
```py
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)
```

Finally, we test our classifier on new data.
```py
# make test data, to test classifier
examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]

# transform test data to form of original data
example_counts = vectorizer.transform(examples)

# apply classifier to transformed data
predictions = classifier.predict(example_counts)

# check results
predictions
#>> array(['spam', 'ham'], dtype='<U4')
```
As we can see, our NB classifier properly categorized both of our test messages.



<!--
----
## ___
Instructor provided steps for ___ are shown in [`___`](./notebooks/provided/___.ipynb)
```py

```

----
## ___
Instructor provided steps for making a ___ are shown in [`___`](./notebooks/provided/___.ipynb)
```py

```

----
## 
Instructor provided steps for making a 
 are shown in [``](./notebooks/provided/.ipynb)
```py

```

----
## 
Instructor provided steps for 
 are shown in [``](./notebooks/provided/.ipynb)
```py

```
-->