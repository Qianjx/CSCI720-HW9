---
puppeteer:
  landscape: false
  format: "A4"
  timeout: 3000 # <= Special config, which means waitFor 3000 ms
  printBackground: true
---

 #  <center> CSCI 720 Big Data Analytics HW09 Results </center> 
---
Student: Guo, Zizhun & Qian, Martin
Submission: Apr/18st/2020
Due Date: Apr/18st/2020 11:59 PM 

---

#### 3. Feature Selection using Cross-Correlation

###### CC in three significant digits:
```py
def CC_analysis(data):
    '''
    Using correlation coefficient to analyze data
    '''
    CC_mat = data.loc[:, data.columns != 'Class'].corrwith(data['Class'])
    CC_mat = CC_mat.apply(lambda x: round(x, 3))
    return CC_mat
```

```py
# CC results
Age             -0.283
Ht              -0.010
TailLn          -0.266
HairLn          -0.095
BangLn           0.203
Reach           -0.092
EarLobes         0.043
TailLessHair    -0.222
TailLessBang    -0.340
ShagFactor      -0.515
TailAndHair     -0.259
TailAndBangs    -0.159
HairAndBangs     0.049
AllLengths      -0.157
ApeFactor       -0.537
HeightLessAge    0.193
dtype: float64
```
Feature1: **ApeFactor**; CC = **-0.537**
Feature2: **BangLn**; CC = **0.203**

###### Projection Vector:
```py
    plt.plot([CA[0],CB[0]],[CA[1],CB[1]],\
        'm-' , label = 'projection vector')
    x = range(2 , 8)
```
###### Decision Boundary:
```py
    plt.plot(x, -(CA[0]-CB[0])/(CA[1]-CB[1]+0.)*(x-(CA[0]+CB[0])/2.)+(CA[1]+CB[1])/2. ,\
         'k-', label='decision boundry')
```


![feature_selection](https://i.imgur.com/p5kWXqI.png)
<center> Image1: Feature Selection Scatter Plot </center>

###### Test the Classifier:
```py
    clf = LinearDiscriminantAnalysis(solver = 'eigen', n_components = 1)
    clf.fit(data.loc[: ,[feature1, feature2]], data['Class'])
    curr_score = clf.score(data.loc[: ,[feature1, feature2]], data['Class'])
    print(curr_score)
```
**Results (Accuracy):** **0.753**

#### 4. Brute Force Search for 2 Best Features
```py
def BFS_analysis(data):
    '''
    Using Brute Force Search to analyze data
    '''
    score = 0
    score_second = 0
    for feature1 in data.columns:
        for feature2 in data.columns:
            if feature1 != feature2 and feature1 != 'Class' and feature2 != 'Class':
                # LDA classifier
                clf = LinearDiscriminantAnalysis(solver = 'eigen', n_components = 1)
                clf.fit(data.loc[: ,[feature1, feature2]], data['Class'])
                curr_score = clf.score(data.loc[: ,[feature1, feature2]], data['Class'])
                if( curr_score > score):
                    score_second = score
                    score = curr_score
                    features = [feature1 , feature2]
                    best_clf = clf
    return features, score, best_clf
```
Feature1: **ShagFactor**
Feature2: **ApeFactor**
**Results (Accuracy):** **0.817**
![CC_initial](https://i.imgur.com/Vxk50dU.png)
<center> Image2: Bruce For Search Best Classifier Scatter Plot </center>

##### Question: What is the best classification accuracy this produces? What is the second best classification accuracy this produces?

**Best: 0.817**
**Second Best: 0.778**

#### 5. Principal Components Analysis

```py
    # print Covariance Matrix
    X_std = feature_generation(data.loc[:, data.columns != 'Class'])
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)
```

###### Covariance Matrix
```py
    pca = PCA(n_components = 3)
    pca.fit(data.loc[:, data.columns != 'Class'])
    print("eigen vectors of PCA\n"+str(pca.components_))
    print("singular values of PCA\n"+str(pca.singular_values_))
```
###### Eigenvectors of PCA:
```py

[[-0.023 -0.58  -0.046 -0.006 -0.004 -0.582 -0.003 -0.041 -0.042 -0.001
  -0.052 -0.051 -0.01  -0.056 -0.002 -0.557]
 [ 0.723  0.19   0.128  0.019 -0.005  0.227  0.003  0.109  0.133  0.023
   0.147  0.123  0.014  0.142  0.037 -0.533]
 [-0.249 -0.146  0.335  0.093  0.077 -0.133  0.002  0.241  0.258  0.017
   0.428  0.411  0.17   0.505  0.012  0.103]]

```
singular values of PCA
```py
[931.941 573.253 457.488]
```
##### What do the coefficients in the eigenvector tell you about which features are important?


#### 6. Projection onto PCA
Since the first two singular values are the greatest among all, project the data onto the first two Principal Components.

![pca](https://i.imgur.com/mTcVZky.png)
<center> Image3: PCA Scatter Plot using first two eigen vectors</center>