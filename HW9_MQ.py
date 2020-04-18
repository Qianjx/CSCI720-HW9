import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import random
import math
from HW_08_Martin_Qian import Gradient_Descent__Fit_Through_a_Line_v100

def balancing_data(data):
    '''
    get a balanced dataset
    '''
    data_assam = data.loc[data['Class'] == 'Assam']
    data_bhuttan = data.loc[data['Class'] == 'Bhuttan']
    size_A = data_assam.shape[0]
    size_B = data_bhuttan.shape[0]
    
    '''
    if ( size_A > size_B ):
        erased_index = random.sample(range(size_A), size_A - size_B)
        data_assam.drop(erased_index)
        print(size_A, size_B)
    '''

    '''
    To make it easier 
    We simply remove last 2 elements
    '''
    data_assam.drop([1295,1296])

    data_assam.loc[:,'Class'] = 1
    data_bhuttan.loc[:,'Class'] = -1

    return data_assam.append(data_bhuttan,ignore_index=True)

def feature_generation(data):
    '''
    generate a few more features
    '''
    data['TailLessHair'] = data['TailLn'] - data['HairLn']
    data['TailLessBang'] = data['TailLn'] - data['BangLn']
    data['ShagFactor'] = data['HairLn'] - data['BangLn']
    data['TailAndHair'] = data['TailLn'] + data['HairLn']
    data['TailAndBangs'] = data['TailLn'] + data['BangLn']
    data['HairAndBangs'] = data['HairLn'] + data['BangLn']
    data['AllLengths'] = data['TailLn'] + data['HairLn'] + data['BangLn']
    data['ApeFactor'] = data['Reach'] - data['Ht']
    data['HeightLessAge'] = data['Ht'] - data['Age']

    return data


def CC_analysis(data):
    '''
    Using correlation coefficient to analyze data
    '''
    CC_mat = data.loc[:, data.columns != 'Class'].corrwith(data['Class'])
    CC_mat = CC_mat.apply(lambda x: round(x, 3))
    return CC_mat

def feature_selection_using_CC(CC_mat, data):
    '''
    using CC to select two features that has the greatest abs and has opposite sign
    using LDA to test Accuracy
    '''
    print("cross correlation matrix:\n" + str(CC_mat))
    # assign the feature by abs sort and choose the greates
    feature1 = CC_mat.abs().sort_values(ascending=False).index[0]
    if CC_mat[feature1] < 0: 
        feature2 = CC_mat.sort_values(ascending=False).index[0]
    else: 
        feature2 = CC_mat.sort_values(ascending=True).index[0]
    print(str(feature1) +' '+ str(feature2) + ' Selected')
    plot(data, feature1, feature2)
    
    # LDA classify the trainning data
    clf = LinearDiscriminantAnalysis(solver = 'eigen', n_components = 1)
    clf.fit(data.loc[: ,[feature1, feature2]], data['Class'])
    curr_score = clf.score(data.loc[: ,[feature1, feature2]], data['Class'])
    print("Accuracy for these features:"+ str(round(curr_score, 3)))

def BFS_analysis(data):
    '''
    Using Brute Force Search to analyze data
    '''
    score = 0
    score_second = 0
    for feature1 in data.columns:
        for feature2 in data.columns:
            # Get all combination of feature pairs.
            if feature1 != feature2 and feature1 != 'Class' and feature2 != 'Class':
                clf = LinearDiscriminantAnalysis(solver = 'eigen', n_components = 1)
                clf.fit(data.loc[: ,[feature1, feature2]], data['Class'])
                curr_score = clf.score(data.loc[: ,[feature1, feature2]], data['Class'])
                if( curr_score > score):
                    score_second = score
                    score = curr_score
                    features = [feature1 , feature2]
                    best_clf = clf
    return features, score, best_clf, score_second

def PCA_analysis(data):
    '''
    Using PCA to analyze data
    '''
    '''
    # print Covariance Matrix
    X_std = feature_generation(data.loc[:, data.columns != 'Class'])
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)
    '''
    pca = PCA(n_components = 3)
    pca.fit(data.loc[:, data.columns != 'Class'])
    print("eigen vectors of PCA\n"+str(pca.components_.round(3)))
    print("singular values of PCA\n"+str(pca.singular_values_.round(3)))
    print('We will use the first 2 eigen vectors')

    # doing projection
    data['PCA_feature1'] = data.loc[:, data.columns != 'Class'].dot(pca.components_[0])
    data['PCA_feature2'] = data.drop(['Class','PCA_feature1'], axis = 1).dot(pca.components_[1])
    
    return data


def plot(data, feature1, feature2):
    '''
    General plotting method
    '''

    plt.xlabel(feature1)
    plt.ylabel(feature2)

    plt.scatter(
        data[feature1],
        data[feature2],
        c=data['Class'],
        cmap='rainbow',
        alpha=0.7,
        edgecolors='b'
    )

    '''
    # This part is for plotting decision boundry
    CA = np.mean(data.loc[data['Class'] == 1,[feature1, feature2]])
    CB = np.mean(data.loc[data['Class'] == -1,[feature1, feature2]])
    print(CA)
    print(CB)
    plt.plot([CA[0],CB[0]],[CA[1],CB[1]],\
        'm-' , label = 'projection vector')
    x = range(2 , 8)
    plt.plot(x, -(CA[0]-CB[0])/(CA[1]-CB[1]+0.)*(x-(CA[0]+CB[0])/2.)+(CA[1]+CB[1])/2. ,\
         'k-', label='decision boundry')
    # formula for Vertical line
    '''

    plt.show()

def main():

    # read csv file
    abominable_data = pd.read_csv('Abominable_Data_HW19_v420.csv')
    test_data = pd.read_csv('Abominable_UNCLASSIFIED_Data_HW19_v420.csv')
    data = feature_generation(balancing_data(abominable_data))
    
    print("1. Cross Correlation part:")
    # feature selection
    feature_selection_using_CC(CC_analysis(data), data)

    print('\n2. LDA part:')
    # cross correlation analysis part
    features, score, best_clf, score_second = BFS_analysis(data)
    print('These features are selected:' + str(features) +\
         '\nHighest and Second Highest Accuracy is '\
         + str(round(score, 3)) + ' ' + str(round(score_second, 3)))
    plot(data, features[0], features[1])
    
    print('\n3. PCA part:')
    # PCA part
    # new data is input data after projection on PCA
    new_data = PCA_analysis(data)
    plot(new_data, 'PCA_feature1', 'PCA_feature2')
    
    #test the classifier on unclassified test data
    test_result= best_clf.predict(feature_generation(test_data)[features])
    pd.DataFrame(test_result).to_csv('HW_09_Classiﬁed_Results.csv')
    print('test result generated')

    print('\n4. Gradient Descent part:')

    # gradient decent
    # initial theta is 62.7 degrees, rho 6.10
    theta, rho =  Gradient_Descent__Fit_Through_a_Line_v100(\
        [data[features[0]],data[features[1]],data['Class']] , 62.7, 11.2 ,4.5)

    A       =  math.cos(math.radians(( theta ) ))
    B       =  math.sin(math.radians(( theta ) ))
    C       =  -rho 

    print('result of gradient descent:\n %s*x + %s*y + %s = 0' %(str(A),str(B),str(C)))
    plt.scatter(
        data[features[0]],
        data[features[1]],
        c=data['Class'],
        cmap='rainbow',
        alpha=0.7,
        edgecolors='b'
    )
    x = [0.,10.]
    plt.plot(x, [(-C-A*x[0])/B, (-C-A*x[1])/B], 'k-' )
    plt.show()


if __name__ == "__main__":
    main()