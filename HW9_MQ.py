import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import random

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
    return CC_mat

def BFS_analysis(data):
    '''
    Using Brute Force Search to analyze data
    '''
    score = 0
    for feature1 in data.columns:
        for feature2 in data.columns:
            if feature1 != feature2 and feature1 != 'Class' and feature2 != 'Class':
                clf = LinearDiscriminantAnalysis(solver = 'eigen', n_components = 1)
                clf.fit(data.loc[: ,[feature1, feature2]], data['Class'])
                curr_score = clf.score(data.loc[: ,[feature1, feature2]], data['Class'])
                if( curr_score > score):
                    score = curr_score
                    features = [feature1 , feature2]
                    best_clf = clf
    return features, score, best_clf

def PCA_analysis(data):
    '''
    Using PCA to analyze data
    '''
    pca = PCA(n_components = 2)
    pca.fit(data.loc[:, data.columns != 'Class'])
    print(pca.components_)
    print(pca.singular_values_)
    data['PCA_1'] = data.loc[:, data.columns != 'Class'].dot(pca.components_[0])
    data['PCA_2'] = data.drop(['Class','PCA_1'], axis = 1).dot(pca.components_[1])
    
    return data


def plot(data, feature1, feature2):
    '''
    General plotting method
    '''

    plt.xlim(0,10)
    plt.ylim(0,10)

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

    CA = np.mean(data.loc[data['Class'] == 1,[feature1, feature2]])
    CB = np.mean(data.loc[data['Class'] == -1,[feature1, feature2]])
    plt.plot([CA[0],CB[0]],[CA[1],CB[1]],\
        'm-' , label = 'projection vector')
    x = range(2 , 8)
    plt.plot(x, -(CA[0]-CB[0])/(CA[1]-CB[1]+0.)*(x-(CA[0]+CB[0])/2.)+(CA[1]+CB[1])/2. ,\
         'k-', label='decision boundry')
    # formula for Vertical line
    plt.show()

def main():

    # read csv file
    abominable_data = pd.read_csv('Abominable_Data_HW19_v420.csv')
    test_data = pd.read_csv('Abominable_UNCLASSIFIED_Data_HW19_v420.csv')

    data = feature_generation(balancing_data(abominable_data))
    # print("After processing, data.head:\n"+ str(data.head()))

    print(CC_analysis(data))
    # cross correlation analysis part
    features, score, best_clf = BFS_analysis(data)
    plot(data, features[0], features[1])
    test_result= best_clf.predict(feature_generation(balancing_data(test_data[features])))
    # brute force search part

    #new_data = PCA_analysis(balancing_data(abominable_data))

    #plot(new_data, 'PCA_1', 'PCA_2')
  
    # doing some preprocessing on the dataset
    #data = feature_generation(balancing_data(abominable_data))

if __name__ == "__main__":
    main()