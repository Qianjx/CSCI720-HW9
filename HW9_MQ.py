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
    clf = LinearDiscriminantAnalysis(solver = 'eigen', n_components = 1)
    clf.fit(data.loc[: ,['Ht', 'HairLn']], data['Class'])
    print(clf.explained_variance_ratio_)

def PCA_analysis(data):
    '''
    Using PCA to analyze data
    '''
    pca = PCA(n_components = 2)
    pca.fit(data.drop('Class' , axis = 1))
    print(pca.explained_variance_ratio_)


def plot(data, feature1, feature2):
    '''

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
    plt.show()

def main():

    # read csv file
    abominable_data = pd.read_csv('Abominable_Data_HW19_v420.csv')

    data = feature_generation(balancing_data(abominable_data))
    # print("After processing, data.head:\n"+ str(data.head()))

    print(CC_analysis(data))

    # BFS_analysis(data)
    PCA_analysis(data)

    
    # plot(data, 'ShagFactor', 'ApeFactor')

    
    # doing some preprocessing on the dataset
    #data = feature_generation(balancing_data(abominable_data))


    

if __name__ == "__main__":
    main()