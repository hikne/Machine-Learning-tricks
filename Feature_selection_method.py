#############################
###       libraries       ###
#############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random
from IPython.display import display
## ML libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


##################################
##          functions           ##
##################################

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
    

def feature_selection(data,features,cv=7,max_features=60,nb_iter=10**3,score='accuracy',score_pen_tresh=.6,
                     clf=None, clf_params={'max_depth':12,'min_samples_split':3,'random_state':42}):
    """
    Operate a classifier based feature selection.
    @params:
        data              - Required  : dataset (DataFrame)
        features          - Required  : features list (List)
        cv                - Optional  : cross-validation number of folds (Int)
        max_features      - Optional  : Maximum number of features to consider at each draw (Int)
        nb_iter           - Optional  : number of iterations to execute (Int)
        score             - Optional  : scoring method (Str)
        score_pen_tresh   - Optional  : score treshold of penality (Float)
        clf               - Optional  : Classifiers/ regressor algorithm (sickit-learn instance).
        clf_params        - Optional  : Classifiers/regressor parameters (dict)
    """
    ## initialize Kfolds for cross-validation
    kfold = model_selection.KFold(n_splits=cv, random_state=42)
    ## compute model scores for each randomdly drawn sample of features
    # set feature scores to 0.0
    scores={col:0.0 for col in features}
    # if clf is not given, then take a Decision tree classifier by default.
    clf=clf(**clf_params)
    if clf==None:
      clf=DecisionTreeClassifier(**clf_params)
    for i in range(nb_iter):
        ## draw a features subsample 
        sub_feat=random.choice(a=features,size=random.randint(low=3,high=max_features),replace=False)
        ## compute cross validation score mean
        cross_val= model_selection.cross_validate(clf, data[sub_feat],data['label'], cv=kfold, scoring=score,n_jobs=-1,return_estimator=True)
        sco=cross_val['test_score'].mean()
        feature_importances=np.array([estimator.feature_importances_ for estimator in cross_val['estimator']]).mean(axis=0)
        ## if the score is less than "score_pen_tresh" , then the subsample features gets a penalty instead of a reward
        if sco<score_pen_tresh:
            ## the penalty is measured with the distance to the treshold 
            sco=score_pen_tresh-sco
        ## update concerned features scores
        for j in range(len(sub_feat)):
            scores[sub_feat[j]]+=sco*feature_importances[j] ## the reward/penalty is proportional to feature importnace
        ## print progress
        printProgressBar(i + 1, nb_iter, prefix = 'Progress:', suffix = 'Complete',length=50)   
    ## display  scores distribution
    plt.figure(figsize=(10,4))
    sns.kdeplot(list(scores.values()),shade=True)
    plt.title('Feature selection scores')
    plt.show()
    ## return scores
    return pd.DataFrame(list(scores.values()),columns=['score'],index=list(scores.keys()))

