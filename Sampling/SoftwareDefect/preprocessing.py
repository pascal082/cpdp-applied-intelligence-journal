from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
import os
from imblearn.under_sampling import RandomUnderSampler


def smote(X, y):
    X_smote, y_smote = SMOTE(random_state=40).fit_resample(X, y)
    return (X_smote, y_smote)


def randomOverSampling(X, y):
    X_randomOver, y_randomOver = RandomOverSampler(random_state=40).fit_resample(X, y)

    return (X_randomOver, y_randomOver)


def randomUnderSampling(X, y):
    X_randomOver, y_randomOver = RandomUnderSampler(random_state=40).fit_resample(X, y)

    return (X_randomOver, y_randomOver)


def adasyn(X, y):
    X_adasyn, y_adasyn = ADASYN(random_state=40).fit_resample(X, y)
    return (X_adasyn, y_adasyn)
