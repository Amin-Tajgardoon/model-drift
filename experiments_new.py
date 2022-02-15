#!/usr/bin/env python
# coding: utf-8

"""
Name:AUC_GH.py
Description: Train models on mimic-iii data with knowledge of the years
Author(s): Bret Nestor
Date: Written Sept.6, 2018
Licence:
"""
# print("start")

# from operator import mod

import pandas as pd
import numpy as np
import os
import pathlib
from scipy.sparse.construct import rand
import scipy.stats as stats
# from scipy.stats.stats import mode


### sklearn tools
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, GroupKFold, train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, RFECV
from sklearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import ParameterGrid
from scipy import stats
from sklearn.utils import resample

# XGboost
import xgboost as xgb


# calibration tools
from HIDENIC_overtime_analysis.util.utils import get_calibration_metrics
# https://pypi.org/project/uncertainty-calibration/
import calibration as cal

from category_encoders import TargetEncoder

## independence test and measures
from hyppo.ksample import KSample, MMD
from hyppo.tools import chi2_approx
import ot

## SHAP explanation library
import shap


# import torch
# from torch import nn
# from torch import Tensor
# from torch.autograd import grad
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset
# from torch.utils.data import Dataset
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm, trange

# #for GRU-D
# import torch
# print("2.75")
# try:
#     from GRUD import *
# except:
#     from utils.GRUD import *

# try:
#     from EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
# except:
#     from utils.EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
# import umap

import random
from tqdm import tqdm
import pickle

# import matplotlib.pyplot as plt

import time



# # list global variables

filtered_df=None
label_df=None
years_df=None
sites_df=None

common_indices=None
demo_df=None
embedded_model=None
scaler=None

best_params=None
train_cv_results=None
keep_cols=None
train_means=None


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_measures(y, y_pred_prob, modeltype, threshold=0.5):
    pred = np.array(y_pred_prob >= threshold).astype(int)
    try:
        AUC=sklearn.metrics.roc_auc_score(y, y_pred_prob)
    except Exception as err:
        print("couldn't compute AUC: {}".format(err))
        AUC=np.nan
    try:
        APR=sklearn.metrics.average_precision_score(y, y_pred_prob)
    except Exception as err:
        print("couldn't compute APR: {}".format(err))
        APR=np.nan
    try:
        F1=sklearn.metrics.f1_score(y, pred)
        ACC=sklearn.metrics.accuracy_score(y, pred)
    except Exception as err:
        print("couldn't compute F1, ACC: {}".format(err))
        F1=ACC=np.nan
    try:
        if (modeltype in ['1class_svm', '1class_svm_novel', 'iforest', 'svm']):
            ## min-max scaling of y_pred
            p = (y_pred_prob - np.min(y_pred_prob))/ (np.max(y_pred_prob) - np.min(y_pred_prob))
        else:
            p=y_pred_prob
        _,_,ECE,MCE = get_calibration_metrics(y, p, 10, 'quantile')
    except Exception as err:
        print("couldn't compute ECE,MCE: {}".format(err))
        ECE = MCE = np.nan
    try:
        if (modeltype in ['1class_svm', '1class_svm_novel', 'iforest', 'svm']):
            ## min-max scaling of y_pred
            p = (y_pred_prob - np.min(y_pred_prob))/ (np.max(y_pred_prob) - np.min(y_pred_prob))
        else:
            p=y_pred_prob
        O_E = np.mean(y)/np.mean(p) ## observed-mean over expected-mean
    except Exception as err:
        print("couldn't compute O_E: {}".format(err))
        O_E = np.nan
    try:
        BRIER = sklearn.metrics.brier_score_loss(y, y_pred_prob)
    except Exception as err:
        BRIER = None
    try:
        CE = cal.get_calibration_error(y_pred_prob, y)
    except Exception as err:
        CE=None
    try:
        ECE2 = cal.get_ece(y_pred_prob, y)
    except Exception as err:
        ECE2=None

    return {'AUC':AUC, 'F1':F1, 'ACC':ACC, 'APR':APR, 'ECE':ECE, 'MCE':MCE, 'O_E':O_E, 'BRIER':BRIER, 'CE':CE, 'ECE2':ECE2}

def get_prediction(modeltype, model, X_df, y, subject_id):
    # Different models have different score funcions
    if modeltype in ['lstm','gru']:
        y_pred_prob=model.predict(np.swapaxes(X_df, 1,2))
        pred = list(map(int, y_pred_prob > 0.5))
    elif modeltype=='grud':
        #create test_dataloader X_df, demo_df
        test_dataloader=PrepareDataset(X_df, y, subject_id, train_means, BATCH_SIZE = 1, seq_len = 25, ethnicity_gender=True, shuffle=False)

        predictions, labels, _, _ = predict_GRUD(model, test_dataloader)
        y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
        y=np.squeeze(np.asarray(labels))
        pred=np.argmax(np.squeeze(predictions), axis=1)
        # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
    elif modeltype in ['lr', 'rf', 'mlp', 'knn', 'nb']:
        y_pred_prob=model.predict_proba(X_df)[:,1]
        pred=model.predict(X_df)
    elif modeltype in ['svm']:
        y_pred_prob=model.decision_function(X_df)
        pred=model.predict(X_df)
    elif modeltype in ['rbf-svm']:
        y_pred_prob=model.predict_proba(X_df)[:,1]
        pred=[1 if x > 0.5 else 0 for x in y_pred_prob]
    elif modeltype in ['1class_svm', 'iforest', '1class_svm_novel']:
        ## one-class classifier
        y_pred_prob= -1.0 * model.decision_function(X_df)
        pred= model.predict(X_df)
        pred[pred==1] = 0
        pred[pred==-1] = 1
    elif modeltype == 'mlp_torch':
        y_pred_prob = predict_mlp_torch(model, X_df.values, y, batch_size=64)
        pred=[1 if x > 0.5 else 0 for x in y_pred_prob]
    else:
        raise Exception('dont know proba function for classifier = "%s"' % modeltype)
    return y, y_pred_prob, pred


    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    if bin_strategy == 'quantile':  ## equal-frequency bins 
        # Determine bin edges by distribution of data 
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif bin_strategy == 'uniform': ## equal-width bins
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'bin_strategy' input. bin_Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
    
    abs_error = np.abs(prob_pred - prob_true)

    expected_error = np.average(abs_error, weights=(bin_total / y_true.size)[nonzero])
    max_error = np.max(abs_error)

    return prob_true, prob_pred, expected_error, max_error

def get_ECE(y, y_pred, **kwargs):
    _, _, ECE, _ = get_calibration_metrics(y_true=y, y_prob=y_pred, n_bins=10, bin_strategy='quantile')
    return ECE

def get_MCE(y, y_pred, **kwargs):
    _, _, _, MCE = get_calibration_metrics(y_true=y, y_prob=y_pred, n_bins=10, bin_strategy='quantile')
    return MCE

def get_globals():
    """
    Get all the global variables. Useful for when the script is being imported
    Returns:
        filtered_df, label_df, years_df, common_indices, demo_df, embedded_model, scaler, best_params
    """
    global filtered_df
    global label_df
    global years_df
    return filtered_df, label_df, years_df, common_indices, demo_df, embedded_model, scaler, best_params

def save_filtered_data(data_dir):

    t0 = time.time()

    if len(data_dir)>0:
        DATA_DIR=os.path.join(data_dir, 'filtered_data.h5')
    else:
        DATA_DIR='E:/Data/HIDENIC_EXTRACT_OUTPUT_DIR/POP_SIZE_100/ITEMID_REP/filtered_data_100.h5'
    
    filtered_df.to_hdf(DATA_DIR, key='filtered_df', mode='a')
    demo_df.to_hdf(DATA_DIR, key='demo_df', mode='a')
    label_df.to_hdf(DATA_DIR, key='label_df', mode='a')
    pd.Series(common_indices).to_hdf(DATA_DIR, key='common_indices', mode='a')

    t1 = time.time()
    print("finished saving filtered data in {:10.1f} seconds.".format(t1-t0))
    return


def load_data(data_dir="", num_samples=None):
    """
    Load the data from a static file at the specified data_dir extension
    Inputs:
        max_time (int): the maximum number of hours to use for the prediction task
        gap_time (int): the minimum number of hours between the maximum time and the prediction task. We don't want the prediction task happening within, or immediately after the observation period.
        data_dir (str): the path to where the data are contained
    Returns:
        None : the results are updated as globals.
    """
    global filtered_df
    global label_df
    global common_indices
    global demo_df

    idx = pd.IndexSlice

    if len(data_dir)>0:
        DATA_DIR=os.path.join(data_dir, 'all_hourly_data.h5')

    if not(os.path.isfile(DATA_DIR)):
        DATA_DIR=input('Could not find data.hdf. Please enter the full path to the file:\n')
        if not(os.path.isfile(DATA_DIR)):
            raise('Not a valid directory:\n {}'.format(DATA_DIR))
    
    print("reading vitals_labs")
    t0 = time.time()
    df=pd.read_hdf(DATA_DIR, key='vitals_labs')
    df = df.droplevel("icustay_id")
    t1 = time.time()
    print("finished reading vitals_labs features in {:10.1f} seconds.".format(t1-t0))
    if num_samples is not None:
        print(f'selecting only {num_samples} rows from vitals_labs dataframe ...')
        df=df.iloc[:num_samples,:]
    print(df.shape)

    # read demographics and otcomes
    print("loading demographics ...")
    demo_df = read_demographics(data_dir)
    sites_df = read_sites_data(data_dir)
    demo_df = demo_df.join(sites_df[['icu_unit', 'hospital', 'icu_category']], how='left')
    demo_df = demo_df.droplevel("icustay_id")
    del sites_df
    print("demographics loaded")

    ## convert to datetime
    demo_df.intime = pd.to_datetime(demo_df.intime)

    # get common hospital stays
    common_indices=list(set(demo_df.index.get_level_values('hadm_id')).intersection(set(df.index.get_level_values('hadm_id'))))

    print("applying common indices ...")
    t0 = time.time()
    #apply common indices
    demo_df=demo_df.loc[demo_df.index.get_level_values("hadm_id").isin(common_indices), :]
    filtered_df=df.loc[df.index.get_level_values("hadm_id").isin(common_indices), :]
    del df
    t1 = time.time()
    print("applied common indices in {:10.1f} seconds.".format(t1-t0))

    ## create label_df, consists of target variables only
    label_df = demo_df[['mort_icu', 'los_icu']].copy()
    label_df['los_3']=np.zeros((len(label_df),1)).ravel()
    label_df.loc[label_df['los_icu']>=3*24, 'los_3'] = 1
    demo_df['los_3'] = label_df['los_3'].values

    return filtered_df, label_df, demo_df

def read_demographics(data_dir):
    idx = pd.IndexSlice
    if len(data_dir)>0:
        DEMO_DIR=os.path.join(data_dir, 'static_data.csv')
    else:
        DEMO_DIR='E:/Data/HIDENIC_EXTRACT_OUTPUT_DIR/POP_SIZE_100/ITEMID_REP/static_data_100.csv'
    if not(os.path.isfile(DEMO_DIR)):
        DEMO_DIR=input('Could not find static.csv. Please enter the full path to the file:\n')
        if not(os.path.isfile(DEMO_DIR)):
            raise('Not a valid directory')
    
    demo_df=pd.read_csv(DEMO_DIR, index_col=[0,1,2])
    # demo_df = demo_df.droplevel('icustay_id')
    demo_df.loc[:, 'los_icu']*=24 #icu data from days to hours
    # mask = demo_df.loc[:, 'los_icu'].values > max_time+gap_time
    # demo_df=demo_df.loc[idx[mask, :], :]
    demo_df = pd.get_dummies(demo_df, columns=['gender', 'race'])

    return demo_df


# In[5]:


def generate_labels(label_df, subject_index, label):
    """
    Get the labels for a particular target matching the subject index
    Input:
        label_df (pd.DataFrame): the dataframe containing the desired labels
        subject_index (list): the indices of the data to find the training labels
        label (str): the name of the desired label
    Returns:
        (list): all of the labels for the given dubject index and label
    """
    return label_df.loc[subject_index, label].values.tolist()


# In[6]:


def impute_simple(df, means=None, method='mean', missing_indicator=True):
    """
    concatenate the forward filled value, the mask of the measurement, and the time of the last measurement
    refer to paper
    Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, "Recurrent Neural Networks for Multivariate Time Series with Missing Values," Scientific Reports, vol. 8, no. 1, p. 6085, Apr 2018.

    Input:
        df (pandas.DataFrame): the dataframe with timeseries data in the index.
        time_index (string, optional): the heading name for the time-series index.
    Returns:
        df (pandas.DataFrame): a dataframe according to the simple impute algorithm described in the paper.
    """

    if missing_indicator:
        masked_df=pd.isna(df)
        masked_df=masked_df.astype(int)
        masked_df.rename({'mean': 'is_nan'}, axis=1, level=1, inplace=True)
        df_prime=pd.concat([df, masked_df], axis=1)
    else:
        df_prime=df.copy()
        del df
    df_prime.columns=df_prime.columns.rename("simple_impute", level=0)#rename the column level

    #fill using ffill, bfill, and means
    df_prime=df_prime.unstack().groupby(level=0, axis=1).apply(
        lambda g: g.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1))
    if method=='mean':
        if means is not None:
            df_prime.fillna(means)
        else:
            df_prime.apply(lambda x: x.fillna(x.mean()))

    #swap the levels so that the simple imputation feature is the lowest value
    col_level_names=list(df_prime.columns.names)
    col_level_names.append(col_level_names.pop(0))

    df_prime=df_prime.reorder_levels(col_level_names, axis=1)
    df_prime.sort_index(axis=1, inplace=True)

    return  df_prime


# # def read_years_data

# In[7]:


def read_years_data(data_dir=""):
    """
    Read the csv linking the years to the dataset (not reproducible without limited use agreement).
    Input:
        data_dir (str): the path to the mimic year map
    """
    global years_df
    global common_indices
    global demo_df

#     if len(data_dir)>0:
#         pathname=os.path.join(data_dir,'yearmap_2018_07_20_tjn.csv' )
#     else:
#         pathname="/scratch/tjn/mimic-years/yearmap_2018_07_20_tjn.csv"

#     if not(os.path.isfile(pathname)):
#         pathname=input('Could not find yearmap_2018_07_20_tjn.csv. Please enter the full path to the file:\n')
#         if not(os.path.isfile(pathname)):
#             raise('Not a valid directory for years_df')

#     years_df=pd.read_csv(pathname, index_col=0, header=None)
    years_df = demo_df.reset_index()[['hadm_id', 'intime']].set_index('hadm_id')
    years_df['year'] = pd.to_datetime(years_df['intime']).dt.year
    years_df['month'] = pd.to_datetime(years_df['intime']).dt.month
    years_df.drop(columns=['intime'], inplace=True)
    
#     years_df.columns=["hadm_id".encode(), "year"]
#     years_df.set_index('hadm_id'.encode(), inplace=True)
    
    years_df=years_df.loc[common_indices]
    print(" loaded years")
    return

def read_sites_data(data_dir=""):
    """
    """
    # global sites_df
    # global common_indices

    if len(data_dir)>0:
        DATA_PATH=os.path.join(data_dir, 'site_info.pkl')
    else:
        DATA_PATH='E:/Data/HIDENIC_EXTRACT_OUTPUT_DIR/POP_SIZE_100/ITEMID_REP/site_info.pkl'

    sites_df = pd.read_pickle(DATA_PATH)
    sites_df.set_index(['subject_id', 'hadm_id', 'icustay_id'], inplace=True)
    # sites_df=sites_df.loc[common_indices]
    print("loaded sites info")
    return sites_df

# In[8]:


def flattened_to_sequence(X, vect=None):
    """
    Turn pandas dataframe into sequence

    Inputs:
        X (pd.DataFrame): a multiindex dataframe of MIMICIII data
        vect (tuple) (optional): (timesteps, non-time-varying columns, time-varying columns, vectorizer(dict)(a mapping between an item and its index))
    Returns:
        X_seq (np.ndarray): Input 3-dimensional input data with the [n_samples, n_features, n_hours]
    """
    hours_in_values=X.index.get_level_values(X.index.names.index('hours_in'))

    output=np.dstack((X.loc[(slice(None), slice(None), i), :].values for i in sorted(set(hours_in_values))))

    #ex. these are the same
    # print(output[0,:10, 5])
    # print(X.loc[(X.index.get_level_values(0)[0], X.index.get_level_values(1)[0], 5), X.columns.tolist()[:10]].values)


    return output, None


def get_ctakes_level(df_in, abs_val=True):
    """
    Replaces itemids with ctakes outputs by averaging across outcomes throughout time.
    Inputs:
        df_in (df): the original itemid df.
    Returns:
        cui_df (df): a dataframe with the ctakes column headings
    """


    filename=os.path.join(os.getcwd(), 'ctakes_extended_spanning.p')
    # print(filename)
    if os.path.exists(filename):
        with open(filename , 'rb') as f:
            mappings=pickle.load(f)
    else:
        raise Exception('no file named {}'.format(filename))

    mappings=dict(mappings) #dict of itemid: [cuis]

    columns=df_in.columns.tolist()


    try:
        col_number=list(df_in.columns.names).index('itemid'.encode()) #drop unecessary index
    except:
        col_number=list(df_in.columns.names).index('itemid') #drop unecessary index
    itemids=list(df_in.columns.get_level_values(col_number))



    #remove all itmeids that aren't in our itemids
    # print("deleting unecessary keys")
    for key in list(mappings.keys()):
        try:
            k2=int(key.split("_")[0])
            if k2 not in itemids:
                raise
        except:
            del mappings[key]


    reverse_mappings=invert_dict(mappings) # dict of cui: [itemids]

    all_features=sorted(list(reverse_mappings.keys()))


    cui_df=pd.DataFrame(index=df_in.index)
    for cui in sorted(list(reverse_mappings.keys())):
        for item in reverse_mappings[cui]:
            cols=[]
            try:
                item=int(item.split('_')[0])
            except:
                continue
            if item in itemids:
                cols.append(columns[itemids.index(item)])
        if len(cols)!=1:
            cols=list(zip(*cols))

        try:
            data_selection=df_in.loc[:, cols].values
        except:
            # print(cols)
            # print(list(zip(*cols)))
            raise Exception("cannot index dataframe with the columns: {}".format(cols))

        averaged_selection=np.mean(data_selection, axis=1)
        if abs_val:
            averaged_selection=np.abs(averaged_selection)
        #append averaged selection to list
        cui_df.loc[:, cui]=averaged_selection

    cui_df.columns.names=['cui']

    return cui_df


# # def data_preprocessing

# In[45]:


def data_preprocessing(train_df, test_df, target, mode="basic", missing_rate_cutoff=0.8, imputation_method="simple_impute", transform_params=None):
    """
    Clean the data into machine-learnable matrices.

    Inputs:
        df_in (pd.DataFrame): a multiindex dataframe of MIMICIII data
        level (string or bytes): the level of the column index to use as the features level
        imputation_method (string): the method for imputing the data, either Forward, or Simple
        Target (string): the heading to select from the labels in the outcomes_df, i.e. LOS or died
    Returns:
        X (pd.DataFrame): the X data to input to the model
        y (array): the labels for the corresponding X-data
    """
    global demo_df
    global embedded_model
    global scaler
    global keep_cols
    global select_cols
    global hospital_target_encoder
    global age_scaler
    global demo_mode
    
    idx = pd.IndexSlice

    missing_indicator=True
    if mode=="basic_no_indicator":
        missing_indicator=False

    if transform_params is not None:
        scaler=transform_params["scaler"]
        keep_cols=transform_params["keep_cols"]
        select_cols=transform_params["select_cols"]
        hospital_target_encoder=transform_params["hospital_target_encoder"]
        age_scaler=transform_params["age_scaler"]
        demo_mode=transform_params["demo_mode"]

    if train_df is not None:
        train_df = train_df.loc[:, idx[:, "mean"]] # drop count and std columns
        # train_df = train_df.droplevel("icustay_id") #drop unecessary index level
    if test_df is not None:    
        test_df = test_df.loc[:, idx[:, "mean"]] # drop count and std columns
        # test_df = test_df.droplevel("icustay_id") #drop unecessary index level

    # if mode == "basic": # just use labs & vitals with missing_rate <= missing_rate_cutoff
    if train_df is not None:
        # select features with missing_rate <= missing_rate_cutoff
        missing_rates = train_df.isna().mean() 
        select_cols = missing_rates[missing_rates <= missing_rate_cutoff].index
        train_df = train_df.loc[:, select_cols]
    if test_df is not None:
        test_df = test_df.loc[:, select_cols]

    if imputation_method=="simple_impute":
        print("imputing simple")
        if train_df is not None:
            train_df = impute_simple(train_df, means=None, missing_indicator=missing_indicator)           
            df_means=train_df.mean(skipna=True, axis=0)
            df_stds=train_df.std(skipna=True, axis=0)
            scaler=(df_means, df_stds)
        if test_df is not None:
            df_means, df_stds = scaler
            test_df = impute_simple(test_df, means=df_means, missing_indicator=missing_indicator)

        print("imputed")

    #standard scaler
    print("fitting new scaler")

    if train_df is not None:
        # first stack the columns
        train_df=train_df.stack(level='hours_in', dropna=False)

        # now remove the columns with redundant values
        keep_cols=train_df.columns.tolist()
        keep_cols=[col for col in keep_cols if len(train_df[col].unique())!=1]
        train_df=train_df[keep_cols]

        #now we can unstack the hours and sort them to the same as the original dataframe
        train_df=train_df.unstack()

        # if impute:
        train_df.columns=train_df.columns.swaplevel('hours_in', 'simple_impute')

        #take the columns again so that we don't have to unstack every time
        keep_cols=train_df.columns.tolist()

        # pandas scaler version
        df_means=train_df.mean(skipna=True, axis=0)
        df_stds=train_df.std(skipna=True, axis=0)
        scaler=(df_means, df_stds)
        train_df = (train_df-df_means)/df_stds

        ## impute NAs/Infs with zero (caused by rescaling and std=0)
        # pd.options.mode.use_inf_as_na = True # replace inf with nan
        train_df.replace([np.inf, -np.inf], np.nan, inplace=True) ## replace inf with nan
        train_df.fillna(0, inplace=True)

    if test_df is not None:
        ## fit scaler to test dataset
        test_df = test_df[keep_cols]
        df_means, df_stds =scaler
        test_df = (test_df-df_means)/df_stds
        ## impute NAs/Infs with zero (caused by rescaling and std=0)
        # pd.options.mode.use_inf_as_na = True # replace inf with nan
        test_df.replace([np.inf, -np.inf], np.nan, inplace=True) ## replace inf with nan
        test_df.fillna(0, inplace=True)
    
    print("finished scaler")


    ## join demographic features
    demo_column_names=demo_df.columns.tolist()
    if train_df is not None:
        nlevels = train_df.columns.nlevels
        columns_names = train_df.columns.names
    else:
        nlevels = test_df.columns.nlevels
        columns_names = test_df.columns.names
    if demo_df.columns.nlevels != nlevels:
        print("making demo_df levels equal to train_df/test_df ...")

        for i in range((nlevels - 1)):
            demo_df=pd.concat([demo_df], axis=1, keys=['DEMO']) #.swaplevel(0, 1, 1).swaplevel(0, 1, 1)

        for level, item in enumerate(list(columns_names)):
            demo_df.columns=demo_df.columns.rename(item, level=level)
            demo_df.columns=demo_df.columns.set_levels(demo_column_names,level=level)

        demo_df.columns=demo_df.columns.set_codes([[i for i in range(len(demo_df.columns.levels[0]))] for i in range(3)],level=[0,1,2])

    # make both dataframes have the same number of column levels
    while  len(demo_df.columns.names)!=len(columns_names):
        if len(demo_df.columns.names)>len(columns_names):
            demo_df.columns=demo_df.columns.droplevel(0)
        elif len(demo_df.columns.names)<len(columns_names):
            raise Exception("number of demo_df columns is less than the number of train_df/test_df columns")

    demo_cols = ['age'] + [c for c in demo_df.columns.get_level_values(0) if c.startswith('gender_') or c.startswith('race_')]

    ## join demos to train dataset
    if train_df is not None:
        right = demo_df.loc[[(item[0], item[1]) for item in train_df.index.tolist()], (slice(None), slice(None), demo_cols)].values.astype(np.float32)
        left=train_df.values.astype(np.float32)
        result=np.concatenate((left, right), axis=1) #slow step timeit for combiing np.random.rand(15000, 10000) with np.random.rand(15000, 7)-> concat=50 iterations 192.49 s , hstack=50 iterations  212.61 , append=50 iterations 233.63 s
        ind=pd.MultiIndex.from_tuples([(item[0], item[1]) for item in train_df.index.tolist()], names=['subject_id', 'hadm_id'])
        cols=train_df.columns.union(demo_df.loc[:, (slice(None), slice(None), demo_cols)].columns, sort=None)
        train_df=pd.DataFrame(data=result, index=ind, columns=cols)
        
        ## fill any missing demographics with mean/mode
        demo_mode = train_df.loc[:, idx[:,:, demo_cols[1:]]].mode().iloc[0,:]
        train_df.loc[:, idx[:,:, demo_cols[1:]]].fillna(demo_mode, inplace=True)

        ## standardize age
        age_mean, age_std = train_df.loc[:, idx[:,:, "age"]].mean(), train_df.loc[:, idx[:,:, "age"]].std()
        age_scaler = (age_mean, age_std)
        train_df.loc[:, idx[:,:, "age"]].fillna(age_mean, inplace=True)
        train_df.loc[:, idx[:,:, "age"]] = (train_df.loc[:, idx[:,:, "age"]] - age_mean)/age_std

        ## target-encode Hospital
        hospital_target_encoder = TargetEncoder(smoothing=1.0, handle_unknown='value')
        hospital_target_encoder.fit(demo_df.loc[train_df.index, 'hospital'],  demo_df.loc[train_df.index, target])
        train_df.loc[:, idx["hospital","hospital", "hospital"]] = hospital_target_encoder.transform(
            demo_df.loc[train_df.index, "hospital"]).values.ravel()
    
    if test_df is not None:
    ## join demos to test dataset
        right = demo_df.loc[[(item[0], item[1]) for item in test_df.index.tolist()], (slice(None), slice(None), demo_cols)].values.astype(np.float32)
        left=test_df.values.astype(np.float32)
        result=np.concatenate((left, right), axis=1) #slow step timeit for combiing np.random.rand(15000, 10000) with np.random.rand(15000, 7)-> concat=50 iterations 192.49 s , hstack=50 iterations  212.61 , append=50 iterations 233.63 s
        ind=pd.MultiIndex.from_tuples([(item[0], item[1]) for item in test_df.index.tolist()], names=['subject_id', 'hadm_id'])
        cols=test_df.columns.union(demo_df.loc[:, (slice(None), slice(None), demo_cols)].columns, sort=None)
        test_df=pd.DataFrame(data=result, index=ind, columns=cols)

        ## fill any missing demographics with mode (transform)
        test_df.loc[:, idx[:,:, demo_cols[1:]]].fillna(demo_mode, inplace=True)

        ## standardize age (transform to test)
        age_mean, age_std = age_scaler
        test_df.loc[:, idx[:,:, "age"]].fillna(age_mean, inplace=True)
        test_df.loc[:, idx[:,:, "age"]] = (test_df.loc[:, idx[:,:, "age"]] - age_mean)/age_std

        ## target-encode Hospital
        test_df.loc[:, idx["hospital","hospital", "hospital"]] = hospital_target_encoder.transform(
            demo_df.loc[test_df.index, "hospital"]).values.ravel()

    # train_subject_index=[(item[0], item[1]) for item in train_df.index.tolist()] #select the subject_id and hadm_id
    # test_subject_index=[(item[0], item[1]) for item in test_df.index.tolist()] #select the subject_id and hadm_id

    # try:
    #     #get list of gender and ethnicity here.
    #     gender=X_df.loc[(slice(None), slice(None)), (slice(None), slice(None), 'gender_F')].values.ravel()

    #     ethnicity=X_df.loc[(slice(None), slice(None)), (slice(None), slice(None), [c for c in X_df.columns.get_level_values(0) if c.startswith('race_')])].values

    #     #one hot encode ethnicity
    #     ethnicity=np.argmax(ethnicity, axis=1).ravel()
    # except:
    #     #dimensionality reduction techniques lose feature levels so they are different shapes.
    #     gender=X_df.loc[(slice(None), slice(None)), ('gender_F', 0)].values.ravel()

    #     ethnicity=X_df.loc[(slice(None), slice(None)), ([c for c in X_df.columns.get_level_values(0) if c.startswith('race_')], 0)].values
    #     #one hot encode ethnicity
    #     ethnicity=np.argmax(ethnicity, axis=1).ravel()

    # assert gender.shape==ethnicity.shape
   
    train_y = None if train_df is None else label_df.loc[train_df.index, target]
    test_y = None if test_df is None else label_df.loc[test_df.index, target]

    transform_params={}
    transform_params["scaler"]=scaler
    transform_params["keep_cols"]=keep_cols
    transform_params["select_cols"]=select_cols
    transform_params["hospital_target_encoder"]=hospital_target_encoder
    transform_params["age_scaler"]=age_scaler
    transform_params["demo_mode"]=demo_mode

    return train_df, train_y, test_df, test_y, transform_params #, timeseries_vect, representation_vect, gender, ethnicity, [item[0] for item in subject_index]


# GRU-D stuff
def time_since_last(arr, mask):
    """
    Calculate the time since the last event
    """
    output=np.ones_like(arr)
    # same as
    ones_index=np.asarray(np.where(mask==1)).ravel()
    tmp=ones_index+1
#     output[tmp[tmp<len(output)]]=1
#     output[np.where(arr==0)]=0
    zeros_index=np.asarray(np.where(arr==0)).ravel()
    #fill values until the next zero
    both=np.sort(np.unique(np.concatenate((ones_index, zeros_index))))
    mask_zeros_index=np.asarray(np.where(mask==0)).ravel()
    if len(mask_zeros_index)==0:
        return output
    diff=np.insert(np.diff(mask_zeros_index), 0, 0)#remove donsecutive elements (where diff==1)
#     print(mask_zeros_index)
#     print(diff)
    mask_zeros_index=mask_zeros_index[diff!=1]
#     print(mask_zeros_index)

    for ind, val in enumerate(mask_zeros_index):
        #get the next one or stop zero index
        try:
            end_of_seq=np.min(both[np.where(both>val)])
            output[val:end_of_seq+1]=np.arange(1, end_of_seq+2-val)
        except:
            end_of_seq=len(output)
            output[val:]=np.arange(1, len(output[val:])+1)

    #zero index up to and including next one is np.arange(0, len_until_1+1)
    for ind in zeros_index:
        try:
            next_one=np.min(both[np.where(both>ind)])
            output[ind:next_one+1]=np.arange(0, next_one+1-ind)
        except:
            next_one=len(output)
            output[ind:]=np.arange(0, len(output[ind:])+1)
    return output

def PrepareDataset(df_input,                     y,                     subject_index_in,                    train_means,                     BATCH_SIZE = 40,                     seq_len = 25,                    ethnicity_gender=False,                     shuffle=False,
                    drop_last=True):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """

    df_in=df_input.copy()

    assert set([item[0] for item in df_in.index.tolist()])==set(subject_index_in)

#     subject_index=[(item[0], item[1]) for item in df_in.index.tolist()] #select the subject_id and hadm_id
    subject_index=sorted(df_in.index.tolist()) #select the subject_id, hadm_id, and hours_in
    subject_index=sorted(subject_index, key=lambda x: subject_index_in.index(x[0])) #now it is in the order of the subject_index_in


    #masked data
    masked_df=pd.notna(df_in)
    masked_df=masked_df.apply(pd.to_numeric)

    #we can fill in the missing values with any number now (they get multiplied out to be zero)
    df_in=df_in.fillna(0)

    #time since last measurement
    time_index=None
    if not(time_index):
        time_index='hours_in'
    index_of_hours=df_in.index.names.index(time_index)
    time_in=[item[index_of_hours] for item in df_in.index.tolist()]
    time_df=df_in.copy()
    for col in tqdm(time_df.columns.tolist()):
        mask=masked_df[col].values
        time_df[col]=time_since_last(time_in, mask)
    time_df=time_df.fillna(0)

    #last observed value
    X_last_obsv_df=df_in.copy()

    # do the mean imputation for only the first hour
    columns=X_last_obsv_df.columns.tolist()
    subset_data=X_last_obsv_df.loc[(slice(None), slice(None), 0), columns]
    subset_data=subset_data.fillna(pd.DataFrame(np.mean(train_means, axis=1).reshape(1, -1), columns=columns).mean()) #replace first values with the mean of the whole row (not sure how original paper did it, possibly just fill with zeros???)


    #replace first hour data with the imputed first hour data
    replace_index=subset_data.index.tolist()
    X_last_obsv_df.loc[replace_index, columns]=subset_data.values
    X_last_obsv_df=X_last_obsv_df.fillna(0)

    # now it is safe for forward fill
    #forward fill the rest of the sorted data
    X_last_obsv_df=X_last_obsv_df.loc[subject_index,:]
    X_last_obsv_df=X_last_obsv_df.fillna(method='ffill')



    gender=df_in.loc[(subject_index_in, slice(None), 0), 'F'].values.ravel()
    ethnicity=df_in.loc[(subject_index_in, slice(None), 0), ['asian', 'black','hispanic','white', 'other']].values
    #one hot encode ethnicity
    ethnicity=np.argmax(ethnicity, axis=1).ravel()



    # collect matrices into 3d numpy arrays

    measurement_data=np.swapaxes(flattened_to_sequence(df_in.loc[subject_index, :])[0], 1,2)
    X_last_obsv_data=np.swapaxes(flattened_to_sequence(X_last_obsv_df.loc[subject_index, :])[0], 1,2)
    mask_data=np.swapaxes(flattened_to_sequence(masked_df.loc[subject_index, :])[0], 1,2)
    time_data=np.swapaxes(flattened_to_sequence(time_df.loc[subject_index, :])[0], 1,2)


    measurement_data=torch.from_numpy(measurement_data.astype(np.float32))
    X_last_obsv_data=torch.from_numpy(X_last_obsv_data.astype(np.float32))
    mask_data=torch.from_numpy(mask_data.astype(np.float32))
    time_data=torch.from_numpy(time_data.astype(np.float32))
    label= torch.as_tensor(y.astype(np.long))
    ethnicity=torch.as_tensor(ethnicity.astype(np.long))
    gender=torch.as_tensor(gender.astype(np.long))

    if ethnicity_gender:
        #optional to include for more complex model designs
        train_dataset = utils.TensorDataset(measurement_data, X_last_obsv_data, mask_data, time_data, label, ethnicity, gender)
    else:
        train_dataset = utils.TensorDataset(measurement_data, X_last_obsv_data, mask_data, time_data, label)

    dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=shuffle, drop_last = drop_last)

    return dataloader




def create_rnn(seqlen, n_features, hidden_layer_size, optimizer, activation, dropout, recurrent_dropout, recurrent_unit):
    """
    Create RNN model in external function with sklearn style API
    """
    model=rnn_tf2(input_shape=(seqlen,n_features), hidden_layer_size=hidden_layer_size, modeltype=recurrent_unit,  activation=activation, input_dropout=recurrent_dropout, output_dropout=dropout, optimizer=optimizer)

    return model

# class MLP_large(nn.Module):
#     def __init__(self, in_size):
#         super().__init__()      
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(in_size, 1000),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(1000, 500),
#             nn.Dropout(p=0.5)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(500, 200),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(100, 2),
#             nn.Softmax(dim=1)
#         )
        
#     def forward(self, x):
#         fe = self.feature_extractor(x)
#         return self.classifier(fe)

# class MLP_small(nn.Module):
#     def __init__(self, in_size):
#         super().__init__()      
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(in_size, 500),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(500, 100),
#             nn.Dropout(p=0.5)
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Linear(100, 50),
#             nn.Dropout(p=0.5),
#             nn.ReLU(),
#             nn.Linear(50, 2),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         fe = self.feature_extractor(x)
#         return self.classifier(fe)

def loop_iterable(iterable):
    while True:
        yield from iterable
        
def predict_mlp_torch(model_state, X, y, batch_size):
    with torch.no_grad():
        mlp_model = best_params['mlp_model']
        model = mlp_model(X.shape[1])
        model.load_state_dict(model_state)
        dataset = TensorDataset(Tensor(X), Tensor(y).view(-1,1))
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=1, pin_memory=True)
        y_pred = []
        for x, y_true in dataloader:
            x, y_true = x.to(device), y_true.to(device)
            y_pred += model(x)[:,1].numpy().tolist()
    return np.array(y_pred)

def create_dataloaders(X, y, batch_size):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train).view(-1,1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                              num_workers=1, pin_memory=True)
    val_dataset = TensorDataset(Tensor(X_val), Tensor(y_val).view(-1,1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False,
                              num_workers=1, pin_memory=True)

    return train_loader, val_loader

def do_epoch(model, dataloader, criterion, n_iter, optim=None):
    total_loss, total_accuracy = 0, 0
    total_auc, auc_iter = 0, 0
    batch_iterator = loop_iterable(dataloader)
    # for x, y_true in tqdm(dataloader, leave=False):
    for _ in trange(n_iter, leave=False):
        x, y_true = next(batch_iterator)
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)[:,1].view(-1,1)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
        with torch.no_grad():
            try:
                total_auc += roc_auc_score(y_true.numpy(), y_pred.numpy())
                auc_iter += 1
            except:
                pass
    mean_loss = total_loss / n_iter #len(dataloader)
    mean_accuracy = total_accuracy / n_iter #len(dataloader)
    mean_auc = total_auc / auc_iter if auc_iter!=0 else 0

    return mean_loss, mean_accuracy, mean_auc


def mlp_torch(X, y, params, batch_size, epochs, n_iter, logdir):
    mlp_model, weight_decay, lr, auc_plateau_max = params['mlp_model'], params['weight_decay'], params['lr'], params['auc_plateau_max']
    train_loader, val_loader = create_dataloaders(X, y, batch_size)
    model = mlp_model(in_size=X.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True, eps=1e-8)
    criterion = torch.nn.BCELoss()

    auc_plateau = 0
    best_auc = 0
    best_model_state = None
    logdir_suffix = model.__class__.__name__ + ',lr=' + str(lr) + ',l2=' + str(weight_decay)
    writer = SummaryWriter(logdir/logdir_suffix)
    for epoch in range(1, epochs+1):
        model.train()
        n_iter = len(train_loader)*2
        train_loss, train_accuracy, train_auc = do_epoch(model, train_loader, criterion, n_iter=n_iter, optim=optim)

        model.eval()
        with torch.no_grad():
            n_iter = len(val_loader)
            val_loss, val_accuracy, val_auc = do_epoch(model, val_loader, criterion, n_iter=n_iter, optim=None)

        writer.add_scalars('loss', {"train_loss": train_loss,
                                    "val_loss": val_loss}, epoch)
        writer.add_scalars('auc', {"train_auc": train_auc,
                                    "val_auc": val_auc}, epoch)
        writer.add_scalars('acc', {"train_acc": train_accuracy,
                                    "val_acc": val_accuracy}, epoch)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, '
                   f'train_auc={train_auc:.4f}, '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}, val_auc={val_auc:.4f}')

        if epoch == 1:
            best_model_state = model.state_dict()

        if val_auc > best_auc:
            auc_plateau = 0
            print('Saving model...')
            best_auc = val_auc
            # torch.save(model.state_dict(), outdir/'mlp.pt')
            best_model_state = model.state_dict()
        else:
            auc_plateau += 1
            if auc_plateau > auc_plateau_max:
                print(f"val_auc not improved for {auc_plateau_max:03d} epochs. Stopping the training ...\n")
                return best_model_state, best_auc

        lr_schedule.step(val_loss)

    writer.flush()
    writer.close()

    return best_model_state, best_auc

def classifier_select(X, y, target, is_time_series, subject_index, output_dir, modeltype='rf', random_state=1, randomSearchCVargs={},
                    feature_selection=False, data_preprocessing_mode='basic',  **feature_selection_args):
    """
    Select the best model for the data using CV
    Inputs:
        X: the input data for the patients
        y: the labels for the patients
        is_timeseries: whether or not the model is recurrent
        subject_index: the subjects of the dataframe to use in the model.
        modeltype (str): select from the implemented models
    Returns:
        model : A model for testing the test data on.
    """
    global best_params
    global train_cv_results
    global train_means
    global n_threads
    global logdir

    idx = pd.IndexSlice

    if modeltype=='grud':
        #get the means of the training set
        train_means=np.nanmean(np.swapaxes(flattened_to_sequence(X.loc[(subject_index, slice(None), slice(None)),:])[0], 1,2), axis=0, keepdims=True) #dimension is [1, num_steps, num_feat]
        #fill in missing values with the mean of that entire row
        def replace_nan(x):
            x[np.where(np.isnan(x))]=np.nanmean(np.squeeze(x))
            return x
        train_means=np.apply_along_axis(replace_nan, 2, train_means)
        train_means[np.where(np.isnan(train_means))]=0 # fill with zeros where the mean is unknown

        print(train_means.shape)

        # split the train and val data


        ind_label=np.concatenate((np.asarray(subject_index).reshape(-1,1), np.asarray(y).reshape(-1,1)), axis=1)
        print(ind_label.shape)
        np.random.shuffle(ind_label) # use numpy so that we can exploit the environment random seed
        subject_index=list(ind_label[:,0].reshape(-1))
        y=ind_label[:,1].reshape(-1)


        val_len=int(0.8*len(subject_index))

        #train_dataloader
        train_dataloader=PrepareDataset(X.loc[(subject_index[:val_len], slice(None), slice(None)),:], y[:val_len], subject_index[:val_len], train_means, BATCH_SIZE = 64, seq_len = 25, ethnicity_gender=False, shuffle=True)
        # val_dataloader
        valid_dataloader=PrepareDataset(X.loc[(subject_index[val_len:], slice(None), slice(None)),:], y[val_len:], subject_index[val_len:], train_means, BATCH_SIZE = 64, seq_len = 25, ethnicity_gender=False, shuffle=True)
        # fit model (assume same hidden size?)

        measurement, measurement_last_obsv, mask, time_, labels = next(iter(train_dataloader))

        [batch_size, step_size, fea_size] = measurement.size()
        input_dim = fea_size
        hidden_dim = 67 #fea_size #Original GRU-D paper uses a size of 67 on MIMIC  III
        output_dim = 67 #fea_size

        grud = GRUD(input_dim, hidden_dim, output_dim, train_means, output_last = True)

        model, stats= train_GRUD(grud, train_dataloader, valid_dataloader, num_epochs = 300, patience = 10, min_delta = 0.00001)
        # return best model.
        return model

    if is_time_series:
        # Unlike for scikit-learn, there's no built in CV hyperparam search for keras. Doing it manually


        ##########################
        # What hyperparams to optimize?
        param_grid = {}

        param_grid['epochs'] = [2, 5, 10, 15] # randint(5,20)
        param_grid['hidden_layer_size'] = [16, 32,64,128]
        param_grid['optimizer'] = ['rmsprop', 'adam', 'adagrad']
        param_grid['activation'] = ['tanh', 'relu']
        param_grid['dropout'] = [0, 0.1, 0.2, 0.3, 0.4]
        param_grid['recurrent_dropout'] = [0, 0.1, 0.2, 0.3, 0.4]
        param_grid['recurrent_unit'] = [modeltype]
        #print(param_grid)
        ##########################

        X=np.swapaxes(X, 1,2)

        ##########################
        # Make 5 folds
        inds = list(range(X.shape[0]))
        random.shuffle(inds)

        # assign each patient to one of the folds
        num_folds = 5
        fold_inds = [[] for _ in range(num_folds)]
        for i,ind in enumerate(inds):
            fold_inds[i%num_folds].append(ind)

        # create the folds
        folds = []
        for fold_num in range(num_folds):
            train_inds = list( set(inds) - set(fold_inds[fold_num]) )

            x_train = X[train_inds         ,:,:]
            x_test  = X[fold_inds[fold_num],:,:]

            y_train = y[train_inds         ]
            y_test  = y[fold_inds[fold_num]]

            f = {'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test}
            folds.append(f)
        #print('folds made')
        ##########################

        num_samples, timesteps, n_features = X.shape
        print(X.shape)

        # best_params = {'dropout': 0.24864319443479377, 'optimizer': 'adam', 'activation': 'tanh', 'recurrent_dropout': 0.018258041475959774, 'hidden_layer_size': 16, 'recurrent_unit':modeltype}
        # best_params = {'recurrent_unit': modeltype, 'recurrent_dropout': 0, 'optimizer': 'rmsprop', 'hidden_layer_size': 128, 'dropout': 0.2, 'activation': 'tanh'}

        try:
            #if best_params is not Nonetype this will work
            if 'epochs' in best_params:
                epochs = best_params['epochs']
                del best_params['epochs']
            else:
                epochs = 5

            # epochs=int((params['recurrent_dropout']+params['dropout'])/0.2)+2
            do_hyperparam_search=0
        except:
            #otherwise we have to do a parameter search
            do_hyperparam_search=1

        try:
            model.sess.close()
            tf.reset_default_graph()
        except:
            pass





        ##########################
        if do_hyperparam_search:

            print('search')

            # Perform randomized hyperparam search, validated via cross-val
            best_loss = float('inf')
            best_params = None

            # best_params={'recurrent_unit': modeltype, 'recurrent_dropout': 0, 'optimizer': 'adagrad', 'hidden_layer_size': 128, 'dropout': 0.2, 'activation': 'relu'}

            if not best_params:

                # Try a set of parameters
                for sample_num, params in enumerate(ParameterSampler(param_grid, n_iter=30)):
                    print(sample_num)
                    print(params)

                    if 'epochs' in params:
                        epochs = params['epochs']
                        del params['epochs']
                    else:
                        epochs = 5



                    # How well do these parameters do in cross-validation?
                    val_losses = []
                    model=create_rnn(timesteps, n_features, **params)
                    for num_fold,f in enumerate(folds):
                        # train on (n-1) folds and eval on the held-out fold
                        model=create_rnn(timesteps, n_features, **params)
                        try:
                            raise Exception("keras model not implemented")
                            tb_callback=keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
                            history = model.fit(f['x_train'], f['y_train'], validation_data=(f['x_test'], f['y_test']), epochs=epochs, batch_size=64, callbacks=[tb_callback]) #verbose=0
                            val_loss = history.history['val_loss'][-1]
                        except:
                            assert len(f['y_test'])>0
                            history = model.fit(f['x_train'], f['y_train'], validation_data=(f['x_test'], f['y_test']), epochs=epochs, batch_size=64) #verbose=0
                            val_loss=history

                        #optionally compute AUC for best params


                        # if using tf model
                        model.sess.close()
                        tf.reset_default_graph()

                        try:
                            if K.backend == 'tensorflow':
                                K.clear_session()
                        except:
                            pass

                        print(val_loss)

                        # if nan loss, then abandon hope on this set
                        if np.isnan(val_loss):
                            val_losses = [float('inf')]
                            break

                        val_losses.append(val_loss)

                        # if you're more than an order of magnitude more than the best, then stop wasting our optimization time
                        if np.mean(val_losses) > 10*best_loss:
                            break
                        if num_fold>1 and min(val_losses) > 3*best_loss:
                            break

                    avg_heldout_loss = np.mean(val_losses)
                    print('loss:', avg_heldout_loss)

                    # Is this the best heldout performance so far?
                    if avg_heldout_loss < best_loss:
                        print('NEW BEST:', avg_heldout_loss)
                        best_loss = avg_heldout_loss
                        best_params = params
                        best_params['epochs']=epochs
                    print('\n\n\n')
                    try:
                        del model
                        tf.reset_default_graph()
                    except:
                        pass


            print(best_loss)
            print(best_params)
        ##########################

        ##########################
        # Create best model

        print('\n\nMaking best model', best_params)

        if 'epochs' in best_params:
            epochs = best_params['epochs']
            del best_params['epochs']
        else:
            epochs = 5


        # build model with best hyperparams
        model = create_rnn(timesteps, n_features, **best_params)
        model.fit(X, y, epochs=epochs, batch_size=64)
        ##########################

        best_params['epochs']=epochs

        return model

    else:
        # Not a time series

        if modeltype == 'rf':
            model=RandomForestClassifier(random_state=random_state, n_jobs=-1)

            class_weight=[None, 'balanced_subsample']
            n_estimators = [500] #[50, 100, 500, 1000] #[int(x) for x in np.linspace(start = 50, stop = 2000, num = 20)]
            # max_features = ['auto'] #['auto', 'log2']
            # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            # max_depth.append(None)
            # min_samples_split = [2,3, 5, 7, 10]
            min_samples_leaf = [1, 5, 10, 50, 100]
            # bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,
                        'class_weight': class_weight,
                    #    'max_features': max_features,
                    #    'max_depth': max_depth,
                    #    'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                    #    'bootstrap': bootstrap
                    }
        elif modeltype == 'xg':
            model=xgb.XGBClassifier(objective='binary:logistic', nthread=-1, seed=random_state)

            booster = ['gbtree', 'gblinear'] # specify which booster to use: gbtree, gblinear or dart.
            n_estimators = [100]
            learning_rate = [0, 0.1, 0.3, 0.7, 1] # Boosting learning rate (xgb’s “eta”) [0,1]
            min_split_loss = [0, 10, 100] # >= 0
            max_depth = [6, 10] # >=0
            max_delta_step = [0, 5, 10] #it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
            reg_lambda = [0.1, 1, 10] # L2 regularization term on weights
            reg_alpha = [0, 0.1, 10] # L1
            tree_method = ['auto', 'hist'] #Specify which tree method to use. Default to auto.
            # scale_pos_label = [] # set during cross-validation

            random_grid = {'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'min_split_loss': min_split_loss,
                        'max_delta_step': max_delta_step,
                        'max_delta_step': max_delta_step,
                        'reg_lambda': reg_lambda,
                        'reg_alpha': reg_alpha,
                        'tree_method': tree_method,
                        'booster': booster
                    }
        
        elif modeltype == 'iforest':
            ## outlier detection
            model=IsolationForest(random_state=random_state)

            n_estimators = [50, 100, 500, 1000] #[int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)]
            max_features = [0.3, 0.5, 0.7]
            contamination=['auto']
            behaviour=["new"]
            random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'contamination': contamination,
                       'behaviour': behaviour}

        elif modeltype=='lr':
            print('Logistic Regression model!!')
            model=LogisticRegression(random_state=random_state)

            class_weight=[None, 'balanced']
            C_values = np.logspace(-3, 2, 6)
            penalties = ['l2', 'l1']
            tol_values = np.logspace(-4, 0, 3)
            solvers = ['liblinear','saga']
            max_iter = [1e2, 1e3]
            # warm_starts = [True, False]

            random_grid = {'C':C_values,
                           'class_weight': class_weight,
                           'penalty':penalties,
                           'tol':tol_values,
                           'solver':solvers,
                           'max_iter':max_iter,
                        #    'warm_start':warm_starts
                          }

        elif modeltype == 'svm':
            model=LinearSVC()

            C_values = np.linspace(.001, 100, num = 3)
            penalties = ['l2', 'l1']
            # tol_values = np.linspace(.0001, 1, num = 11)
            tol_values = np.logspace(-4, 0, num = 5)
            loss = ['hinge','squared_hinge']
            multi_class = ['ovr', 'crammer_singer']

            random_grid = {'C':C_values,
                           'penalty':penalties,
                           'tol':tol_values,
                           'multi_class':multi_class,
                           'loss':loss
                          }

        elif modeltype == 'rbf-svm':
            model = SVC(probability=True)

            kernel = ['rbf']
            C_range = np.logspace(-2, 2, 5)
            # gamma_range = np.logspace(-9, 3, 13)
            gamma_range = np.logspace(-5, -3, 3) #np.logspace(-5, 1, 7)

            random_grid = {'kernel':kernel,
                           'C':C_range,
                           'gamma':gamma_range
                          }

        elif modeltype == '1class_svm':
            ## outlier detection
            model = OneClassSVM()

            kernel = ['rbf']
            gamma_range = np.logspace(-5, -3, 3)
            # tol_values = np.logspace(-4, 0, num = 5)
            nu_values = [0.05, 0.1, 0.2, 0.5] #np.linspace(0.1, 0.5, 5)    

            random_grid = {'kernel': kernel,
                           'gamma': gamma_range,
                        #    'tol':tol_values,
                           'nu': nu_values
                          }

        elif modeltype == '1class_svm_novel':
            ## Novelty detection
            model = OneClassSVM()

            kernel = ['rbf']
            gamma_range = np.logspace(-5, -3, 3)
            # tol_values = np.logspace(-4, 0, num = 5)
            nu_values = [0.05, 0.1, 0.2, 0.5] #np.linspace(0.1, 0.5, 5)    

            random_grid = {'kernel': kernel,
                           'gamma': gamma_range,
                        #    'tol':tol_values,
                           'nu': nu_values
                          }

            ## filter X and y to exclude outliers, i.e. exclude y=1
            X = X.loc[y==0,:]
            y = y[y==0]


        elif modeltype == 'mlp':
            model = MLPClassifier(random_state=random_state)

            hidden_layer_sizes=[(100,), (100,50), (500,100,50)]
            activation=['tanh', 'relu']
            alpha=[1e-4, 1e-3, 1e-2]
            learning_rate_init=[0.01, 0.001]

            default_grid={
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'alpha': alpha,
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate_init':learning_rate_init
                }

            sgd_grid_1={
                'solver': ['sgd'],
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'alpha': alpha,
                'learning_rate_init':learning_rate_init,
                'max_iter':[1000],
                'learning_rate': ['constant', 'adaptive'],
                'momentum': [0.1, 0.5, 0.9],
                'nesterovs_momentum':[True, False],
                'early_stopping': [True],
                }
            sgd_grid_2={
                'solver': ['sgd'],
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'alpha': alpha,
                'learning_rate_init':learning_rate_init,
                'max_iter':[1000],
                'learning_rate': ['invscaling'],
                'power_t':[0.1, 0.5, 0.9],
                'momentum': [0.1, 0.5, 0.9],
                'nesterovs_momentum':[True, False],
                'early_stopping': [True],
                }

            adam_grid={
                'solver': ['adam'],
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': activation,
                'alpha': alpha,
                'learning_rate_init':learning_rate_init,
                'max_iter': [1000],
                'beta_1': [0.5, 0.9],
                'beta_2': [.7, 0.999],
                'early_stopping': [True]
                }
            
            random_grid = [default_grid, sgd_grid_1, sgd_grid_2, adam_grid]

        elif modeltype == 'knn':
            model = KNeighborsClassifier()

            weights = ['uniform', 'distance']
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

            random_grid = {'weights':weights,
                           'algorithm':algorithm
                          }

        elif modeltype == 'nb':
            model = GaussianNB()
            
            smoothing_vals = np.logspace(-9, 0, 4)
            random_grid = {"var_smoothing": smoothing_vals}

        elif modeltype == 'mlp_torch':
            if feature_selection:
                k = feature_selection_args['K'][0]
                select_k = SelectKBest(f_classif, k=k)
                X_new = select_k.fit_transform(X, y)
                selected_features_mask = select_k.get_support()
            else:
                X_new = X.copy().values
                selected_features_mask = None

            mlp_model = [MLP_large, MLP_small]
            weight_decay = [1e-1, 1e-2]
            lr = [1e-3]
            auc_plateau_max = [10]
            random_grid = {'mlp_model':mlp_model,
                            'weight_decay':weight_decay,
                            'lr':lr,
                            'auc_plateau_max':auc_plateau_max                          
                          }
            param_grid = sklearn.model_selection.ParameterGrid(random_grid)
            best_val_auc = 0
            for params in param_grid:
                model, val_auc = mlp_torch(X_new, y, params, batch_size=64, epochs=30, n_iter=300, logdir=logdir)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model = model
                    best_params = params
            return best_model, None, selected_features_mask

        else:
            raise Exception('modeltype = "%s" is invalid' % modeltype)

        
        if modeltype in ['1class_svm', 'iforest', '1class_svm_novel']:
            ## auroc doesn't work for one-class models
            scoring = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, pos_label=-1)
            y[y==1] = -1
            y[y==0] = 1
            refit_score=True
        else:
            scoring={'AUC':'roc_auc', 
                    'APR':'average_precision', 
                    'ECE': make_scorer(get_ECE, greater_is_better=False, needs_proba=True),
                    'MCE': make_scorer(get_MCE, greater_is_better=False, needs_proba=True)
                    }
            refit_score='AUC'

        # params_grid to randomly chosen iterable
        random_grid = list(ParameterGrid(random_grid))
        np.random.seed(random_state)
        random_grid = np.random.choice(random_grid, size=min(len(random_grid), randomSearchCVargs['n_iter']), replace=False)

        # get target labels
        hadm_ids = y.index.get_level_values("hadm_id")

        kfold = StratifiedKFold(n_splits=randomSearchCVargs['cv'])

        preds, params = {}, {}
        for iter_i, param_dict in enumerate(random_grid):
            print('\n', '#'*60)
            print(f'iteration {iter_i}')
            print(param_dict)
            params[iter_i] = param_dict
            write_to_disc(params, output_dir, f'{target}_{modeltype}_{data_preprocessing_mode}_params.pkl', mode='wb')

            preds[iter_i] = {}
            for cv_i, (train_index, test_index) in enumerate(kfold.split(hadm_ids, y.values)):
                print('\n', '*'*40)
                print(f'iteration {iter_i}, cv {cv_i}')
                train_hadm_ids, test_hadm_ids = hadm_ids[train_index], hadm_ids[test_index]
                X_train = X.loc[X.index.get_level_values("hadm_id").isin(train_hadm_ids), :]
                X_test = X.loc[X.index.get_level_values("hadm_id").isin(test_hadm_ids), :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                print('data preprocessing ...')
                X_train, _, X_test, _,_ = data_preprocessing(X_train, X_test, target, 
                                                            mode=data_preprocessing_mode, missing_rate_cutoff=0.8)
                assert((X_train.index.get_level_values("hadm_id") == y_train.index.get_level_values("hadm_id")).all())
                assert((X_test.index.get_level_values("hadm_id") == y_test.index.get_level_values("hadm_id")).all())
                print(f'X_train shape:{X_train.shape}, X_test shape:{X_test.shape}')
                print(f'mean_y_train={np.mean(y_train):5.3f}, mean_y_test={np.mean(y_test):5.3f}')

                if modeltype=='xg': # add hyperparameter for class imbalance
                    ratio = float(np.sum(y_train == 0)) / np.sum(y_train==1)
                    param_dict['scale_pos_weight'] = ratio
                    print(f'scale_pos_weight parameter set to: {ratio:10.5f}')
                model.set_params(**param_dict)
                t0=time.time()
                model.fit(X_train.values, y_train.values)
                t1=time.time()
                print(f'training finished in {(t1-t0):10.2f}seconds')
                y_pred = model.predict_proba(X_test.values)[:,1]
                preds[iter_i][cv_i] = (y_test, y_pred)
                write_to_disc(preds, output_dir, f'{target}_{modeltype}_{data_preprocessing_mode}_train_predictions.pkl', mode='wb')

        return preds, params

        # model.set_params(**best_params)
        # k=selected_features_mask.sum() if feature_selection else None
        # select_k=SelectKBest(f_classif, k=k)
        # y_pred_prob=np.zeros(len(y))
        # for train_index, test_index in kfold.split(X, y):
        #     X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        #     y_train = y[train_index]
        #     if feature_selection:
        #         X_train=select_k.fit_transform(X_train, y_train)
        #         X_test=select_k.transform(X_test)
        #     model.fit(X_train, y_train)            
        #     _, y_pred_prob[test_index], _=get_prediction(modeltype, model, X_test, y=None, subject_id=None)

        # return trained_model  #, y_pred_prob, selected_features_mask

def invert_dict(d):
    """
    invert the dictionary so that the resulting keys contain lists of all previous keys
    Input: 
        d (dict)
    Returns:
        (dict)
    """
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse

def get_score(y_true, y_pred, metric, modeltype, threshold=0.5):
    score_dict = get_measures(y_true, y_pred, modeltype, threshold)
    if metric in score_dict.keys():
        return score_dict[metric]
    else:
        raise Exception(f'{metric} not supported!')
    
def get_best_model(modeltype, opt_metric, params, preds):
    n_iter = len(preds.keys())
    scores=np.zeros(n_iter)
    for iter_i in range(n_iter):
        n_cv = len(preds[iter_i].keys())
        for cv_i in range(n_cv):
            y_true, y_pred = preds[iter_i][cv_i]
            scores[iter_i] += get_score(y_true, y_pred, opt_metric, modeltype)
        scores[iter_i] /= n_cv
    if opt_metric in ['BRIER', 'ECE', 'MCE', 'CE', 'ECE2']:
        best_index = np.argmin(scores)
    else:
        best_index = np.argmax(scores)       
    best_param = params[best_index]
    print(f'Best model selected with {opt_metric}={scores[best_index]:5.3f}')
    if modeltype=='lr':
        model = LogisticRegression()
    elif modeltype=='rf':
        model = RandomForestClassifier(random_state=0, n_jobs=-1)
    elif modeltype=='xg':
        model = xgb.XGBClassifier(objective='binary:logistic', nthread=-1, seed=0)
    elif modeltype=='mlp':
        model = MLPClassifier(random_state=0)
    else:
        raise Exception(f'{modeltype} not supported yet!')
    return model.set_params(**best_param)

def fit_model(model, X, y):
    model.fit(X,y)
    return model

def predict(model, X):
    return model.predict_proba(X)[:,1]

def write_to_disc(data, out_dir, filename, mode='wb'):
    with open(os.path.join(out_dir, filename), mode) as f:
        pickle.dump(data, f)
    return

def shift_and_impute_temporal_data(X_t, shift_featues, shift_values):
    X1 = X_t.copy()
    X2 = shift_temporal_data(X_t, shift_featues, shift_values)
    X1_mean=X1.mean()
    X1 = X1.groupby('hadm_id').apply(
            lambda g: g.fillna(method='ffill').fillna(method='bfill').mean())\
                .fillna(X1_mean)
    X2 = X2.groupby('hadm_id').apply(
            lambda g: g.fillna(method='ffill').fillna(method='bfill').mean())\
                .fillna(X1_mean)
        
    return X1,X2

def shift_temporal_data(X, features, shift_values):
    idx=pd.IndexSlice
    Y = X.copy()
    Y.loc[:, idx[features, 'mean']] += shift_values
    return Y


def predict_and_measure(modeltype, model, X_test, y_test):
    y_pred = predict(model, X_test.values)
    measures = get_measures(y_test, y_pred, modeltype, threshold=0.5)
    return measures,y_pred

def fit_best_model(X, y, modeltype, train_params, train_preds, opt_metric):
    model = get_best_model(modeltype, opt_metric, train_params, train_preds)
    model = fit_model(model, X.values, y.values)
    return model

def get_train_params_and_preds(target, data_preprocessing_mode, output_dir, modeltype):
    with open(os.path.join(output_dir, f'{target}_{modeltype}_{data_preprocessing_mode}_params.pkl'), 'rb') as f:
        train_params = pickle.load(f)
    with open(os.path.join(output_dir, f'{target}_{modeltype}_{data_preprocessing_mode}_train_predictions.pkl'), 'rb') as f:
        train_preds = pickle.load(f)
    return train_params,train_preds

def get_processed_data(years, target, data_preprocessing_mode, df):
    X, y = get_temporal_data(years, target, df)

    print(f'Processing train data {years}...')
    X, y, _, _,transform_params = data_preprocessing(X, None, target, mode=data_preprocessing_mode)
    return X,y, transform_params

def get_temporal_data(years, target, df):
    global demo_df
    global label_df
    # train indices
    mask=demo_df['intime'].dt.year.isin(years)
    hadm_ids=demo_df.index.get_level_values('hadm_id')[mask]
    X = df.loc[df.index.get_level_values('hadm_id').isin(hadm_ids), :]
    y = label_df.loc[label_df.index.get_level_values('hadm_id').isin(hadm_ids), target]

    print(X.shape)
    print(y.value_counts())
    return X,y

def measure_distribution_distance(dist_metrics, train_years, test_years, target, max_time, data_preprocessing_mode, output_dir):
    global filtered_df
    global label_df
    global demo_df
    
    idx = pd.IndexSlice
    print('filtering data hours ...')
    df = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                            (filtered_df.index.get_level_values('hours_in') <= max_time)]
    print('Done.')    

    # train indices
    mask=demo_df['intime'].dt.year.isin(train_years)
    train_hadm_ids=demo_df.index.get_level_values('hadm_id')[mask]
    X_train = df.loc[df.index.get_level_values('hadm_id').isin(train_hadm_ids), :]
    y_train = label_df.loc[label_df.index.get_level_values('hadm_id').isin(train_hadm_ids), target]

    print(X_train.shape)
    print(y_train.value_counts())

    print(f'Processing train data {train_years}...')
    X_train, _, _, _, _ = data_preprocessing(X_train, None, target, mode=data_preprocessing_mode)


    measures = {}
    for dist_metric in dist_metrics:
        print(f'\n using {dist_metric} metric ...')
        measures[dist_metric] = {}
        for year in test_years:
            print(f'\t get test data, year={year} ...')
            X_test, y_test = get_and_preprocess_test_data(target, data_preprocessing_mode, idx, df, year)
            print(f'\t X_test.shape: {X_train.shape}')
            
            print(f'\t compute {dist_metric} measure ...')
            t0=time.time()
            stat, pvalue = get_shift_measures(dist_metric, X_train.values, y_train.values, X_test.values, y_test.values)
            t=time.time()-t0
            measures[dist_metric][year] = (stat, pvalue, t)
            print(f'\t distance computed in {t:5.2f} seconds.')

            write_to_disc(measures, output_dir, f'{target}_{dist_metric}_{data_preprocessing_mode}_measurements.pkl', mode='wb')

    return

def get_shift_measures(shift_metric, X1, y1, X2, y2):
    # print(f'shift_metric: {shift_metric}')
    stat, pvalue = None, None
    if shift_metric=='MMD':
        # if X1.shape[1] <= 20 and X1.shape[0] > 16e3:
        #     print('calculating chi2_approx ...')
        #     stat, pvalue = chi2_approx(MMD().statistic, X1, X2)
        #     # from scipy.stats.distributions import chi2
        #     # n = X1.shape[0]
        #     # stat = MMD.statistic(x=X1, y=X2)
        #     # pvalue = chi2.sf(stat * n + 1, 1)
        # else:
        stat, pvalue = MMD().test(X1, X2, workers=-1, auto=True, random_state=0)
    elif shift_metric=='MMD_rbf_tl':
        stat = mmd_rbf(X1, X2, gamma=1.0)
    elif shift_metric=='MMD_alibi':
        from alibi_detect.cd import MMDDrift
        cd = MMDDrift(X1, backend='tensorflow', p_val=.05)
        stat = cd.predict(X2, return_p_val=True, return_distance=True)
    elif shift_metric=='WD':
        M = ot.dist(X1, X2, 'euclidean')
        M /= M.max()
        a, b = np.ones((X1.shape[0],)) / X1.shape[0], np.ones((X2.shape[0],)) / X2.shape[0]  # uniform distribution on samples
        stat = ot.emd2(a.astype(np.float64), b.astype(np.float64), M.astype(np.float64), numItermax=int(1e7), numThreads="max")
    elif shift_metric=='summary_stats':
        abs_mean_diff= np.abs(np.mean(X1, axis=0) - np.mean(X1, axis=0))
        abs_med_diff= np.abs(np.median(X1, axis=0) - np.median(X1, axis=0))
        abs_std_diff= np.abs(np.std(X1, axis=0) - np.std(X1, axis=0))
        stat=(abs_mean_diff, abs_med_diff, abs_std_diff)
    elif shift_metric=='euclidean_distance':
        stat = np.sqrt(np.sum(np.square(X1-X2), axis=0))
    elif shift_metric=='univariate_ttest':
        stat, pvalue = stats.ttest_ind(X1, X2)
    return stat, pvalue

def mmd_rbf(X, Y, gamma=1.0):
    """
    Implementation from: 
    https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = sklearn.metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = sklearn.metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = sklearn.metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def get_and_preprocess_test_data(target, data_preprocessing_mode, df, year):
    X_test = get_test_data(df, year)
    print(f'\t Processing test data {year}...')
    X_test, y_test = preprocess_test_data(target, data_preprocessing_mode, X_test)
    return X_test, y_test

def preprocess_test_data(target, data_preprocessing_mode, X_test, transform_params):
    _, _, X_test, y_test,_ = data_preprocessing(None, X_test, target, mode=data_preprocessing_mode, 
                                transform_params=transform_params)
    assert((X_test.index.get_level_values("hadm_id") == y_test.index.get_level_values("hadm_id")).all())
    return X_test,y_test

def get_test_data(df, year):
    idx = pd.IndexSlice
    mask=demo_df.loc[:, idx['intime', 'intime','intime']].dt.year == year
    hadm_ids=demo_df.index.get_level_values('hadm_id')[mask]
    X_test = df.loc[df.index.get_level_values('hadm_id').isin(hadm_ids), :]
    return X_test

# # def main

def main_generate_predictions(modeltypes, metrics, train_years, test_years, target, max_time, data_preprocessing_mode, output_dir):
    global filtered_df
    global label_df
    global demo_df
    
    idx = pd.IndexSlice
    print('filtering data hours ...')
    df = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                            (filtered_df.index.get_level_values('hours_in') <= max_time)]
    print('Done.')    

    train_df, train_y = get_processed_data(train_years, target, data_preprocessing_mode, df)

    for modeltype in modeltypes:
        print('#'*50 + f'\n modeltype: {modeltype}\n')

        train_params, train_preds = get_train_params_and_preds(target, data_preprocessing_mode, output_dir, modeltype)
        preds, measures = {}, {}
        for opt_metric in metrics:
            print(f'\n optimizing model based on {opt_metric} ...')
            preds[opt_metric], measures[opt_metric] = {}, {}
            model = fit_best_model(train_df, train_y, modeltype, train_params, train_preds, opt_metric)
            for year in test_years:
                X_test, y_test = get_and_preprocess_test_data(target, data_preprocessing_mode, df, year)
            
                print(f'applying best model to test dataset {year}...')
                measures, y_pred = predict_and_measure(modeltype, model, X_test, y_test)
                preds[opt_metric][year] = (y_test, y_pred)
                measures[opt_metric][year] = measures

                write_to_disc(preds, output_dir, f'{target}_{modeltype}_{data_preprocessing_mode}_test_predictions.pkl', mode='wb')
                write_to_disc(measures, output_dir, f'{target}_{modeltype}_{data_preprocessing_mode}_test_measures.pkl', mode='wb')

    return


def main_gradual_shift(shift_params, modeltypes, opt_metrics, years, target, max_time, 
                        data_preprocessing_mode, in_dir, output_dir, compute_model_performance=True):
    global filtered_df
    global label_df
    global demo_df
    
    idx = pd.IndexSlice
    print('filtering data hours ...')
    df = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                            (filtered_df.index.get_level_values('hours_in') <= max_time)]
    print('Done.')

    print('read temporal data ...')
    X_t, y_t = get_temporal_data(years, target, df)
    print(f'X_t: {X_t.shape}')

    print('\nread non-temporal data ...')
    X, y, transform_params = get_processed_data(years, target, data_preprocessing_mode, df)
    print(f'X: {X.shape}')

    print('\nfilter temporal data ...')
    # hamids=X.index.get_level_values("hadm_id").unique()
    features=X.columns.get_level_values(2).unique()
    X_t = X_t.loc[:, idx[features, 'mean']] #.droplevel(1, axis=1)
    print(X_t.shape)

    # # COMMENT THIS (for test purposes)
    # X_t, y_t = X_t.iloc[0:1000], y_t.iloc[0:1000]
    # X, y = X.iloc[0:1000], y.iloc[0:1000]
    # # COMMENT THIS (for test purposes)


    shift_metrics = shift_params['metrics']
    shift_features = shift_params["features"]
    base_shift_values = np.array(shift_params["shift_values"])
    shift_measures={m:[] for m in shift_metrics}
    shift_measures.update({m+'_temporal_features':[] for m in shift_metrics})
    
    preds = {mt:{} for mt in modeltypes}
    for mt in modeltypes:
        preds[mt] = {om:[] for om in opt_metrics}
    perf_measures = {mt:{} for mt in modeltypes}
    for mt in modeltypes:
        perf_measures[mt] = {om:[] for om in opt_metrics}

    output_prefix=f'{target}_{data_preprocessing_mode}_{"_".join(shift_metrics)}_{"_".join(shift_features)}_{"_".join([str(v) for v in base_shift_values])}_num_iter_{shift_params["num_iterations"]}'
    
    shift_measures_output_filename = output_prefix + '_shift_measures.pkl'
    predictions_output_filename = output_prefix + '_predictions.pkl'
    performance_measures_output_filename = output_prefix + '_performance_measures.pkl'

    
    for i in range(shift_params["num_iterations"]+1):
        shift_values = i * base_shift_values
        print(f'\n{"#"*50}\nIteration: {i}, shift_features={shift_features}, shift_values={shift_values}\n')
        
    
    ### simulate and measure shift for temporal data
    ### commented the code below to save time

    #     print(f'measure shift for temporal data ...')
    #     ## measure shift in temporal data
    #     X1, X2 = shift_and_impute_temporal_data(X_t, shift_features, shift_values)
    #     print(f'X1: {X1.shape}, X2: {X2.shape}')

    #     for m in shift_metrics:
    #         print(f'\t metric: {m}')
    #         t0=time.time()
    #         s, p = get_shift_measures(m, X1.values, None, X2.values, None)
    #         t1=time.time()
    #         print(f'\tcompute time={(t1-t0):5.2f} seconds.')
    #         #print(f'\tdivergence={s:.5f}, p={p}')
    #         shift_measures[m+'_temporal_features'].append((s, p))

    #     write_to_disc(shift_measures, output_dir, shift_measures_output_filename, mode='wb')

        ## simualte and measure shift for NON-temporal data

        print(f'measure shift for non-temporal data ...')
        X2 = shift_temporal_data(X_t, shift_features, shift_values)
        X2, y2 = preprocess_test_data(target, data_preprocessing_mode, X2, transform_params)
        print(f'\nX2: {X2.shape}')
        
        print(f'computing shift measures {shift_params["metrics"]} for iter {i} ...')
        for m in shift_metrics:
            print(f'\t metric: {m}')
            t0=time.time()
            stat, pval = get_shift_measures(m, X.values, y.values, X2.values, y2.values)
            t1=time.time()
            print(f'\tcompute time={(t1-t0):5.2f} seconds.')
            print(f'\t{m}={stat}, pvalue={pval}')
            shift_measures[m].append((stat, pval))
        write_to_disc(shift_measures, output_dir, shift_measures_output_filename, mode='wb')

        if not compute_model_performance:
            continue
        
        ## measure performance of saved-models on shifted data (non-temporal)

        print(f'\nMeasure model performance on shifted data ...')
        for modeltype in modeltypes:
            print(f'\n modeltype: {modeltype}\n')
            train_params, train_preds = get_train_params_and_preds(target, data_preprocessing_mode, in_dir, modeltype)
            
            ## first select best model based on each given opt_metric
            for opt_metric in opt_metrics:
                print(f'\t optimizing model based on {opt_metric} ...')
                model = fit_best_model(X, y, modeltype, train_params, train_preds, opt_metric)

                print(f'\t applying best model to test dataset, iter: {i}...')
                measures, y_pred = predict_and_measure(modeltype, model, X2, y2)
                preds[modeltype][opt_metric].append((y2, y_pred))
                perf_measures[modeltype][opt_metric].append(measures)

                write_to_disc(preds, output_dir, predictions_output_filename, mode='wb')
                write_to_disc(perf_measures, output_dir, performance_measures_output_filename, mode='wb')

    return

def main_explain_gradual_shift(explanation_size, modeltype, shift_params, opt_metric, 
                            target, years, max_time, data_preprocessing_mode, in_dir, output_dir, random_state=0):
    global filtered_df
    global label_df
    global demo_df
    
    idx = pd.IndexSlice
    print('filtering data hours ...')
    df = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                            (filtered_df.index.get_level_values('hours_in') <= max_time)]
    print('Done.')

    print('read temporal data ...')
    X_t, y_t = get_temporal_data(years, target, df)
    print(f'X_t: {X_t.shape}')

    print('\nread non-temporal data ...')
    X, y, transform_params = get_processed_data(years, target, data_preprocessing_mode, df)
    print(f'X: {X.shape}')

    print('\nfilter temporal data ...')
    # hamids=X.index.get_level_values("hadm_id").unique()
    features=X.columns.get_level_values(2).unique()
    X_t = X_t.loc[:, idx[features, 'mean']] #.droplevel(1, axis=1)
    print(X_t.shape)

    # # COMMENT THIS
    # X_t, y_t = X_t.iloc[0:1000], y_t.iloc[0:1000]
    # X, y = X.iloc[0:1000], y.iloc[0:1000]
    # # COMMENT THIS

    print(f'\n modeltype: {modeltype}\n')
    train_params, train_preds = get_train_params_and_preds(target, data_preprocessing_mode, in_dir, modeltype)
    print(f'\t optimizing model based on {opt_metric} ...')
    model = fit_best_model(X, y, modeltype, train_params, train_preds, opt_metric)

    np.random.seed(random_state)
    exp_samples=np.random.randint(0, X.shape[0], size=explanation_size)
    bg_samples=np.arange(X.shape[0])[~np.isin(np.arange(X.shape[0]), exp_samples)]
    explainer=shap.TreeExplainer(model=model, data=X.values[bg_samples,:],
                                model_output="probability", feature_perturbation = "interventional")

    shift_features = shift_params["features"]
    base_shift_values = np.array(shift_params["shift_values"])
    
    explain_dict={}
    for i in range(shift_params["num_iterations"]+1):
        shift_values = i * base_shift_values
        print(f'\n{"#"*50}\nIteration: {i}, shift_features={shift_features}, shift_values={shift_values}\n')
        
        ## measure shift for NON-temporal data
        print(f'extract non-temporal data ...')
        X2 = shift_temporal_data(X_t, shift_features, shift_values)
        X2, y2 = preprocess_test_data(target, data_preprocessing_mode, X2, transform_params)
        print(f'\nX2: {X2.shape}')
        
        print(f'generate SHAP values. iter {i} ...')
        shap_values = explainer(X2.values[exp_samples,:])
        shap_values.feature_names=['_'.join([str(i),str(j),str(k)]) for i,j,k in X2.columns]
        explain_dict[i] = shap_values
        
        write_to_disc(explain_dict, output_dir,
                    f'{target}_{modeltype}_{data_preprocessing_mode}_shift_{"_".join(shift_features)}_shap_values.pkl', mode='wb')

    return


def main_overtime_overall(random_state=None, max_time=24, target='mort_icu', modeltype='rf',  opt_metric='AUROC', data_preprocessing_mode='basic',
                          train_years=[], test_years=[], data_dir="", output_dir="", randomSearchCVargs={}, **feature_selection_args):
    """
    This function trains data from training_years hidenic and tests on data after training_years, once every test_month_interval month.

    """
    global filtered_df
    global label_df
    global demo_df
 
    np.random.seed(random_state)

    # if save_data:
    #     data_out=os.path.join(data_dir, "preprocessed_data/", prefix + "overall-overtime-style_{}_{}_{}_Simple_seed_{}_target={}.h5".format(
    #                         "-".join([str(i) for i in training_years]), "feature-selection" if feature_selection else "", representation, str(random_seed), str(target)))

     # hours for windowing
    idx = pd.IndexSlice
    print('filtering data hours ...')
    df = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                              (filtered_df.index.get_level_values('hours_in') <= max_time)]
    print('Done.')    


    # train indices
    mask=demo_df['intime'].dt.year.isin(train_years)
    train_hadm_ids=demo_df.index.get_level_values('hadm_id')[mask]
    
    train_df = df.loc[df.index.get_level_values('hadm_id').isin(train_hadm_ids), :]
    train_y = label_df.loc[label_df.index.get_level_values('hadm_id').isin(train_hadm_ids), target]

    print(train_df.shape)
    print(train_y.value_counts())

    print('model selection ....')
    t0 = time.time()
    preds, params = classifier_select(train_df, train_y, target, randomSearchCVargs=randomSearchCVargs, output_dir=output_dir,
                                    is_time_series=False, subject_index=None, modeltype=modeltype, random_state=random_state,
                                    data_preprocessing_mode=data_preprocessing_mode)
    t1 = time.time()
    print(f'model selection finished in {(t1-t0):5.2f} seconds.')

    # write_to_disc(preds, output_dir, f'{modeltype}_train_predictions.pkl')
    # write_to_disc(params, output_dir, f'{modeltype}_params.pkl')

    return


def main_rolling(random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix='', model_types=['rf'], data_dir="", output_dir=""):
    """
    This function trains data model on all previous data.
    """
    global filtered_df
    global label_df
    global years_df
    global train_means

    np.random.seed(random_seed)



    # (Amin)
    # torch.manual_seed(random_seed)



    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                              (filtered_df.index.get_level_values('hours_in') <= max_time)]
    #drop hours in row
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    # print("Loading the  years.")
    # read_years_data()


    ## from 2010 onwards
    years_set=set(years_df['year'][years_df['year'] >= 2010].values.tolist())
  

    # dump_filename=os.path.join(prefix, "AUC_Rolling-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.txt".format(modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target)))
    for modeltype in model_types:

        print('#'*100)
        print('model_type= ', modeltype)
        print('#'*100)

        is_time_series = modeltype in ['lstm', 'gru', 'grud']
   
        dump_filename=os.path.join(output_dir, prefix+"result_Rolling-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.txt".format(
            modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target)))
        with open(dump_filename, 'w') as f:
            for year_train in sorted(years_set):
                if year_train+1 not in years_set:
                    continue
                #train on years less than or equal to this year
                training_years=[item  for item in years_set if item <= year_train]
                year_index=years_df[years_df['year'].isin(training_years)].index.tolist()

                X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level, 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'))


                # X_df, y=data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], 'LEVEL2'.encode(), 'Simple', 'mort_icu')

                try:
                    model.fit(X_df, np.asarray(y).ravel())
                except:
                    print("Finding the best {} model using a random search".format(modeltype.upper()))
                    print(X_df.shape)
                    model = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype)


                # open the trained model and test on years 2003 onwards
                year_index=years_df[years_df['year'].isin([year_train+1])].index.tolist()
                # get the X and y data for testing
                X_df, y, _, _, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level, 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'), timeseries_vect=timeseries_vect, representation_vect=representation_vect)



                # Different models have different score funcions
                if modeltype in ['lstm','gru']:
                    y_pred_prob=model.predict(np.swapaxes(X_df, 1,2))
                    # y_pred_prob=model.predict(X_df)
                    pred = list(map(int, y_pred_prob > 0.5))
                elif modeltype=='grud':
                    #create test_dataloader X_df, demo_df
                    test_dataloader=PrepareDataset(X_df, y, subject_id, train_means, BATCH_SIZE = 1, seq_len = 25, ethnicity_gender=True, shuffle=False)

                    predictions, labels, _, _ = predict_GRUD(model, test_dataloader)
                    y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
                    y=np.squeeze(np.asarray(labels))
                    pred=np.argmax(np.squeeze(predictions), axis=1)
                    # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
                elif modeltype in ['lr', 'rf', 'mlp', 'knn']:
                    y_pred_prob=model.predict_proba(X_df)[:,1]
                    pred=model.predict(X_df)
                elif modeltype in ['svm', 'rbf-svm']:
                    y_pred_prob=model.decision_function(X_df)
                    pred=model.predict(X_df)
                else:
                    raise Exception('dont know proba function for classifier = "%s"' % modeltype)

                


                AUC=sklearn.metrics.roc_auc_score(y, y_pred_prob)
                F1=sklearn.metrics.f1_score(y, pred)
                ACC=sklearn.metrics.accuracy_score(y, pred)
                APR=sklearn.metrics.average_precision_score(y, y_pred_prob)
                print("year: {}, AUC: {}, APR: {}".format(year_train+1, AUC, APR))
                f.write("year, {}, AUC, {} \r\n".format(str(year_train+1), str(AUC)))
                f.write("year, {}, F1, {} \r\n".format(str(year_train+1), str(F1)))
                f.write("year, {}, Acc, {} \r\n".format(str(year_train+1), str(ACC)))
                f.write("year, {}, APR, {} \r\n".format(str(year_train+1), str(APR)))

                f.write("year, {}, label, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in y])))
                f.write("year, {}, pred, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in pred])))
                f.write("year, {}, y_pred_prob, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in y_pred_prob])))

                f.write("year, {}, gender, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in gender])))
                f.write("year, {}, ethnicity, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in ethnicity])))
                f.write("year, {}, subject, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in subject_id])))

                f.write("year, {}, best_params, <{}> \r\n".format(str(year_train+1), best_params))

                #delete all models

        print("Finished {}".format(dump_filename))
    return


# # def main_rolling_limited

# In[17]:


def main_rolling_limited(random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix='', model_types=['rf'], data_dir="", output_dir=""):
    """
    This function trains model on previous year only.
    """
    global filtered_df
    global label_df
    global years_df
    global train_means

    np.random.seed(random_seed)


    # (Amin)
    # torch.manual_seed(random_seed)



    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) & (filtered_df.index.get_level_values('hours_in') <= max_time)]
    #drop hours in row
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    # print("Loading the  years.")
    # read_years_data()


    ## from 2010 onwards
    years_set=set(years_df['year'][years_df['year'] >= 2010].values.tolist())

    for modeltype in model_types:
        is_time_series = modeltype in ['lstm', 'gru', 'grud']

        dump_filename=os.path.join(output_dir, prefix + "result_Rolling_limited-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.txt".format(modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target)))
        with open(dump_filename, 'w') as f:
            for year_train in sorted(years_set):
                if year_train+1 not in years_set:
                    continue
                #train on this year
                year_index=years_df[years_df['year'].isin([year_train])].index.tolist()

                #split years into 20% test 80% train.
                # train_years, test_years=sklearn.model_selection.train_test_split(year_index,  test_size=0.2, random_state=int(random_seed)+1, stratify=years_df.loc[year_index,:].values.tolist())
                # X_df, y=data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], 'LEVEL2'.encode(), 'Simple', 'mort_icu')

                X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level, 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'))
                print("Finding the best {} model using a random search".format(modeltype))
                model = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype)

                # open the trained model and test on years 2003 onwards
                year_index=years_df[years_df['year'].isin([year_train+1])].index.tolist()
                # get the X and y data for testing
                X_df, y, __, __, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level, 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'), timeseries_vect=timeseries_vect, representation_vect=representation_vect)

                # problem is with 2003 test data
                


                # Different models have different score funcions
                if modeltype in ['lstm','gru']:
                    y_pred_prob=model.predict(np.swapaxes(X_df, 1,2))
                    pred = list(map(int, y_pred_prob > 0.5))
                elif modeltype=='grud':
                    test_dataloader=PrepareDataset(X_df, y, subject_id, train_means, BATCH_SIZE = 64, seq_len = 25, ethnicity_gender=True, shuffle=True)

                    predictions, labels, _, _ = predict_GRUD(model, test_dataloader)

                    try:
                        assert isinstance(predictions, list)
                        predictions=np.concatenate(predictions, axis=0)
                        assert isinstance(labels, list)
                        labels=np.concatenate(labels, axis=0)
                    except:
                        pass


                    y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
                    y=np.squeeze(np.asarray(labels))
                    pred=np.argmax(np.squeeze(predictions), axis=1)
                    # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
                elif modeltype in ['lr', 'rf', 'mlp', 'knn']:
                    y_pred_prob=model.predict_proba(X_df)[:,1]
                    pred=model.predict(X_df)
                elif modeltype in ['svm', 'rbf-svm']:
                    y_pred_prob=model.decision_function(X_df)
                    pred=model.predict(X_df)
                else:
                    raise Exception('dont know proba function for classifier = "%s"' % modeltype)


                AUC=sklearn.metrics.roc_auc_score(y, y_pred_prob)
                F1=sklearn.metrics.f1_score(y, pred)
                ACC=sklearn.metrics.accuracy_score(y, pred)
                APR=sklearn.metrics.average_precision_score(y, y_pred_prob)
                print("year, {}, AUC, {}, APR, {} \r\n".format(str(year_train+1), str(AUC), str(APR)))
                f.write("year, {}, AUC, {} \r\n".format(str(year_train+1), str(AUC)))
                f.write("year, {}, F1, {} \r\n".format(str(year_train+1), str(F1)))
                f.write("year, {}, Acc, {} \r\n".format(str(year_train+1), str(ACC)))
                f.write("year, {}, APR, {} \r\n".format(str(year_train+1), str(APR)))

                f.write("year, {}, label, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in y])))
                f.write("year, {}, pred, <{}>\r\n".format(str(year_train+1), ",".join([str(i) for i in pred])))
                f.write("year, {}, y_pred_prob, <{}>\r\n".format(str(year_train+1), ",".join([str(i) for i in y_pred_prob])))

                f.write("year, {}, gender, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in gender])))
                f.write("year, {}, ethnicity, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in ethnicity])))
                f.write("year, {}, subject, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in subject_id])))

                f.write("year, {}, best_params, <{}> \r\n".format(str(year_train+1), best_params))



        print("Finished {}".format(dump_filename))

    return


# # def main_no_years

# In[18]:


def main_no_years(random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix='', model_types=['rf'], data_dir="", output_dir=""):
    """
    """
    global filtered_df
    global label_df
    global years_df
    global train_means
    global best_params


    n_splits=5

    np.random.seed(random_seed)





    # (Amin)
    # torch.manual_seed(random_seed)





    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) & (filtered_df.index.get_level_values('hours_in') <= max_time)]
    #drop hours in row
    print(filtered_df.head(5))
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)


    # print("data preprocessing")
    # is_time_series=True


    #remove all the overhead data now
    # filtered_df=None
    # label_df=None
    # years_df=None

    # print(filtered_df.head(5))
    kf = sklearn.model_selection.KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
    X_index=filtered_df_time_window.index.tolist()
    # print(filtered_df_time_window.index.names)
    X_index=[item[0] for item in X_index]
    # print(len(X_index))
    X_index=np.asarray(list(set(X_index))).reshape(-1,1)
    print(len(X_index))
    kf.get_n_splits(X_index)

    # input()

    for modeltype in model_types:

        is_time_series = modeltype in ['lstm', 'gru', 'grud']

        split=0

        for train_index, test_index in kf.split(X_index):
            print(split)

            # print(train_index.shape)
            # print(X_index[train_index].shape)

            print(set(list(X_index[train_index].ravel())).intersection(set(list(X_index[test_index].ravel()))))

            # input()

            print("fetching train data")
            X_train, y_train, timeseries_vect, representation_vect, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[X_index[train_index].ravel(), :,:,:],:], level, 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'))
            print("fetching test_data")
            X_test, y_test, __, __, gender, ethnicity, subject_id_test  = data_preprocessing(filtered_df_time_window.loc[idx[X_index[test_index].ravel(),:,:,:],:], level, 'Simple', target, representation, is_time_series, representation_vect=representation_vect, impute=not(modeltype=='grud'))

            print(set(subject_id).intersection(set(subject_id_test)))

            # input()

            if modeltype not in ['lstm', 'grud']:
                assert np.sum(np.sum(np.isnan(X_train.values.astype(np.float32))))<1
                assert np.sum(np.sum(np.isnan(y_train)))<1
                assert np.sum(np.sum(np.isnan(X_test.values.astype(np.float32))))<1
                assert np.sum(np.sum(np.isnan(y_test)))<1

            # X_train, X_test = X_df.loc[[X_index[ind] for ind in train_index],:], X_df.loc[[X_index[ind] for ind in test_index],:]
            # y_train, y_test = y[train_index], y[test_index]

            #find the model
            print("Finding the best %s model using a random search" % modeltype.upper())
            model = classifier_select(X_train, np.asarray(y_train).ravel(), is_time_series, subject_id,  modeltype=modeltype, random_state=split)

            #make a prediction
            if modeltype in ['lstm','gru']:
                y_pred_prob=model.predict(np.swapaxes(X_test, 1, 2))
                pred = list(map(int, y_pred_prob > 0.5))
            elif modeltype=='grud':
                #create test_dataloader X_df, demo_df
                test_dataloader=PrepareDataset(X_test, y_test, subject_id_test, train_means, BATCH_SIZE = 1, seq_len = 25, ethnicity_gender=True, shuffle=False)

                predictions, labels, _, _ = predict_GRUD(model, test_dataloader)
                y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
                y=np.squeeze(np.asarray(labels))
                pred=np.argmax(np.squeeze(predictions), axis=1)
                # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
            elif modeltype in ['lr', 'rf', 'mlp', 'knn']:
                y_pred_prob=model.predict_proba(X_test)[:,1]
                pred=model.predict(X_test)
            elif modeltype in ['svm', 'rbf-svm']:
                y_pred_prob=model.decision_function(X_test)
                pred=model.predict(X_test)
            else:
                raise Exception('dont know proba function for classifier = "%s"' % modeltype)
            AUC=sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
            F1=sklearn.metrics.f1_score(y_test, pred)
            ACC=sklearn.metrics.accuracy_score(y_test, pred)
            APR=sklearn.metrics.average_precision_score(y_test, y_pred_prob)

            dump_filename=os.path.join(output_dir, prefix + "result_no_years-style_{}_{}_Simple_{}_seed-{}_target={}_n_splits-{}.txt".format(
                modeltype.upper(), representation, level, str(random_seed), str(target), str(n_splits)))
            with open(dump_filename, 'a+') as f:
                try:
                    #dev AUC
                    y_pred_prob_=model.predict_proba(X_train)[:,1]
                    pred_=model.predict(X_train)
                    AUC_=sklearn.metrics.roc_auc_score(y_train, y_pred_prob_)
                    f.write("split, {}, dev_A-U-C, {} \r\n".format(str(split), str(AUC_)))

                except:
                    pass
                try:
                    f.write("split, {}, params, {} \r\n".format(str(split), ", ".join([str(k)+":"+str(v) for k, v in best_params.items()])))
                except:
                    print(best_params)
                f.write("split, {}, AUC, {} \r\n".format(str(split), str(AUC)))
                f.write("split, {}, F1, {} \r\n".format(str(split), str(F1)))
                f.write("split, {}, Acc, {} \r\n".format(str(split), str(ACC)))
                f.write("split, {}, APR, {} \r\n".format(str(split), str(APR)))

                f.write("split, {}, label, <{}> \r\n".format(str(split), ",".join([str(i) for i in y_test])))
                f.write("split, {}, pred, <{}>\r\n".format(str(split), ",".join([str(i) for i in pred])))
                f.write("split, {}, y_pred_prob, <{}>\r\n".format(str(split),",".join([str(i) for i in y_pred_prob])))
                f.write("split, {}, gender, <{}> \r\n".format(str(split), ",".join([str(i) for i in gender])))
                f.write("split, {}, ethnicity, <{}> \r\n".format(str(split), ",".join([str(i) for i in ethnicity])))
                f.write("split, {}, subject, <{}> \r\n".format(str(split), ",".join([str(i) for i in subject_id_test])))

            split+=1


    return

def main_hospital_wise(random_seed=None, max_time=24, level='itemid', representation='raw', test_size=0.2,
         target='mort_icu', prefix="", model_types=['rf'],  data_dir="", train_hospitals=[], output_dir="",
         test_hospitals=[], save_data=False, feature_selection=False, **feature_selection_args):
    """
    This function trains data from training_years hidenic and tests on data after training_years, once every test_month_interval month.

    """
    global filtered_df
    global label_df
    global sites_df
    global scaler
    global train_means
    global years_df

    np.random.seed(random_seed)

    # (Amin) not using torch yet
    # torch.manual_seed(random_seed)

    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

    if save_data:
        data_out=os.path.join(data_dir, "preprocessed_data/", prefix + "hospital-style_trainHospitals_{}_{}_{}_Simple_seed_{}_target={}.h5".format(
                "-".join(train_hospitals), "-".join([m.upper() for m in model_types]), representation, str(random_seed), str(target)))
        p = pathlib.Path(data_out)
        if p.is_file():
            raise Exception("File already exists: " + data_out)
            return

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                              (filtered_df.index.get_level_values('hours_in') <= max_time)]

    #drop hours in row
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    source_index = sites_df[sites_df['hospital'].isin(train_hospitals)].index.tolist()
    ## sort index by year and split the last len(index)*test_size for test set
    source_index = years_df.loc[source_index, ['year', 'month']].sort_values(['year', 'month']).index
    ind = int(len(source_index)*(1-test_size))
    train_index, source_test_index = source_index[0:ind], source_index[ind:]
    # train_index, source_test_index = train_test_split(source_index,  test_size=test_size, random_state=int(random_seed), 
    #                                             stratify=label_df.loc[idx[:, source_index], target].values.tolist())
    
    for modeltype in model_types:

        is_time_series = modeltype in ['lstm', 'gru', 'grud']
        # is_time_series=True

        print("data preprocessing")

        X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(
            filtered_df_time_window.loc[idx[:,train_index,:],:], level, 'Simple', 
            target, representation, is_time_series, impute=not(modeltype=='grud'))

        print(X_df.shape)

        print("Finding the best %s model using a random search" % modeltype.upper())

        model, y_pred_prob, selected_features_mask = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype,
                                                                    feature_selection=feature_selection, **feature_selection_args)

        if feature_selection:
            X_df=X_df.loc[:, selected_features_mask]
        if save_data:
            key_suffix="-".join(train_hospitals)+"_"+modeltype.upper()
            X_df.to_hdf(data_out, key="X_train_" + key_suffix, mode='a')
            pd.Series(y).to_hdf(data_out, key='y_train_' + key_suffix, mode='a')

        # Record what the best performing model was
        model_filename=os.path.join(output_dir, 
                                    prefix + "bestmodel-hospital-style_trainHospitals_{}_{}_{}_Simple_{}_seed-{}_target={}.pkl".format(
                                        modeltype.upper(), "-".join(train_hospitals), representation, level, str(random_seed), str(target))
                                    )
        print(model_filename)
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

        if (test_hospitals is None or len(test_hospitals) == 0):
            test_hospitals=set(sites_df['hospital'].values.tolist())

        dump_filename=os.path.join(output_dir, 
                                    prefix + "result_hospital-style_{}_{}_{}_Simple_{}_seed-{}_target={}.txt".format(
                                        "-".join(train_hospitals), modeltype.upper(), representation, level, str(random_seed), str(target))
                                    )
        with open(dump_filename, 'w') as f:
            for hospital in tqdm(sorted(test_hospitals)):

                if hospital in train_hospitals:
                    test_index = source_test_index
                else:           
                    test_index=sites_df[sites_df['hospital'] == hospital].index.tolist()
                # get the X and y data for testing
                X_df, y, _, _, gender, ethnicity, subject_id= data_preprocessing(
                    filtered_df_time_window.loc[idx[:,test_index,:],:], level, 'Simple', target,
                    representation, is_time_series, impute=not(modeltype=='grud'),
                    timeseries_vect=timeseries_vect, representation_vect=representation_vect)
                
                if feature_selection:
                    X_df=X_df.loc[:, selected_features_mask]
                        
                if save_data:
                    key_suffix = hospital + "_" + modeltype.upper()
                    X_df.to_hdf(data_out, key="X_test_" + key_suffix, mode='a')
                    pd.Series(y).to_hdf(data_out, key='y_test_' + key_suffix, mode='a')

                y, y_pred_prob, pred = get_prediction(modeltype, model, X_df, y, subject_id)

                AUC, F1, ACC, APR, ECE, MCE, O_E = get_measures(y, y_pred_prob, pred, modeltype)

                f.write("hospital, {}, AUC, {} \r\n".format(str(hospital), str(AUC)))
                f.write("hospital, {}, F1, {} \r\n".format(str(hospital), str(F1)))
                f.write("hospital, {}, Acc, {} \r\n".format(str(hospital), str(ACC)))
                f.write("hospital, {}, APR, {} \r\n".format(str(hospital), str(APR)))
                f.write("hospital, {}, ECE, {} \r\n".format(str(hospital), str(ECE)))
                f.write("hospital, {}, MCE, {} \r\n".format(str(hospital), str(MCE)))
                f.write("hospital, {}, O_E, {} \r\n".format(str(hospital), str(O_E)))

                f.write("hospital, {}, label, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in y])))
                f.write("hospital, {}, pred, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in pred])))
                f.write("hospital, {}, y_pred_prob, <{}>\r\n".format(str(hospital), ",".join([str(i) for i in y_pred_prob])))
                try:
                    f.write("hospital, {}, gender, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in gender])))
                    f.write("hospital, {}, ethnicity, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in ethnicity])))
                    f.write("hospital, {}, subject, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in subject_id])))
                except:
                    pass

                f.write("hospital, {}, best_params, <{}> \r\n".format(str(hospital), best_params))

        print("Finished {}".format(dump_filename))

    return

def main_hospital_overtime(random_seed=None, max_time=24, level='itemid', representation='raw', test_month_interval=2, training_years=[2010],
         target='mort_icu', prefix="", model_types=['rf'],  data_dir="", train_hospitals=[], output_dir="", test_hospitals=[], save_data=False,
         feature_selection=False, **feature_selection_args):
    """
    This function trains data from training_years hidenic and tests on data after training_years, once every test_month_interval month.

    """
    global filtered_df
    global label_df
    global sites_df
    global scaler
    global train_means


    def _train_test_util(models, is_time_series, impute):
        '''
        purpose: to share processed data between models and save computation time
        should be called separately for standard models and time_series models
        '''
        if save_data:
            data_type="time_series" if is_time_series else "flat"
            data_out=os.path.join(data_dir, "preprocessed_data/", prefix + "hospital-overtime-style_{}_{}_data_{}_{}_Simple_seed_{}_target={}.h5".format(
                    "-".join(train_hospitals),  "-".join([str(i) for i in training_years]), data_type, representation, str(random_seed), str(target)))

        print("data preprocessing")

        train_X_df, train_y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(
            filtered_df_time_window.loc[idx[:,train_index,:],:], level, 'Simple', 
            target, representation, is_time_series, impute=impute)

        print(train_X_df.shape)


        dump_filename=os.path.join(output_dir, 
                                    prefix + "result_hospital-overtime-style_{}_{}_{}_{}_Simple_{}_seed-{}_target={}.txt".format(
                                        "-".join(train_hospitals), "-".join([str(i) for i in training_years]), "-".join([m.upper() for m in models]), representation, level, str(random_seed), str(target))
                                    )
        with open(dump_filename, 'w') as f:        
            for modeltype in models:
                print("Finding the best %s model using a random search" % modeltype.upper())
                model, y_pred_prob, selected_features_mask = classifier_select(train_X_df, np.asarray(train_y).ravel(), is_time_series, subject_id, modeltype=modeltype,
                                                                    feature_selection=feature_selection, **feature_selection_args)

                ## write the training result of the best estimator
                best_rank_index=train_cv_results['rank_test_AUC'] == 1
                for score in ['AUC', 'APR', 'ECE', 'MCE']:
                    f.write("modeltype, {}, train_cv_score, {}, {} \r\n".format(modeltype.upper(), score, str(train_cv_results['mean_test_%s' % score][best_rank_index])))
                f.write("modeltype, {}, train_label, <{}> \r\n".format(modeltype.upper(), ",".join([str(i) for i in train_y])))
                f.write("modeltype, {}, train_y_pred_prob, <{}> \r\n".format(modeltype.upper(), ",".join([str(i) for i in y_pred_prob])))
                    
                # Record what the best performing model was
                model_filename=os.path.join(output_dir, 
                                            prefix + "bestmodel_hospital-overtime_{}_{}_{}_{}_Simple_{}_seed-{}_target={}.pkl".format(
                                                "-".join(train_hospitals), "-".join([str(i) for i in training_years]), modeltype.upper(), representation, level, str(random_seed), str(target))
                                            )
                print(model_filename)
                with open(model_filename, 'wb') as model_file:
                    pickle.dump(model, model_file)
                
                if feature_selection:
                    X_df=train_X_df.loc[:, selected_features_mask].copy()
                else:
                    X_df=train_X_df.copy()

                if save_data:
                    key_suffix="-".join(train_hospitals)+"_"+modeltype.upper()
                    X_df.to_hdf(data_out, key="X_train_" + key_suffix, mode='a')
                    pd.Series(train_y).to_hdf(data_out, key='y_train_' + key_suffix, mode='a')

                for hospital in tqdm(sorted(test_hospitals)):
                    print('test hospital: {}'.format(hospital))
                    site_index=sites_df[sites_df['hospital'] == hospital].index.tolist()
                    print('site_index size: {}'.format(str(len(site_index))))

                    for year in tqdm(sorted(test_years)):
                        print('year:', str(year))
                        for month in range(1, 13, test_month_interval):
                            test_months=np.arange(month, month+test_month_interval, 1)
                            print('months: ', [str(m) for m in test_months])
                            # test on years 2011 onwards, every 2 months
                            date_index=years_df[(years_df['year'].isin([year])) &
                                                (years_df['month'].isin(test_months))].index.tolist()
        
                            print('date_index size: {}'.format(str(len(date_index))))
                            test_index=set(site_index).intersection(date_index)
                            print('test_index size: {}'.format(str(len(test_index))))

                            if (len(test_index)<50):
                                print("test size is too small: skipping this iteration ...")
                                continue

                            # get the X and y data for testing
                            X_df, y, _, _, gender, ethnicity, subject_id= data_preprocessing(
                                filtered_df_time_window.loc[idx[:,test_index,:],:], level, 'Simple', target,
                                representation, is_time_series, impute=impute,
                                timeseries_vect=timeseries_vect, representation_vect=representation_vect)
                            
                            if feature_selection:
                                X_df=X_df.loc[:, selected_features_mask]

                            if save_data:
                                key_suffix= hospital + "_" + str(year) + "_" + "-".join([str(i) for i in test_months])+"_" + modeltype.upper()
                                X_df.to_hdf(data_out, key="X_test_" + key_suffix, mode='a')
                                pd.Series(y).to_hdf(data_out, key='y_test_' + key_suffix, mode='a')

                            
                            y, y_pred_prob, pred = get_prediction(modeltype, model, X_df, y, subject_id)

                            AUC, F1, ACC, APR, ECE, MCE, O_E = get_measures(y, y_pred_prob, pred, modeltype)

                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, AUC, {} \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), str(AUC)))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, F1, {} \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), str(F1)))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, Acc, {} \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), str(ACC)))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, APR, {} \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), str(APR)))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, ECE, {} \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), str(ECE)))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, MCE, {} \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), str(MCE)))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, O_E, {} \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), str(O_E)))

                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, label, <{}> \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in y])))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, pred, <{}> \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in pred])))
                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, y_pred_prob, <{}>\r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in y_pred_prob])))
                            try:
                                f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, gender, <{}> \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in gender])))
                                f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, ethnicity, <{}> \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in ethnicity])))
                                f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, subject, <{}> \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in subject_id])))
                            except:
                                pass

                            f.write("modeltype, {}, hospital, {}, year, {}, months, <{}>, best_params, <{}> \r\n".format(modeltype.upper(), str(hospital), str(year), ",".join([str(i) for i in test_months]), best_params))

                print("Finished {}".format(dump_filename))

    
    np.random.seed(random_seed)




    # (Amin) not using torch yet
    # torch.manual_seed(random_seed)





    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                              (filtered_df.index.get_level_values('hours_in') <= max_time)]

    #drop hours in row
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    year_index=years_df[years_df['year'].isin(training_years)].index.tolist()
    site_index=sites_df[sites_df['hospital'].isin(train_hospitals)].index.tolist()
    train_index=set(year_index).intersection(site_index)

    years_set=set(years_df['year'].values.tolist())
    ## exclude any year that is smaller than training_years
    test_years=set([yr for yr in years_set if (yr > np.array(training_years)).all()])

    if (test_hospitals is None or len(test_hospitals) == 0):
        test_hospitals=set(sites_df['hospital'].values.tolist())

    ## train_test standard models
    std_models = [m for m in model_types if m not in ['lstm', 'gru', 'grud']]
    if len(std_models)!=0:
        is_time_series = False
        impute=True
        _train_test_util(std_models, is_time_series, impute)

    ## train_test timeseries models
    timeseries_models = [m for m in model_types if m in ['lstm', 'gru']]
    if len(timeseries_models)!=0:
        is_time_series = True
        impute=True
        _train_test_util(timeseries_models, is_time_series, impute)
    
    ## train_test GRUD model
    if 'grud' in model_types:
        is_time_series = True
        impute=False
        _train_test_util(['grud'], is_time_series, impute)
 
    return


def main_icu_type(random_seed=None, max_time=24, level='itemid', representation='raw', test_size=0.2,
         target='mort_icu', prefix="", model_types=['rf'],  data_dir="", train_icu_types=[], output_dir="", test_icu_types=[]):
    """
    This function trains data from training_years hidenic and tests on data after training_years, once every test_month_interval month.

    """
    global filtered_df
    global label_df
    global sites_df
    global scaler
    global train_means

    np.random.seed(random_seed)




    # (Amin) not using torch yet
    # torch.manual_seed(random_seed)





    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                              (filtered_df.index.get_level_values('hours_in') <= max_time)]

    #drop hours in row
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    source_index=sites_df[sites_df['icu_category'].isin(train_icu_types)].index.tolist()

    train_index, source_test_index = train_test_split(source_index,  test_size=test_size, random_state=int(random_seed), 
                                                stratify=label_df.loc[idx[:, source_index], target].values.tolist())
    

    for modeltype in model_types:

        is_time_series = modeltype in ['lstm', 'gru', 'grud']
        # is_time_series=True

        print("data preprocessing")

        X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(
            filtered_df_time_window.loc[idx[:,train_index,:],:], level, 'Simple', 
            target, representation, is_time_series, impute=not(modeltype=='grud'))

        print(X_df.shape)

        print("Finding the best %s model using a random search" % modeltype.upper())

        model = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype)

        # Record what the best performing model was
        model_filename=os.path.join(output_dir, 
                                    prefix + "bestmodel-icuType-style_trainICUtypes_{}_{}_{}_Simple_{}_seed-{}_target={}.pkl".format(
                                        modeltype.upper(), "-".join(train_icu_types), representation, level, str(random_seed), str(target))
                                    )
        print(model_filename)
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

        if (test_icu_types is None or len(test_icu_types) == 0):
            icu_types=set(sites_df['icu_category'].values.tolist())
            test_icu_types=icu_types

        dump_filename=os.path.join(output_dir, 
                                    prefix + "result_icuType-style_{}_{}_{}_Simple_{}_seed-{}_target={}.txt".format(
                                        "-".join(train_icu_types), modeltype.upper(), representation, level, str(random_seed), str(target))
                                    )
        with open(dump_filename, 'w') as f:
            for icu_type in tqdm(sorted(test_icu_types)):
                
                print("ICU_TYPE: {}".format(icu_type))

                if icu_type in train_icu_types:
                    test_index = source_test_index
                else:           
                    test_index=sites_df[sites_df['icu_category'] == icu_type].index.tolist()
                # get the X and y data for testing
                X_df, y, _, _, gender, ethnicity, subject_id= data_preprocessing(
                    filtered_df_time_window.loc[idx[:,test_index,:],:], level, 'Simple', target,
                    representation, is_time_series, impute=not(modeltype=='grud'),
                    timeseries_vect=timeseries_vect, representation_vect=representation_vect)

                y, y_pred_prob, pred = get_prediction(modeltype, model, X_df, y, subject_id)

                AUC, F1, ACC, APR, ECE, MCE, O_E = get_measures(y, y_pred_prob, pred, modeltype)

                f.write("icu_type, {}, AUC, {} \r\n".format(str(icu_type), str(AUC)))
                f.write("icu_type, {}, F1, {} \r\n".format(str(icu_type), str(F1)))
                f.write("icu_type, {}, Acc, {} \r\n".format(str(icu_type), str(ACC)))
                f.write("icu_type, {}, APR, {} \r\n".format(str(icu_type), str(APR)))
                f.write("icu_type, {}, ECE, {} \r\n".format(str(icu_type), str(ECE)))
                f.write("icu_type, {}, MCE, {} \r\n".format(str(icu_type), str(MCE)))
                f.write("icu_type, {}, O_E, {} \r\n".format(str(icu_type), str(O_E)))

                f.write("icu_type, {}, label, <{}> \r\n".format(str(icu_type), ",".join([str(i) for i in y])))
                f.write("icu_type, {}, pred, <{}> \r\n".format(str(icu_type), ",".join([str(i) for i in pred])))
                f.write("icu_type, {}, y_pred_prob, <{}>\r\n".format(str(icu_type), ",".join([str(i) for i in y_pred_prob])))
                try:
                    f.write("icu_type, {}, gender, <{}> \r\n".format(str(icu_type), ",".join([str(i) for i in gender])))
                    f.write("icu_type, {}, ethnicity, <{}> \r\n".format(str(icu_type), ",".join([str(i) for i in ethnicity])))
                    f.write("icu_type, {}, subject, <{}> \r\n".format(str(icu_type), ",".join([str(i) for i in subject_id])))
                except:
                    pass

                f.write("icu_type, {}, best_params, <{}> \r\n".format(str(icu_type), best_params))

        print("Finished {}".format(dump_filename))

    return

def main_single_site(site_name="UPMCPUH", random_seed=None, max_time=24, level='itemid', representation='raw',
         target='mort_icu', prefix="", model_types=['rf'],  data_dir="", training_years=[2010], output_dir="", test_month_interval=2,
         save_data=False, feature_selection=False, **feature_selection_args):
    """
    This function trains data on a single site from training_years hidenic and tests on data after training_years, once every test_month_interval month.

    """
    global filtered_df
    global label_df
    global years_df
    global scaler
    global train_means
    global sites_df
    global train_cv_results



    np.random.seed(random_seed)


    # (Amin) not using torch yet
    # torch.manual_seed(random_seed)


    if save_data:
        data_out=os.path.join(data_dir, "preprocessed_data/", prefix + "single-site-style_{}_{}_{}_{}_Simple_seed_{}_target={}.h5".format(
                            site_name.upper(), "-".join([str(i) for i in training_years]), "feature-selection" if feature_selection else "", representation, str(random_seed), str(target)))
    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                              (filtered_df.index.get_level_values('hours_in') <= max_time)]

    #drop hours in row
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    if site_name in sites_df['hospital'].tolist():
        site_index=sites_df[sites_df['hospital']==site_name].index.tolist()
    elif site_name in sites_df['icu_category'].tolist():
        site_index=sites_df[sites_df['icu_category']==site_name].index.tolist()

    years_df_filtered=years_df.loc[years_df.index.intersection(site_index),:]

    train_index=years_df_filtered[years_df_filtered['year'].isin(training_years)].index.tolist()

    print("train_size", len(train_index))
    
    for modeltype in model_types:

        is_time_series = modeltype in ['lstm', 'gru', 'grud']
        # is_time_series=True

        print("data preprocessing")
        X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(
            filtered_df_time_window.loc[idx[:,train_index,:],:], level, 'Simple', 
            target, representation, is_time_series, impute=not(modeltype=='grud'))
        print(X_df.shape)

        print("Finding the best %s model using a random search" % modeltype.upper())

        model, y_pred_prob, selected_features_mask = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype,
                                feature_selection=feature_selection, **feature_selection_args)


        # Record what the best performing model was
        model_filename=os.path.join(output_dir, 
                                    prefix+"bestmodel_single-site-style_site_{}_train-years_{}_{}_{}_Simple_{}_seed-{}_target={}.pkl".format(
                                        site_name.upper(), "-".join([str(i) for i in training_years]), modeltype.upper(), representation, level, str(random_seed), str(target)))
        print(model_filename)
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

        if feature_selection:
            X_df=X_df.loc[:, selected_features_mask]

        if save_data:
            key_suffix="-".join([str(i) for i in training_years])
            if feature_selection:
                key_suffix=key_suffix + "_" + modeltype.upper()
          
            X_df.to_hdf(data_out, key="X_train_" + key_suffix, mode='a')
            pd.Series(y).to_hdf(data_out, key='y_train_' + key_suffix, mode='a')

        years_set=set(years_df_filtered['year'].tolist())
        ## exclude any year that is smaller than training_years
        test_years=set([yr for yr in years_set if (yr > np.array(training_years)).all()])

        dump_filename=os.path.join(output_dir, 
                                    prefix+"result_single-site-style_site_{}_train-years_{}_{}_{}_Simple_{}_seed-{}_target={}.txt".format(
                                        site_name.upper(), "-".join([str(i) for i in training_years]), modeltype.upper(), representation, level, str(random_seed), str(target)))
        with open(dump_filename, 'w') as f:

            best_rank_index=train_cv_results['rank_test_AUC'] == 1
            for score in ['AUC', 'APR', 'ECE', 'MCE']:
                f.write("train_cv_score, {}, {} \r\n".format(score, str(train_cv_results['mean_test_%s' % score][best_rank_index])))
            f.write("train_label, <{}> \r\n".format(",".join([str(i) for i in y])))
            f.write("train_y_pred_prob, <{}> \r\n".format(",".join([str(i) for i in y_pred_prob])))

            for year in tqdm(sorted(test_years)):
                for month in range(1, 13, test_month_interval):
                    test_months=np.arange(month, month+test_month_interval, 1)
                    # open the trained model and test on years 2011 onwards, every 2 months
                    test_index=years_df_filtered[(years_df_filtered['year'].isin([year])) &
                                        (years_df_filtered['month'].isin(test_months))].index.tolist()
                    # if year in training_years:
                    #     #eliminate the training data from testing
                    #     year_index=list(set(year_index).intersection(set(test_years)))

                    # get the X and y data for testing
                    X_df, y, _, _, gender, ethnicity, subject_id= data_preprocessing(
                        filtered_df_time_window.loc[idx[:,test_index,:],:], level, 'Simple', target,
                        representation, is_time_series, impute=not(modeltype=='grud'),
                        timeseries_vect=timeseries_vect, representation_vect=representation_vect)

                    print("test_df shape: ", X_df.shape)

                    if feature_selection:
                        X_df=X_df.loc[:, selected_features_mask]

                    if save_data:
                        key_suffix=str(year) + "_" + "-".join([str(i) for i in test_months])
                        if feature_selection:
                            key_suffix=key_suffix + "_" + modeltype.upper()
                    
                        X_df.to_hdf(data_out, key="X_test_" + key_suffix, mode='a')
                        pd.Series(y).to_hdf(data_out, key='y_test_' + key_suffix, mode='a')

                    y, y_pred_prob, pred = get_prediction(modeltype, model, X_df, y, subject_id)

                    AUC, F1, ACC, APR, ECE, MCE, O_E = get_measures(y, y_pred_prob, pred, modeltype)

                    f.write("year, {}, months, <{}>, AUC, {} \r\n".format(str(year), ",".join([str(i) for i in test_months]), str(AUC)))
                    f.write("year, {}, months, <{}>, F1, {} \r\n".format(str(year), ",".join([str(i) for i in test_months]), str(F1)))
                    f.write("year, {}, months, <{}>, Acc, {} \r\n".format(str(year), ",".join([str(i) for i in test_months]), str(ACC)))
                    f.write("year, {}, months, <{}>, APR, {} \r\n".format(str(year), ",".join([str(i) for i in test_months]), str(APR)))
                    f.write("year, {}, months, <{}>, ECE, {} \r\n".format(str(year), ",".join([str(i) for i in test_months]), str(ECE)))
                    f.write("year, {}, months, <{}>, MCE, {} \r\n".format(str(year), ",".join([str(i) for i in test_months]), str(MCE)))
                    f.write("year, {}, months, <{}>, O_E, {} \r\n".format(str(year), ",".join([str(i) for i in test_months]), str(O_E)))

                    f.write("year, {}, months, <{}>, label, <{}> \r\n".format(str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in y])))
                    f.write("year, {}, months, <{}>, pred, <{}> \r\n".format(str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in pred])))
                    f.write("year, {}, months, <{}>, y_pred_prob, <{}>\r\n".format(str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in y_pred_prob])))
                    try:
                        f.write("year, {}, months, <{}>, gender, <{}> \r\n".format(str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in gender])))
                        f.write("year, {}, months, <{}>, ethnicity, <{}> \r\n".format(str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in ethnicity])))
                        f.write("year, {}, months, <{}>, subject, <{}> \r\n".format(str(year), ",".join([str(i) for i in test_months]), ",".join([str(i) for i in subject_id])))
                    except:
                        pass

                    f.write("year, {}, months, <{}>, best_params, <{}> \r\n".format(str(year), ",".join([str(i) for i in test_months]), best_params))


        print("Finished {}".format(dump_filename))

    return

def main_hospital_pairwise(train_hospitals, test_hospitals=[], target='mort_icu', model_types=['rf'], representation='raw', 
        test_size=0.2, max_time=24, level='itemid', prefix="", data_dir="", output_dir="", save_data=False, random_seed=None, 
        feature_selection=False, **feature_selection_args):

    ## call the main_hospital_wise() function for each train hospital to run a pairwise analysis
    for hospital in train_hospitals:
        print("train hospital =", hospital)
        hospitals = [hospital]
        main_hospital_wise(random_seed, max_time, level, representation, test_size,
            target, prefix, model_types, data_dir, hospitals, output_dir, test_hospitals, save_data,
            feature_selection, **feature_selection_args)
        print("#"*100)

def main_hospital_wise_bootstrap(n_bootstrap=10, random_seed=None, max_time=24, level='itemid', representation='raw', test_size=0,
         target='mort_icu', prefix="", modeltype='rf',  data_dir="", train_hospitals=[], output_dir="",
         test_hospitals=[], save_data=False, feature_selection=False, **feature_selection_args):
    """
    This function trains data from training_years hidenic and tests on data after training_years, once every test_month_interval month.

    """
    global filtered_df
    global label_df
    global sites_df
    global scaler
    global train_means
    global years_df
    global logdir

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # print("Loading the data.")
    # load_data(max_time=max_time, data_dir=data_dir)

    is_load_from_disk=False
    if save_data:
        data_out=os.path.join(data_dir, "preprocessed_data/", prefix + "hospital-bootstrap-style_trainHospitals_{}_{}_{}_Simple_seed_{}_target={}.h5".format(
                "-".join(train_hospitals), modeltype.upper(), representation, str(random_seed), str(target)))
        p = pathlib.Path(data_out)
        if p.is_file():
            is_load_from_disk=True

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in') >= 0) &
                                              (filtered_df.index.get_level_values('hours_in') <= max_time)]

    #drop hours in row
    # filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    source_index = sites_df[sites_df['hospital'].isin(train_hospitals)].index.tolist()
    ## sort index by year and split the last len(index)*test_size for test set

    for n_bs in range(n_bootstrap):
        print('source_index size=', len(source_index))
        indices = resample(source_index, replace=True, n_samples=len(source_index), random_state=n_bs)
        indices = years_df.loc[indices, ['year', 'month']].sort_values(['year', 'month']).index
        ind = int(len(indices)*(1-test_size))
        train_index, source_test_index = indices[0:ind], indices[ind:]
        print('train_index size=', len(train_index))

        # train_index, source_test_index = train_test_split(source_index,  test_size=test_size, random_state=int(random_seed), 
        #                                             stratify=label_df.loc[idx[:, source_index], target].values.tolist())
        
        is_time_series = modeltype in ['lstm', 'gru', 'grud']
        # is_time_series=True

        print("data preprocessing")

        if is_load_from_disk:
            key_suffix="-".join(train_hospitals)+"_"+modeltype.upper()+"_nbs_"+str(n_bs)
            X_df = pd.read_hdf(data_out, key="X_train_" + key_suffix)
            y = pd.read_hdf(data_out, key='y_train_' + key_suffix).values
            subject_id = None
        else:
            X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(
                filtered_df_time_window.loc[idx[:,train_index,:],:], level, 'Simple', 
                target, representation, is_time_series, impute=not(modeltype=='grud'))

        print('processed df shape=', X_df.shape)

        # print("Finding the best %s model using a random search" % modeltype.upper())

        logdir = pathlib.Path(output_dir)/ ('runs/' + target + "_" + "_".join(train_hospitals) + "_nbs_" + str(n_bs))
        model, y_pred_prob, selected_features_mask = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype,
                                                                    feature_selection=feature_selection, **feature_selection_args)

        if not is_load_from_disk:
            if feature_selection:
                X_df=X_df.loc[:, selected_features_mask]
            if save_data:
                key_suffix="-".join(train_hospitals)+"_"+modeltype.upper()+"_nbs_"+str(n_bs)
                X_df.to_hdf(data_out, key="X_train_" + key_suffix, mode='a')
                pd.Series(y).to_hdf(data_out, key='y_train_' + key_suffix, mode='a')

        # Record what the best performing model was
        model_filename=os.path.join(output_dir, 
                                    prefix + "bestmodel-hospital-bootstrap{}-style_trainHospitals_{}_{}_{}_Simple_{}_seed-{}_target={}.pkl".format(
                                        str(n_bs), modeltype.upper(), "-".join(train_hospitals), representation, level, str(random_seed), str(target))
                                    )
        print(model_filename)
        if modeltype == 'mlp_torch':
            torch.save(model, model_filename)
        else:
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)

        if (test_hospitals is None or len(test_hospitals) == 0):
            test_hospitals=set(sites_df['hospital'].values.tolist())
            test_hospitals=[h for h in test_hospitals if h not in train_hospitals]

        dump_filename=os.path.join(output_dir, 
                                    prefix + "result_hospital-bootstrap{}-style_{}_{}_{}_Simple_{}_seed-{}_target={}.txt".format(
                                        str(n_bs), "-".join(train_hospitals), modeltype.upper(), representation, level, str(random_seed), str(target))
                                    )
        with open(dump_filename, 'w') as f:
            for hospital in tqdm(sorted(test_hospitals)):

                if hospital in train_hospitals:
                    # test_index = source_test_index
                    continue
                else:           
                    test_index=sites_df[sites_df['hospital'] == hospital].index.tolist()
                # get the X and y data for testing

                if is_load_from_disk:
                    key_suffix = hospital + "_" + modeltype.upper()+"_nbs_"+str(n_bs)
                    X_df = pd.read_hdf(data_out, key="X_test_" + key_suffix)
                    y = pd.read_hdf(data_out, key='y_test_' + key_suffix).values
                    subject_id = None
                else:
                    X_df, y, _, _, gender, ethnicity, subject_id= data_preprocessing(
                        filtered_df_time_window.loc[idx[:,test_index,:],:], level, 'Simple', target,
                        representation, is_time_series, impute=not(modeltype=='grud'),
                        timeseries_vect=timeseries_vect, representation_vect=representation_vect)
                    if feature_selection:
                        X_df=X_df.loc[:, selected_features_mask]
                    if save_data:
                        key_suffix = hospital + "_" + modeltype.upper()+"_nbs_"+str(n_bs)
                        X_df.to_hdf(data_out, key="X_test_" + key_suffix, mode='a')
                        pd.Series(y).to_hdf(data_out, key='y_test_' + key_suffix, mode='a')

                y, y_pred_prob, pred = get_prediction(modeltype, model, X_df, y, subject_id)
                AUC, F1, ACC, APR, ECE, MCE, O_E = get_measures(y, y_pred_prob, pred, modeltype)

                f.write("hospital, {}, AUC, {} \r\n".format(str(hospital), str(AUC)))
                f.write("hospital, {}, F1, {} \r\n".format(str(hospital), str(F1)))
                f.write("hospital, {}, Acc, {} \r\n".format(str(hospital), str(ACC)))
                f.write("hospital, {}, APR, {} \r\n".format(str(hospital), str(APR)))
                f.write("hospital, {}, ECE, {} \r\n".format(str(hospital), str(ECE)))
                f.write("hospital, {}, MCE, {} \r\n".format(str(hospital), str(MCE)))
                f.write("hospital, {}, O_E, {} \r\n".format(str(hospital), str(O_E)))

                f.write("hospital, {}, label, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in y])))
                f.write("hospital, {}, pred, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in pred])))
                f.write("hospital, {}, y_pred_prob, <{}>\r\n".format(str(hospital), ",".join([str(i) for i in y_pred_prob])))
                try:
                    f.write("hospital, {}, gender, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in gender])))
                    f.write("hospital, {}, ethnicity, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in ethnicity])))
                    f.write("hospital, {}, subject, <{}> \r\n".format(str(hospital), ",".join([str(i) for i in subject_id])))
                except:
                    pass

                f.write("hospital, {}, best_params, <{}> \r\n".format(str(hospital), best_params))

        print("Finished {}".format(dump_filename))

    return

def main_hospital_pairwise_bootstrap(n_bootstrap, train_hospitals, test_hospitals=[], target='mort_icu', modeltype='rf', representation='raw', 
        test_size=0, max_time=24, level='itemid', prefix="", data_dir="", output_dir="", save_data=False, random_seed=None, 
        feature_selection=False, **feature_selection_args):

    ## call the main_hospital_wise() function for each train hospital to run a pairwise analysis
    for hospital in train_hospitals:
        print("train hospital =", hospital)
        hospitals = [hospital]
        main_hospital_wise_bootstrap(n_bootstrap, random_seed, max_time, level, representation, test_size,
            target, prefix, modeltype, data_dir, hospitals, output_dir, test_hospitals, save_data,
            feature_selection, **feature_selection_args)
        print("#"*100)


if __name__=="__main__":
    print("in main")
    import os
    ### location of pytables_hdf5.dll ###
    os.environ['PATH'] += os.pathsep + os.path.expanduser(r'C:\Users\mot16\anaconda3\envs\lemr\Lib\site-packages\tables')
    import argparse

    parser = argparse.ArgumentParser(description='Train models on HIDENIC EMR data')
    parser.add_argument('--experiment_mode', type=str, choices=['shift_simulation', "explanation", 'prediction', 'compute_distance', 'training'])
    parser.add_argument('--target', type=str, choices=['mort_icu', 'los_3'])
    parser.add_argument('--data_dir', type=str, default="", help="full path to the folder containing the data")
    parser.add_argument('--output_dir', type=str, default="", help="full path to the folder of results")
    parser.add_argument('--shift_metrics', type=str, nargs='+', help="['summary_stats','univariate_ttest', 'euclidean_distance', 'WD']")
    parser.add_argument('--shift_features', type=str, nargs='+', help="['RR', 'HR]")
    parser.add_argument('--shift_values', type=int, nargs='+', help="[2, 5]")
    parser.add_argument('--shift_n_iter', type=int)
    parser.add_argument('--num_samples', type=int, default=None)

    args = parser.parse_args()
    
    experiment_mode=args.experiment_mode
    target=args.target
    data_dir=args.data_dir
    output_dir=args.output_dir
    

    if isinstance(args.shift_metrics, str):
        args.shift_metrics=[args.shift_metrics]

    if isinstance(args.shift_features, str):
        args.shift_features=[args.shift_features]

    if isinstance(args.shift_values, int):
        args.shift_values=[args.shift_values]

    
    modeltypes=['rf', 'xg', 'mlp', 'lr']
    data_preprocessing_mode='basic_no_indicator' #'basic'
    randomSearchCVargs = {'n_iter':200, 'cv':5}
    train_years = [2010]
    test_years = [2011,2012,2013,2014]
    random_state=0
    max_time=24

    # data_dir="E:/Data/HIDENIC_EXTRACT_OUTPUT_DIR/POP_SIZE_0/ITEMID_REP/"
    # output_dir="C:/Users/mot16/OneDrive - University of Pittsburgh/_BoxMigration/Projects/Thesis/output/HIDENIC_overtime_analysis/overall_overtime_v2"

    print(f'experiment_mode: {experiment_mode}\n')

    t0 = time.time()
    print("Loading the data.")
    t00 = time.time()

    load_data(data_dir=data_dir, num_samples=args.num_samples)
    
    t01 = time.time()
    print(f"data loaded in {(t01-t00):10.2f} seconds.")
    
    if experiment_mode=='training':
        print('training mode \n' + '*'*50)
        print(f'target={target}')
        for i, modeltype in enumerate(modeltypes):
            if i>0:
                load_data(data_dir=data_dir)
            print(f'\n{"$"*20} {modeltype} {"$"*20}\n')
            main_overtime_overall(target=target, max_time=max_time, random_state=random_state, modeltype=modeltype, 
                            train_years=train_years, test_years=test_years, data_dir=data_dir, output_dir=output_dir,
                            data_preprocessing_mode=data_preprocessing_mode, randomSearchCVargs=randomSearchCVargs)
    elif experiment_mode=='prediction':
        print('prediction mode \n' + '*'*50)
        metrics = ['AUC', 'F1', 'APR', 'ECE', 'BRIER', 'CE', 'ECE2']
        main_generate_predictions(modeltypes, metrics, train_years, test_years, target, max_time, data_preprocessing_mode, output_dir)
    elif experiment_mode=='compute_distance':
        dist_metrics= ["MMD_alibi"] #["MMD_rbf_tl"] #["WD", "MMD"]
        measure_distribution_distance(dist_metrics, train_years, test_years, target, max_time, data_preprocessing_mode, output_dir)
    elif experiment_mode=='shift_simulation':
        # shift_params={
        #     'metrics':['summary_stats','univariate_ttest', 'euclidean_distance', 'WD'],
        #     'features':['RR'],
        #     'shift_values':[2],
        #     'num_iterations':20
        # }
        shift_params={
            'metrics':args.shift_metrics, 
            'features':args.shift_features,
            'shift_values':args.shift_values,
            'num_iterations':args.shift_n_iter
        }
        opt_metrics=['BRIER']
        in_dir = output_dir
        output_dir = os.path.join(output_dir, 'shift_simulation')
        compute_model_performance=True
        main_gradual_shift(shift_params, modeltypes, opt_metrics, train_years, target,
                            max_time, data_preprocessing_mode, in_dir, output_dir,
                            compute_model_performance)
    elif experiment_mode=="explanation":
        explanation_size=5000
        modeltype='xg'
        shift_params={
            'features':['HR'],
            'shift_values':[10],
            'num_iterations':10
        }
        opt_metric='BRIER'
        in_dir = output_dir
        output_dir += '/shift_simulation'
        years=train_years
        main_explain_gradual_shift(explanation_size, modeltype, shift_params, opt_metric, 
                            target, years, max_time, data_preprocessing_mode, in_dir, output_dir, random_state)

    t1 = time.time()
    print(f'Total runtime={(t1-t0):10.2f} seconds.')
    
    exit()

    # global n_threads
    # global output_dir

    # print("in main")
    # import argparse
    # parser = argparse.ArgumentParser(description='Train models on HIDENIC EMR data with year-of-care')
    # parser.add_argument('--test_size', type=float, default=0)
    # parser.add_argument('--max_time', type=int, default=24)
    # parser.add_argument('--random_seed', type=int, nargs='+', default=None)
    # parser.add_argument('--level', type=str, default='itemid', choices=['itemid', 'Level2', 'nlp'])
    # parser.add_argument('--representation', type=str, nargs='+', default=['raw'], help="['raw', 'pca', 'umap', 'autoencoder', 'nlp']")
    # parser.add_argument('--target_list', type=str, nargs='+', default=None, help="choices:['mort_icu', 'los_3']")
    # parser.add_argument('--prefix', type=str, default="")
    # parser.add_argument('--model_types', type=str, nargs='+', default=None, help="choices: ['rf', 'lr', 'svm', 'rbf-svm', 'knn', 'mlp', '1class_svm', '1class_svm_novel', 'iforest', 'lstm', 'gru', 'grud']")
    # parser.add_argument('--train_types', type=str,  nargs='+', default=None, help="choices:['overall_overtime', 'rolling_limited', 'rolling', 'no_years',\
    #                                                                                 'hospital_wise', 'icu_type', 'single_site', hospital_overtime, hospital_pairwise\
    #                                                                                  hospital_pairwise_bootstrap]")
    # parser.add_argument('--data_dir', type=str, default="", help="full path to the folder containing the data")
    # parser.add_argument('--output_dir', type=str, default="", help="full path to the folder of results")
    # parser.add_argument('--gpu', type=str, default=0, nargs='+', help="which GPUS to train on")
    # parser.add_argument('--n_threads', type=int, default=-1, help="Number of threads to use for CPU model searches.")
    # parser.add_argument('--test_month_interval', type=int, default=2, help="determines the test intervals for the first_years train_type")
    # parser.add_argument('--train_hospitals', type=str, nargs='+', default=["UPMCPUH"], help="choices: lsit of e.g. [UPMCPUH, UPMCSHY, UPMCMER]")
    # parser.add_argument('--test_hospitals', type=str, nargs='+', default=None, help="choices: None or list of e.g. [UPMCPUH, UPMCSHY, UPMCMER]")
    # parser.add_argument('--train_icu_types', type=str, nargs='+', default=["CTICU"], help="choices: lsit of e.g. [CTICU, MICU]")
    # parser.add_argument('--test_icu_types', type=str, nargs='+', default=["CTICU", "MICU"], help="choices: None or list of e.g. [CTICU, MICU]")
    # parser.add_argument('--load_filtered_data', type=int, default=0, help="loads stored data that includes features and outcomes, and common_indices. 0: False, 1: True")
    # parser.add_argument('--site_name', type=str, default=None, choices=[None, 'UPMCPUH', 'UPMCSHY', 'CTICU', 'MICU'], help="required if train_type=single_site")
    # parser.add_argument('--save_data', type=int, default=0, help="saves the processed data to output_dir. 0: False, 1: True")
    # parser.add_argument('--feature_selection', type=int, default=0, help="run univariate feature selection. 0: False, 1: True")
    # parser.add_argument('--K', type=int, nargs='+', default=None, help="number of features to select (list of int)")
    # parser.add_argument('--train_years', type=int, nargs='+', default=[2008,2009,2010], help="2008-2014")
    # parser.add_argument('--n_bootstrap', type=int, default=10, help="n_bootstrap runs")


    # args = parser.parse_args()
    
    # if isinstance(args.random_seed, int):
    #     args.random_seed=[args.random_seed]

    # if isinstance(args.representation, str):
    #     args.representation=[args.representation]

    # if isinstance(args.model_types, str):
    #     args.model_types=[args.model_types]

    # if isinstance(args.target_list, str):
    #     args.target_list=[args.target_list]

    # if isinstance(args.train_types, str):
    #     args.train_types=[args.train_types]

    # if isinstance(args.train_hospitals, str):
    #     args.train_hospitals=[args.train_hospitals]

    # if isinstance(args.test_hospitals, str):
    #     args.test_hospitals=[args.test_hospitals]

    # if isinstance(args.train_icu_types, str):
    #     args.train_icu_types=[args.train_icu_types]

    # if isinstance(args.test_icu_types, str):
    #     args.test_icu_types=[args.test_icu_types]

    # if isinstance(args.train_years, int):
    #     args.train_years=[args.train_years]

    # if isinstance(args.gpu, int):
    #     args.gpu=[args.gpu]

    # if 'single_site' in args.train_types:
    #     assert args.site_name is not None, "site_name is required for train_type='single_site'"

    # feature_selection_args={}
    # if args.feature_selection:
    #     if isinstance(args.K, int):
    #         args.K=[args.K]
    #     feature_selection_args={'K': args.K}

    # n_threads=args.n_threads
    # output_dir=args.output_dir

    # # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # # os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(gpu) for gpu in args.gpu])  # specify which GPU(s) to be used
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # filtered_df=None
    # label_df=None
    # common_indices=None
    # demo_df=None
    # years_df=None


    # print("Loading the data.")
    # load_data(max_time=args.max_time, data_dir=args.data_dir, load_filtered_data=args.load_filtered_data)

    # if not args.load_filtered_data:
    #     save_filtered_data(args.data_dir)


    # print("Loading the  years.")
    # read_years_data()

    # print("Loading site info")
    # read_sites_data(data_dir=args.data_dir)

    # if np.isin(args.model_types, ['lstm', 'gru']).any():
    #     import tensorflow as tf
    #     from keras.layers import Input, LSTM, GRU, Dense, Bidirectional, merge
    #     from keras.models import Model
    #     from keras import backend as K
    #     try:
    #         from EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
    #     except:
    #         from utils.EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
    #     config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
    #     config.gpu_options.allow_growth = True  # Don't use all GPUs
    #     config.allow_soft_placement = True  # Enable manual control
    #     sess = tf.Session(config=config)
    #     K.set_session(sess)

    # for train_type in args.train_types:
    #     print("#"*100)
    #     print("train_type= ", train_type)
    #     print("#"*100)

    #     for target in args.target_list:
    #         print("#"*100)
    #         print("target= ", target)
    #         print("#"*100)

    #         for representation in args.representation:
    #             print("#"*100)
    #             print("representation= ", representation)
    #             print("#"*100)
    #             for seed in args.random_seed:
    #                 print("#"*100)
    #                 print("seed= ", seed)
    #                 print("#"*100)
    #                 embedded_model=None
    #                 scaler=None
    #                 best_params=None
    #                 keep_cols=None
    #                 if train_type=='rolling_limited':
    #                     main_rolling_limited(random_seed=seed, max_time=args.max_time, test_size=args.test_size, level=args.level, representation=representation,
    #                     target=target, prefix=args.prefix, model_types=args.model_types, data_dir=args.data_dir, output_dir=args.output_dir)
    #                 elif train_type=='rolling':
    #                     main_rolling(random_seed=seed, max_time=args.max_time, test_size=args.test_size, level=args.level, representation=representation, 
    #                                 target=target, prefix=args.prefix, model_types=args.model_types, data_dir=args.data_dir, output_dir=args.output_dir)
    #                 elif train_type=='no_years':
    #                     main_no_years(random_seed=seed, max_time=args.max_time, test_size=args.test_size, level=args.level, representation=representation,
    #                     target=target, prefix=args.prefix, model_types=args.model_types, data_dir=args.data_dir, output_dir=args.output_dir)
    #                 elif train_type=='overall_overtime':
    #                     main_overtime_overall(random_seed=seed, max_time=args.max_time, test_size=args.test_size, 
    #                         level=args.level, representation=representation, target=target, 
    #                         prefix=args.prefix, model_types=args.model_types, data_dir=args.data_dir, training_years=args.train_years, output_dir=args.output_dir, test_month_interval=args.test_month_interval,
    #                         save_data=args.save_data, feature_selection=args.feature_selection, **feature_selection_args)

    #                 elif train_type=='hospital_wise':
    #                     main_hospital_wise(random_seed=seed, max_time=args.max_time, 
    #                         level=args.level, representation=representation, target=target, 
    #                         prefix=args.prefix, model_types=args.model_types, data_dir=args.data_dir, 
    #                         train_hospitals=args.train_hospitals, test_hospitals=args.test_hospitals, test_size=0.2, output_dir=args.output_dir, save_data=args.save_data)

    #                 elif train_type=='icu_type':
    #                     main_icu_type(random_seed=seed, max_time=args.max_time, 
    #                         level=args.level, representation=representation, target=target, 
    #                         prefix=args.prefix, model_types=args.model_types, data_dir=args.data_dir, 
    #                         train_icu_types=args.train_icu_types, test_icu_types=args.test_icu_types, test_size=0.2, output_dir=args.output_dir)

    #                 elif train_type=='single_site':
    #                     main_single_site(site_name=args.site_name, random_seed=seed, max_time=args.max_time, 
    #                         level=args.level, representation=representation, target=target, prefix=args.prefix, model_types=args.model_types, 
    #                         data_dir=args.data_dir, training_years=args.train_years, output_dir=args.output_dir, test_month_interval=args.test_month_interval,
    #                         save_data=args.save_data, feature_selection=args.feature_selection, **feature_selection_args)
    #                 elif train_type=='hospital_overtime':
    #                     main_hospital_overtime(random_seed=seed, max_time=args.max_time, level=args.level, representation=representation,
    #                         test_month_interval=args.test_month_interval, training_years=[2008, 2009, 2010], target=target, prefix=args.prefix, 
    #                         model_types=args.model_types,  data_dir=args.data_dir, train_hospitals=args.train_hospitals, output_dir=args.output_dir, 
    #                         test_hospitals=args.test_hospitals, save_data=args.save_data, feature_selection=args.feature_selection, **feature_selection_args)

    #                 elif train_type=='hospital_pairwise':
    #                     main_hospital_pairwise(train_hospitals=args.train_hospitals, test_hospitals=args.test_hospitals, target=target, 
    #                         model_types=args.model_types, representation=representation, test_size=0.2, max_time=args.max_time, level=args.level, 
    #                         prefix=args.prefix, data_dir=args.data_dir, output_dir=args.output_dir, save_data=args.save_data, random_seed=seed, 
    #                         feature_selection=args.feature_selection, **feature_selection_args)

    #                 elif train_type=='hospital_pairwise_bootstrap':
    #                     main_hospital_pairwise_bootstrap(n_bootstrap=args.n_bootstrap, train_hospitals=args.train_hospitals, test_hospitals=args.test_hospitals, target=target, 
    #                         modeltype=args.model_types[0], representation=representation, test_size=0, max_time=args.max_time, level=args.level, 
    #                         prefix=args.prefix, data_dir=args.data_dir, output_dir=args.output_dir, save_data=args.save_data, random_seed=seed, 
    #                         feature_selection=args.feature_selection, **feature_selection_args)

    # t1=time.time()
    # print("Total run-time =  {:10.1f} seconds.".format(t1-t0))
