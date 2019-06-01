# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 10:20:30 2019

@author: Luke
"""

# COMBINE READING DATA, CLEAN DATA, AND LOGISTIC REGRESSION

# Import libraries
import pandas as pd
import math as m
from os import listdir
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics

###############################################################################

# GET DATA

# Directory containing project files
DIR = r'C:/Users/Luke/Desktop/_SU19 Files/Coding Project'

# Name of folder in DIR containing csv data
FOLDER_CSV = 'All Data'

# Get names of all files in directory
file_names = listdir(DIR + '/' + FOLDER_CSV)

# Columns to drop 
# month, year, and app_bundle have the same value for all observations
# creative_size encodes the same information as creative_type, so drop 
# creative_size
# day_of_week will be overridden after conversion to local datetime
drop_cols = ['month', 'year', 'app_bundle', 'creative_size', 'day_of_week']

# Initialize the dataframe with the first file, consisting of just column names
df = pd.read_csv(DIR + '/' + FOLDER_CSV + '/2019-04-00.csv')

# Drop unnecessary columns
df = df.drop(drop_cols, axis=1)

# Drop the first file name (headers only) from file_names
file_names.remove('2019-04-00.csv')

# Read the remaining data and append to df
# This takes about 5 minutes
for i in file_names:
    # Read daily csv
    temp = pd.read_csv(DIR + '/' + FOLDER_CSV + '/' + i)
    
    # For computational simplicity, downsample to 1/20 of daily data
    temp = temp.sample(m.floor(temp.shape[0]/20), random_state = 0)
    
    # Drop unnecessary columns
    temp = temp.drop(drop_cols, axis=1)
    
    # Split daily csv into two dataframes: click and no click
    # Note: observations for which clicks = 0 & installs = 1 should actually
    # read clicks = 1 & installs = 1
    click = temp.loc[(temp.clicks == 1) | (temp.installs == 1)]
    click.loc[:, 'clicks'] = 1.0
    no_click = temp[(temp.clicks == 0) & (temp.installs == 0)]
    
    # Downsample no click to be the same size as click
    # For reproducibility, set random_state to 0
    df = pd.concat([
            df, 
            click,
            no_click.sample(n = click.shape[0], random_state = 0)
            ], axis=0)
    
    # Visually track progress by printing the file name
    print(i)

# Save dataset as pickle
df.to_pickle(DIR + '/raw_df.pickle')

# Test reading data back in
test = pd.read_pickle(DIR + '/raw_df.pickle')

###############################################################################

# CLEAN DATA

# Read zipcode file and delete 3 or 4 digit zipcodes
ZIP = pd.read_csv(DIR + '/zipcode/zipcode.csv')
zipdf = ZIP[['zip','state','timezone','dst']]
zipdf = zipdf[zipdf.zip > 9999]

# Load data
df = pd.read_pickle(DIR + '/raw_df.pickle')

df.reset_index(inplace=True)

df = pd.merge(
        df, 
        zipdf, 
        left_on='geo_zip', 
        right_on='zip',
        how='left'
        )

# Initialize column 'local_hour' as int with the same value as 'hour'
hours = df['hour']
hours_list = hours.tolist()
hour_num = list(int(x[0:2]) for x in hours_list)
df['local_hour'] = hour_num

df = df.assign(local_hour=df.local_hour+df.timezone+df.dst)

df['day'] = np.where(df['local_hour'] < 0, df['day'] - 1, df['day'])
df['local_hour'] = np.where(df['local_hour'] < 0, df['local_hour'] + 24, df['local_hour'])

df = df.assign(diff_hour=df.timezone+df.dst)

# TODO: append day of week

# Identify columns with NA or NaN
df.columns[df.isna().any()]

# Replace NA/NaN with the string 'NA'
df = df.fillna(value = {'category': 'NA',
#                        'geo_zip': 'NA',
                        'platform_bandwidth': 'NA',
                        'platform_carrier': 'NA',
                        'platform_device_screen_size': 'NA',
                        'creative_type': 'NA'})
    
# Reformat the category column to be one category per column    
expand_category = df['category'].str.split(',', expand = True) 
expand_category = pd.concat([df.auction_id, expand_category], axis=1)

expand_category = pd.melt(expand_category, id_vars = ['auction_id']) 

expand_category = expand_category[expand_category['value'].notnull()]

expand_category = expand_category.pivot_table(
        index='auction_id',
        columns='value',
        aggfunc='size'
        )

expand_category.columns = ['category_' + str(col) for col in expand_category.columns]
expand_category = expand_category.fillna(0)

df = pd.merge(
        df, 
        expand_category, 
        left_on='auction_id', 
        right_index=True
        )

# Reformat the segment column to be one segment per column    
df['segments'] = df['segments'].str.replace(r"\[","")
df['segments'] = df['segments'].str.replace(r"\]","")
    
expand_segment = df['segments'].str.split(', ', expand = True)
expand_segment = pd.concat([df.auction_id, expand_segment], axis=1)

expand_segment = pd.melt(expand_segment, id_vars = ['auction_id']) 

expand_segment = expand_segment[expand_segment['value'].notnull()]

expand_segment = expand_segment.pivot_table(
        index='auction_id',
        columns='value',
        aggfunc='size'
        )

expand_segment.columns = ['segment_' + str(col) for col in expand_segment.columns]
expand_segment = expand_segment.fillna(0)

df = pd.merge(
        df, 
        expand_segment, 
        left_on='auction_id', 
        right_index=True
        )

# Save dataset as pickle
df.to_pickle(DIR + '/clean_df.pickle')

###############################################################################

# REGRESSIONS

#X = df.loc[:,'segment_':'segment_9']
#X = df.loc[:,'category_-1':'category_NA']
X = df.loc[:,'category_-1':'segment_9']
#y = df.loc[:,'clicks']
y = df.loc[:,'installs']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

model = LogisticRegression(class_weight = 'balanced').fit( X_train, y_train )
#model = LinearRegression().fit( X_train, y_train )

# HOW WELL DOES THE MODEL PREDICT THE TRAINING DATA????
#predicted = model.predict(X_train)
#print(metrics.confusion_matrix(y_train, predicted))
predicted = model.predict(X_test)
print(metrics.confusion_matrix(y_test, predicted))

#model.score( X_train, y_train )
model.score( X_test, y_test )

# TEST DIFFERENT THRESHOLDS
#pred_proba_df = pd.DataFrame(model.predict_proba(X_test))
#threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
#for i in threshold_list:
#    print ('\n******** For i = {} ******'.format(i))
#    Y_test_pred = pred_proba_df.applymap(lambda x: 1 if x>i else 0)
#    test_accuracy = metrics.accuracy_score(y_test.as_matrix().reshape(y_test.as_matrix().size,1),
#                                           Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1))
#    print('Our testing accuracy is {}'.format(test_accuracy))
#
#    print(metrics.confusion_matrix(y_test.as_matrix().reshape(y_test.as_matrix().size,1),
#                           Y_test_pred.iloc[:,1].as_matrix().reshape(Y_test_pred.iloc[:,1].as_matrix().size,1)))