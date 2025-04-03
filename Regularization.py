#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import module
import sys
import pandas as pd
import numpy as np
from scipy.stats import entropy
import os, gzip, shutil

command_arg = sys.argv
file_name = command_arg[0]
num_sequences = int(command_arg[1]) #2
num_sets = int(command_arg[2]) #8
num_group = int(command_arg[3]) #71
condition = int(command_arg[4]) #3
# 1 -> Adult-pair
# 2 -> Child-pair
# 3 -> Adult-Child
len_sequence = int(command_arg[5]) #13

num_trial = num_sequences*num_sets

Sequences = [f"c{i+1}" for i in range(len_sequence)]


# read data
df = pd.read_csv("3_merged/Hierarchy_cond3.csv")
df_regular = df[['sequence','set_number', 'within_trial', 'individual_ID', 'pair_ID', 'age','age_f','generation','condition']].query('within_trial!=0').copy()


#%%
# Calculate Accuracy
def Accuracy(strings):
    """
    Calculate Levenshtein distance between two strings
    All costs (insertion, deletion, and replacement) is 1
    error rate is normalized Levenshtein distance (devided by string's length)
    Accuracy = 1- error
    # argument: strings -> pandas.DataFrame
    # return: Accuracy -> float
    """
    s1 = strings.iloc[0]
    s2 = strings.iloc[1]
    
    if not (s1.isalpha() and s2.isalpha()):
        return -1
    
    n, m = len(s1), len(s2)
    dp = [[0] * (m+1) for _ in range(n+1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,         # insertion
                           dp[i][j - 1] + 1,         # deletion
                           dp[i - 1][j - 1] + cost)  # replacement
    error =  dp[n][m]/len(s1)
    Accuracy = 1-error
    return Accuracy

#%%

array_regular=np.zeros(len(df_regular))
count=0
for i in range(num_group): # pair_ID
    for j in [0,1]: # individual_ID
        for k in range(1,num_sequences+1): # number of interaction block (set)
            for l in range(1,num_sets+1): # within_trial
                if l!=1:
                    df_temp=df_regular.query(f"individual_ID=={j}&pair_ID=={i}&set_number=={k}&{l-1}<=within_trial<={l}")
                    array_regular[count]=Accuracy(df_temp['sequence'])
                    count += 1
                else:
                    count += 1

df_regular["regularization"] =array_regular

df_regular.to_csv("3_merged/df_regular.csv",index=False)
