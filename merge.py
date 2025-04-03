#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:34:17 2023
@author: user


"""
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
num_group = int(command_arg[3])
condition = int(command_arg[4])
# 1 -> Adult-pair
# 2 -> Child-pair
# 3 -> Adult-Child
len_sequence = int(command_arg[5])

num_trial = num_sequences*num_sets

Sequences = [f"c{i+1}" for i in range(len_sequence)]
columns = ["sequence","sequence_ID","trial","individual_ID","pair_ID","age","condition","accuracy"]

def add_set_number(trial,num_sets):
    set_number = int((trial-1)/num_sets)+1
    return set_number

# merge data
df = pd.DataFrame(columns=columns)
df_displayed = pd.DataFrame(columns=columns)

df_demographic = pd.read_csv("1_otree-data/demographic.csv")

for n in range(num_group):
    pair_ID = n
    player_ID = 1000*condition+pair_ID*10
    age = df_demographic.query(f"ID=={player_ID}")["age"].values[0]
    temp1 = pd.read_csv(f"2_dataframes/{player_ID}.csv").query(f"trial<={num_trial}")
    seq=pd.DataFrame(temp1["c1"].str.cat(temp1[Sequences[1:]])).rename(
        columns={"c1":"sequence"})
    temp1 = pd.concat([seq,temp1[["sequence_ID","trial","individual_ID","pair_ID","condition","accuracy"]]],axis=1)
    # assign distinguishable ID
    temp1["sequence_ID"] = player_ID*1000 + temp1["trial"]
    temp1["age"] = age
    df = pd.concat([df,temp1])

    temp1_displayed = pd.read_csv(f"2_dataframes/displayed_{player_ID}.csv").query(f"trial<={num_trial}")
    seq=pd.DataFrame(temp1_displayed["c1"].str.cat(temp1_displayed[Sequences[1:]])).rename(
        columns={"c1":"sequence"})
    temp1_displayed = pd.concat([seq,temp1_displayed[["sequence_ID","trial","individual_ID","pair_ID","condition","accuracy"]]],axis=1)
    # assign distinguishable ID
    temp1_displayed["sequence_ID"] = player_ID*1000 + temp1_displayed["trial"]
    temp1_displayed["age"] = age
    df_displayed = pd.concat([df_displayed,temp1_displayed])

    player_ID += 1
    age = df_demographic.query(f"ID=={player_ID}")["age"].values[0]
    temp2 = pd.read_csv(f"2_dataframes/{player_ID}.csv").query(f"trial<={num_trial}")
    seq=pd.DataFrame(temp2["c1"].str.cat(temp2[Sequences[1:]])).rename(
        columns={"c1":"sequence"})
    temp2 = pd.concat([seq,temp2[["sequence_ID","trial","individual_ID","pair_ID","condition","accuracy"]]],axis=1)
    # assign distinguishable ID
    temp2["sequence_ID"] = player_ID*1000 + temp2["trial"]
    temp2["age"] = age
    df = pd.concat([df,temp2])

    temp2_displayed = pd.read_csv(f"2_dataframes/displayed_{player_ID}.csv").query(f"trial<={num_trial}")
    seq=pd.DataFrame(temp2_displayed["c1"].str.cat(temp2_displayed[Sequences[1:]])).rename(
        columns={"c1":"sequence"})
    temp2_displayed = pd.concat([seq,temp2_displayed[["sequence_ID","trial","individual_ID","pair_ID","condition","accuracy"]]],axis=1)
    # assign distinguishable ID
    temp2_displayed["sequence_ID"] = player_ID*1000 + temp2_displayed["trial"]
    temp2_displayed["age"] = age
    df_displayed = pd.concat([df_displayed,temp2_displayed])

#%%
#Accuracy
df_displayed = df_displayed.drop_duplicates()
df_displayed = df_displayed.reset_index(drop=True)
df_displayed["set_number"] = df_displayed["trial"].apply(add_set_number, num_sets=num_sets).reset_index(drop=True)
df_displayed["within_trial"] = [i+1 for i in range(num_sets)]*num_sequences*2*num_group
# change sequence_ID to avoid duplicate IDs in df
df_displayed["sequence_ID"] = df_displayed["sequence_ID"]*2
df_displayed["generation"] = df_displayed["age"].map((lambda x: "Adult" if x >= 18 else "Child"))

df_displayed.to_csv(f"3_merged/displayed_cond{condition}.csv",index=False)

df_seed = df_displayed.query("within_trial==1")[['sequence', 'sequence_ID', 'trial', 'individual_ID', 'pair_ID','age','condition', 'accuracy', 'set_number']]
df_seed['trial'] = 0
df_seed['accuracy'] = 0
df_seed['within_trial'] = 0
df_seed = df_seed.reset_index(drop=True)

df = df.drop_duplicates()
df = df.reset_index(drop=True)
df["set_number"] = df["trial"].apply(add_set_number, num_sets=num_sets).reset_index(drop=True)
df['within_trial'] = [i+1 for i in range(num_sets)]*num_sequences*2*num_group

df = pd.concat([df,df_seed]).reset_index(drop=True)
df = df.reset_index(drop=True)
df["generation"] = df["age"].map((lambda x: "Adult" if x >= 18 else "Child"))



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
#Similarity within pair
df_pair = pd.concat([df.query("individual_ID==0")["sequence"].reset_index(drop=True)
                    ,df.query("individual_ID==1")["sequence"].reset_index(drop=True)]
                    ,axis=1)
df_within_sim = pd.concat([df_pair.apply(Accuracy,axis=1)
                           ,df.query("individual_ID==0")[["trial", "pair_ID", "condition","set_number"]]],axis=1)
df_within_sim = df_within_sim.rename(columns={0:"within-pair-similarity"})

#%%
# compression ratio

"""Define functions"""
#Calculate "Commpression ratio"
def Compression_ratio(data):
    """
    Calculate "Commpression ratio"
    if compress = True, write out each string and compress text file to gzip
    Size of compressed file ＝ compressed bytes ＋ 24bytes ＋ length of file name」
    # argument: sereies -> pandas.DataFrame
    """
    sequence_ID = data["sequence_ID"]
    #print(sequence_ID)
    sequence = data["sequence"]
    #print(sequence)
    
    fname = f"output{sequence_ID}.txt"
    #write out each string and save text file
    os.makedirs("Compress", exist_ok=True)
    if not os.path.isfile(f"Compress/{fname}"):
        with open(f"Compress/{fname}", mode="w") as f_out:
            f_out.write(sequence.rstrip("\n"))
        #gzip compression
        with open(f"Compress/{fname}", "rb") as f_in:
            with gzip.open(f"Compress/{fname}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    #get file size
    after = os.path.getsize(f"Compress/{fname}.gz") - (len(fname)+24)#must be 44, Remove the extra data size for unzip
    before = os.path.getsize(f"Compress/{fname}")
    #print(after)
    #print(before)
    ratio = after / before
    
    return ratio

df_compression = df.apply(Compression_ratio,axis=1).reset_index(drop=True)
df["Compression_ratio"] = df_compression


#%%
def SequiturHierarchy(sequence, rule_utility=True):
    """
    Grammar compression algorithm (Nevill-manning & Whitten, 1997)
    Find digram (bigram) in string and replace new rule (e.g. 0→ab)
    This process makes hierarchical construction by-product
    # argument: sequence -> string
    # return: S -> string
              Rules -> dictionary
              depth -> int (maximum hierarchy depth)
              max_chunk_size -> int (size of the largest chunk)
              num_chunks -> int (number of chunks in final sequence, excluding letters)
    """
    S = ""
    symbpl_number = 0
    Rules = {}  # {Digram:Rule}
    depths = {}  # Store depth for each rule
    chunk_sizes = {}  # Store the size of each chunk
    
    for symbol in sequence:
        symbpl_number += 1
        S += symbol
        if symbpl_number >= 4:  # don't check digram until 4 characters
            while True:
                Digram = S[-2:]
                if Digram in S[:-2]:
                    new_Rule = str(len(Rules))
                    S = S.replace(Digram, new_Rule)
                    Rules[Digram] = new_Rule
                elif Digram in Rules.keys():
                    S = S.replace(Digram, Rules[Digram])
                else:
                    break
        else:
            pass

    # enforce rule utility
    if rule_utility:
        digrams = list(Rules.keys())
        rules = list(Rules.values())
        digrams.append(S)
        all_str = "".join(digrams)
        non_unique = {}

        # search non_unique rules
        for rule in rules:
            if all_str.count(rule) > 1:
                pass
            else:
                once_key = [k for k, v in Rules.items() if v == rule][0]
                non_unique[once_key] = rule
                del Rules[once_key]
        
        non_unique = dict(reversed(list(non_unique.items())))
        if any(non_unique):
            for nk, nv in non_unique.items():
                key = [k for k, v in Rules.items() if nv in k][0]
                value = Rules[key]
                del Rules[key]
                key = key.replace(nv, nk)
                Rules[key] = value
    
    # Compute hierarchy depth recursively
    def compute_depth(rule_id):
        for digram, rid in Rules.items():
            if rid == rule_id:
                return 1 + max((compute_depth(part) if part in Rules.values() else 0 for part in digram), default=0)
        return 0  # Base case: no chunk
    
    max_depth = max((compute_depth(rule) for rule in Rules.values()), default=0)
    
    # Compute max chunk size by expanding rules recursively
    def compute_chunk_size(rule_id):
        for digram, rid in Rules.items():
            if rid == rule_id:
                return sum(compute_chunk_size(part) if part in Rules.values() else len(part) for part in digram)
        return 0  # Base case: no chunk
    
    max_chunk_size = max((compute_chunk_size(rule) for rule in Rules.values()), default=0)
    
    # Compute number of chunks in final S (excluding letters)
    num_chunks = sum(1 for ch in S if ch.isdigit())
    
    return S, Rules, max_depth, max_chunk_size, num_chunks

df_hierarchy2 = df["sequence"].apply(SequiturHierarchy).reset_index(drop=True)
df["Depth_of_hierarchy2"] = df_hierarchy2.apply(lambda x: x[2])
df["Number_of_chunks"] = df_hierarchy2.apply(lambda x: x[3])
df["Size_of_chunks"] = df_hierarchy2.apply(lambda x: x[4])


def shannon_entropy(s):
    char_list = list(s) # convert strings to list    
    # Caluculate frequency of each character (color)
    char_freq = {char: char_list.count(char) / len(char_list) for char in set(char_list)}
    entropy_value = entropy(list(char_freq.values()))    
    return entropy_value

df_entropy = df["sequence"].apply(shannon_entropy).reset_index(drop=True)
df["Entropy"] = df_entropy.astype('object')

df["age_f"] = df["age"]

def assign_value(generation):
    if all(gen == "Adult" for gen in generation):
        return 1  # Adult-Adult
    elif all(gen == "Child" for gen in generation):
        return 2  # Child-Child
    else:
        return 3  # Adult-Child

df = df.drop(columns=["condition"], errors="ignore")

result = (
    df.groupby("pair_ID")["generation"]
    .apply(assign_value)
    .reset_index(name="condition")
)

df = df.merge(result, on="pair_ID", how="left")
df.to_csv(f"3_merged/Hierarchy_cond{condition}.csv",index=False)





