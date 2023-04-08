#!/usr/bin/env python3

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tqdm import tqdm
from sys import argv
from chembl_webresource_client.new_client import new_client

# setting max number of dataframe rows 
# add in connectivity issue
pd.set_option('display.max_rows', 100)


# Instructions on the programme
input("""\nWelcome, this programme is used to estimate the IC50 of a compound based on it's smiles notation.\nThis programme works by building a logistic regression model from the CHEMBL database.
      \nTo run this programme, you need to search the CHEMBL database for the same biological target as the smiles query.\nThen, build the model by customising the IC50 value which will split the dataset for the model into active or inactive.
      \nFor whatever value you enter, any value above, but not including that, will be considered as inactive.
      \nTo continue, press ENTER: """)
# Search the CHEMBL database for biological targets
def search_query():
    
    target = new_client.target
    user_search = input("\nName the compound or biological target you want to search CHEMBL for: ")
    if user_search == '':
        return search_query()
    elif len(user_search) < 3:
        print("\nSearch too short")
        return search_query()
    try: 
        target_query = tqdm(target.search(user_search), desc="\nSearching", bar_format="{desc}:{r_bar}")
        targets = pd.DataFrame.from_dict(target_query)
        targets = targets[targets['target_type'] == 'SINGLE PROTEIN'].reset_index(drop=True)
        print(targets)
    except (ValueError, AttributeError) as e:
        print(f"Input Error: {str(e)}, try again")
        return search_query()
    
    try:
        user_choice = int(input("\nSelect the organis row ID you want to use as a target\nHit ENTER to search again: "))
        selected_target = targets.target_chembl_id[user_choice]
        print(selected_target)

    except(ValueError, KeyError, AttributeError):
        return search_query()
    
    activity = new_client.activity
    res = tqdm(activity.filter(target_chembl_id=selected_target).filter(standard_type='IC50'), desc="\nFiltering data", bar_format="{desc}:{r_bar}")
    df = pd.DataFrame.from_dict(res)

    print(df)

    should_continue = input("\nDo you want to continue with this selection(y/n): ")

    if should_continue =='n':
        return search_query()
    elif should_continue == 'y':
        pass

        if len(df) < 10:
            print("Dataset too small.")
            return search_query()
        
    else:
        print("\nInvalid input")
        return search_query()
    

    df.dropna(subset=['standard_value', 'canonical_smiles'], inplace=True)



    df2_nr = df.drop_duplicates(['canonical_smiles'])

    selection = ['canonical_smiles', 'standard_value']
    df3 = df2_nr[selection]

    data = df3

    return data

data = search_query()
data['standard_value'] = pd.to_numeric(data['standard_value'], errors='coerce')
data = data.drop(data[data.standard_value < 0 ].index, axis=0).reset_index(drop=True)
data = data.drop(data[data.standard_value == ''].index, axis=0).reset_index(drop=True)
print(data)
print(f"\nDataset enteries: {len(data)}")
IC_50_mean = data['standard_value'].mean()
IC_50_max = data['standard_value'].max()
IC_50_min = data['standard_value'].min()
IC_50_median = data['standard_value'].median()

print(f'\nDataset IC50 Mean: {IC_50_mean} nM\nDataset IC50 Min: {IC_50_min} nM\nDataset IC50 Max: {IC_50_max} nM\nDataset IC50 Median: {IC_50_median} nM\n')

data.to_csv("Training_Dataset.csv", index=False)

