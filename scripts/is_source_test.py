#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:09:52 2024

@author: mohamadjouni
"""


# In[1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



pdf = pd.read_parquet('/Users/mohamadjouni/work/ftransfer_ztf_2024-02-01_689626')
# In[2]

unique_ids = pdf['objectId'].unique().tolist()
#print(len(unique_ids))


# In[3]

k =1


def test_is_source(Id):
    global k
    
    pdf_selectionne = pdf.loc[pdf['objectId'] == Id]
    candidate_df = pdf_selectionne['candidate'].apply(pd.Series)




    # index of the candidate with the biggest 'jd'
    index_max_jd = candidate_df['jd'].idxmax()

    # select this candidate
    pdf_selectionne = pdf_selectionne.loc[index_max_jd]


    pdf_selectionne_cand = pdf_selectionne['prv_candidates'] 



    #  add 'candidate' the actual value 
    keys = pdf_selectionne_cand[0].keys()
    actual_cand = {key: pdf_selectionne['candidate'][key] for key in keys if key in pdf_selectionne['candidate']}



    liste_dicts = list(pdf_selectionne_cand)
    liste_dicts.append(actual_cand)
    df = pd.DataFrame(liste_dicts)



    from fink_utils.photometry.conversion import dc_mag
    from fink_utils.photometry.utils import is_source_behind

    maskValid = (df['rb'] > 0.55) & (df['nbad'] == 0)
    df_valid = df[maskValid].sort_values('jd')

    isSource = is_source_behind(
    df_valid['distnr'].values[0]
    )

    if isSource:
      #print('It looks like there is a source behind. Lets compute the DC magnitude instead.')
      pass
    else:
      print('No source found -- keeping PSF fit magnitude')
      print(pdf_selectionne.objectId)
      k =k + 1

      
      
      

# can be optimized by removing the function !

for Id in unique_ids:
    
     test_is_source(Id)
print(k)