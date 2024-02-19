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

Id = pdf['objectId'][4771]

#id_plus_repete = pdf['objectId'].value_counts().idxmax()

# here opting for the most frequently occurring alert.
#pdf_selectionne = pdf.loc[pdf['objectId'] == id_plus_repete]
pdf_selectionne = pdf.loc[pdf['objectId'] == Id]

list_Ids = [pdf_selectionne]
#list_Ids.append(pdf_selectionne)
print(list_Ids)



top_15 = pdf['objectId'].value_counts().index.tolist()
print(len(top_15))
for i in top_15:
    j = pdf.loc[pdf['objectId'] == i]
    
    list_Ids.append(j)
print("done")
# In[3]

k =1


def test_is_source(pdf_selectionne):
    global k
    #print(k)
    #k =k + 1
    candidate_df = pdf_selectionne['candidate'].apply(pd.Series)



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

      
      
      



for pdf_selec in list_Ids:
     test_is_source(pdf_selec)    #"""  
print(k)