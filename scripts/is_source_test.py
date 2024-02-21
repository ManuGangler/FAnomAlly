#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:09:52 2024

@author: mohamadjouni
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fink_utils.photometry.conversion import dc_mag
from fink_utils.photometry.utils import is_source_behind


def test_is_source(pdf, Id):
    global k
    
    pdf_selectionne = pdf.loc[pdf['objectId'] == Id]
    candidate_df = pdf_selectionne['candidate'].apply(pd.Series)

    index_max_jd = candidate_df['jd'].idxmax()
    pdf_selectionne = pdf_selectionne.loc[index_max_jd]

    pdf_selectionne_cand = pdf_selectionne['prv_candidates'] 

    keys = pdf_selectionne_cand[0].keys()
    actual_cand = {key: pdf_selectionne['candidate'][key] for key in keys if key in pdf_selectionne['candidate']}

    liste_dicts = list(pdf_selectionne_cand)
    liste_dicts.append(actual_cand)
    df = pd.DataFrame(liste_dicts)

    maskValid = (df['rb'] > 0.55) & (df['nbad'] == 0)
    df_valid = df[maskValid].sort_values('jd')

    isSource = is_source_behind(
        df_valid['distnr'].values[0]
    )

    if isSource:
        pass
    else:
        print('No source found -- keeping PSF fit magnitude')
        print(pdf_selectionne.objectId)
        k = k+ 1


def main():
    pdf = pd.read_parquet('/Users/mohamadjouni/work/ftransfer_ztf_2024-02-01_689626')
    unique_ids = pdf['objectId'].unique().tolist()
    global k 
    k = 0
    
    for Id in unique_ids[0:500]:
        test_is_source(pdf, Id)
    
    print(k)


if __name__ == "__main__":
    main()
