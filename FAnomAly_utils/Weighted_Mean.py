#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

def Weighted_Mean_general(group, flux_col='dc_flux', sigflux_col='dc_sigflux'):
    N = len(group)
    if N == 1:
        return pd.Series({flux_col: group[flux_col].iloc[0], sigflux_col: group[sigflux_col].iloc[0]})
    
    wi = 1 / group[sigflux_col] ** 2

    x_bar = (group[flux_col] * wi).sum() / wi.sum()
    
    x_2 = ((group[flux_col] - x_bar) ** 2 * wi).sum()
    
    sigma_x = np.sqrt(1 / wi.sum())
    
    fact = np.sqrt(x_2 / (N - 1))
    
    if fact >= 1:
        sigma_x *= fact
    return pd.Series({flux_col: x_bar, sigflux_col: sigma_x})

def Weighted_Mean_all(df):
    N= len(df)

    wi_dc = 1/df['dc_sigflux']**2
    wi_nr = 1/df['nr_sigflux']**2

    x_bar_dc = (df['dc_flux']*wi_dc).sum() / wi_dc.sum()
    x_bar_nr = (df['nr_flux']*wi_nr).sum() / wi_nr.sum()
        
    return x_bar_dc, x_bar_nr
