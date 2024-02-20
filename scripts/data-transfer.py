#!/usr/bin/env python
# -*- coding: utf-8 -*-


# In[1]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fink_utils.photometry.conversion import apparent_flux
from FAnomAly_utils.flux import flux_nr
from FAnomAly_utils.flux import apparent_flux_Upper
from FAnomAly_utils.Weighted_Mean import Weighted_Mean_general
from FAnomAly_utils.Weighted_Mean import Weighted_Mean_all



pdf = pd.read_parquet('/Users/mohamadjouni/work/ftransfer_ztf_2024-02-01_689626')
# In[2]
unique_ids = pdf['objectId'].unique().tolist()



def function_FN(Id):
    
    #print(Id)
    
    pdf_selectionne = pdf.loc[pdf['objectId'] == Id]

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

# Take only valid measurements
    maskValid = (df['rb'] > 0.55) & (df['nbad'] == 0)
    df_valid = df[maskValid].sort_values('jd')

    isSource = is_source_behind(
    df_valid['distnr'].values[0]
    )

    if isSource:
      #############print('It looks like there is a source behind. Lets compute the DC magnitude instead.')
    
    # Use DC magnitude instead of difference mag
      mag_dc, err_dc = np.transpose(
        [
            dc_mag(*args) for args in zip(
                df_valid['magpsf'].astype(float).values,
                df_valid['sigmapsf'].astype(float).values,
                df_valid['magnr'].astype(float).values,
                df_valid['sigmagnr'].astype(float).values,
                df_valid['isdiffpos'].values
            )
        ]
    )
    
      df_valid['mag_dc'] = mag_dc
      df_valid['err_dc'] = err_dc
    else:
      ##############print('No source found -- keeping PSF fit magnitude')
      df_valid['mag_dc'] = df_valid['magpsf']
      df_valid['err_dc'] = df_valid['sigmapsf']



    ref_r = np.sqrt((df_valid[df_valid['fid'] == 2]['magnr'] ** 2).mean())# Quadratic Mean
    ref_g = np.sqrt((df_valid[df_valid['fid'] == 1]['magnr'] ** 2).mean())


    # # 5) Calculate Apparent DC flux  

    # We utilize a function(`apparent_flux`) located within the `fink_utils` package to compute the apparent flux for the valid data.




    dc_flux, dc_sigflux = np.transpose(
        [
            apparent_flux(*args, jansky=True) for args in zip(
                df_valid['magpsf'].astype(float).values,
                df_valid['sigmapsf'].astype(float).values,
                df_valid['magnr'].astype(float).values,
                df_valid['sigmagnr'].astype(float).values,
                df_valid['isdiffpos'].values
            )
        ]
)

    df_valid['dc_flux'] = dc_flux
    df_valid['dc_sigflux'] = dc_sigflux


    # ## Apparent flux for the nearest source

    # We create a function `apparent_flux` to determine the apparent flux for the nearest source in the reference image.



    nr_flux, nr_sigflux = np.transpose(
        [
            flux_nr(*args, jansky=True) for args in zip(
                df_valid['magnr'].astype(float).values,
                df_valid['sigmagnr'].astype(float).values
            )
        ]
)

    df_valid['nr_flux'] = nr_flux
    df_valid['nr_sigflux'] = nr_sigflux


# # 6) Data missing 
# 

# Our objective here is to retrieve the missing data values, particularly for cases where they represent upper limits.

# Take only Upper limits data
    maskUpper = pd.isna(df['magpsf'])

    df_Upper = df[maskUpper].sort_values('jd')#, ascending=False)


# Compute the average of the sigma magnitude values for the nearest sources.


    sigmnr_r = np.sqrt((df_valid[df_valid['fid'] == 2]['sigmagnr'] ** 2).mean())
    sigmnr_g = np.sqrt((df_valid[df_valid['fid'] == 1]['sigmagnr'] ** 2).mean())


# We define a function named `apparent_flux_Upper` to calculate the apparent flux, along with its associated sigma (error), for both the DC flux and NR flux, specifically for data representing upper limits.
    

    columns_to_keep = ['jd', 'fid','dc_flux', 'dc_sigflux', 'nr_flux', 'nr_sigflux']

    if len(df_Upper) == 0 :
        print("there is no Upperlimits")
    
        combined_df = df_valid[columns_to_keep]
    
    else: 
        

        dc_flux, dc_sigflux,nr_sigflux = np.transpose(
        [
            apparent_flux_Upper(*args, ref_r, ref_g, sigmnr_r, sigmnr_g, jansky=True) for args in zip(
                df_Upper['diffmaglim'].astype(float).values,
                df_Upper['fid'].astype(int).values,

            )
        ]
)

        df_Upper['dc_flux'] = dc_flux
        df_Upper['dc_sigflux'] = dc_sigflux
        df_Upper['nr_sigflux'] = nr_sigflux
        df_Upper['nr_flux'] = dc_flux
        
        combined_df = pd.concat([df_Upper[columns_to_keep], df_valid[columns_to_keep]], axis=0)




# # 

# # 

# # 7) Combine Upper with valid 

# We merge the data from the upper limit and valid datasets based on specific columns.

    combined_df.sort_index(inplace=True)



# # 

# # 

# # 8) Data by days 

# Here, we group the data by modified Julian date on a daily basis and by filter ID (1 for g, 2 for R, 3 for i), computing the average values of flux and sigma flux(for both DC and NR) using the `Weighted_Mean` functions.
    

    combined_df['mjd'] = combined_df['jd'].apply(lambda x: x - 2400000.5)


# #### Convert 'mjd' to integer to remove fractional part
#

    combined_df['mjd'] = combined_df['mjd'].astype(int)


# #### group the data by mjd and by filter


    df2 = combined_df.groupby(['mjd','fid'])


# #### calculate the average values



    df_mod = pd.DataFrame()
    df_mod = df2.apply(Weighted_Mean_general, flux_col='dc_flux', sigflux_col='dc_sigflux')



    df_mod[['nr_flux', 'nr_sigflux']] = df2.apply(Weighted_Mean_general, flux_col='nr_flux',sigflux_col='nr_sigflux')



    df_mod.reset_index(inplace=True)



    df_mod.head(2)


# # 9) Fill the missing days ! 

# If there are missing days without alerts in the data, we can fill these gaps by inserting average values.


    min_mjd = df_mod['mjd'].min()
    max_mjd = df_mod['mjd'].max()
# Create a DataFrame all_days containing a range of MJD values from the minimum to the maximum MJD found in df_mod.
    all_days = pd.DataFrame({'mjd': range(min_mjd, max_mjd + 1)})

    df_extended = df_mod
    df_extended['source'] = 'Original'


#If this condition is true, it indicates that there is missing data.
    if (df_mod.shape[0] < (max_mjd -min_mjd + 1)*2):        
     for filt in np.unique(df_extended['fid']):

        mask = df_extended['fid'] == filt
        sub = df_extended[mask]
        data_days = df_extended[mask]['mjd']

        missing_days = all_days[~all_days['mjd'].isin(data_days)]
    
    
        df_new = pd.DataFrame(index=missing_days['mjd'])
    
        dc_flux ,nr_flux = Weighted_Mean_all(sub)
        
        dc_sigflux = sub['dc_flux'].std()
        nr_sigflux = sub['nr_flux'].std()
        

        
        df_new[['fid','dc_flux', 'dc_sigflux' ,'nr_flux' ,'nr_sigflux']] = [filt,dc_flux, dc_sigflux ,nr_flux ,nr_sigflux]
        df_new['source'] = 'Missing'

        df_extended = pd.concat([df_extended, df_new.reset_index()], ignore_index=True)
        


    df_extended.sort_values(by='mjd',   inplace=True)
    df_extended.reset_index(drop= True, inplace=True)
    
    return pdf_selectionne, df_extended


# # 10) Create a final dataframe to consolidate the values of this alert into a single row.

# In this dataframe, include another dataframe as a dictionary containing the values of mjd,flux, sigma, and so on.



df_anomaly = pd.DataFrame()
# can be optimized by removing the function !

for Id in unique_ids:
    
    Anomaly, df_anm = function_FN(Id)
    #df_anomaly[['objectId', 'candid', 'jd','df']] = [[Anomaly.objectId], [Anomaly.candid],[Anomaly.candidate['jd']], [df_anm.to_dict()]]

    df_anomaly['objectId'] = [Anomaly.objectId]
    df_anomaly['candid'] = [Anomaly.candid]
    df_anomaly['jd'] = [Anomaly.candidate['jd']]
    df_anomaly['df'] = [df_anm.to_dict()]
    

    """fig = plt.figure(figsize=(15, 10))

    colordic = {1: 'C0', 2: 'C1'}
    filtdic = {1: 'g', 2: 'r'}


    for filt in np.unique(df_anm['fid']):
        mask = df_anm['fid'] == filt
        mask_missing =  df_anm['source'] == "Missing"
        mask_original = df_anm['source'] == "Original"
        sub2 = df_anm[mask & mask_missing]
        sub = df_anm[mask & mask_original]
        plt.errorbar(
            sub['mjd'],
            sub['dc_flux']*1e3, 
            sub['dc_sigflux']*1e3,
            ls='', 
            marker='o',
            color=colordic[filt], 

            label=f"{filt} original flux dc"
        )
        plt.errorbar(
            sub2['mjd'],
            sub2['dc_flux']*1e3, 
            sub2['dc_sigflux']*1e3,
            ls='', 
            marker='x',
            color=colordic[filt], 

            label=f"{filt} Missing flux dc"
        )

        
    plt.legend()
    plt.title(f'{Anomaly.objectId}')
    plt.xlabel('Modified Julian Date [UTC]  ')
    plt.ylabel('Apparent  DC and nr flux (millijanksy)')
    print(Anomaly.objectId)"""
    
#print(df_anomaly['df'])


