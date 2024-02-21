#!/usr/bin/env python
# -*- coding: utf-8 -*-


# In[1]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fink_utils.photometry.conversion import apparent_flux
from FAnomAly.flux import flux_nr
from FAnomAly.flux import apparent_flux_Upper
from FAnomAly.Weighted_Mean import Weighted_Mean_general
from FAnomAly.Weighted_Mean import Weighted_Mean_all
import time



from fink_utils.photometry.conversion import dc_mag
from fink_utils.photometry.utils import is_source_behind



pdf = pd.read_parquet('work/ftransfer_ztf_2024-02-01_689626')
# In[2]

unique_ids = pdf['objectId'].unique().tolist()



def function_FN(Id):    
    pdf_selectionne = pdf.loc[pdf['objectId'] == Id]

    candidate_df = pdf_selectionne['candidate'].apply(pd.Series)


    candidate_df = candidate_df.sort_values(by= 'jd')
    # index of the candidate with the biggest 'jd'
    index_max_jd = candidate_df.index[-1]

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



    maskUpper = pd.isna(df['magpsf'])

    df_Upper = df[maskUpper].sort_values('jd')#, ascending=False)


    columns_to_keep = ['jd', 'fid','dc_flux', 'dc_sigflux', 'nr_flux', 'nr_sigflux']
    there_upper = (len(df_Upper)>0)
    if there_upper :

        sigmnr_r = np.sqrt((df_valid[df_valid['fid'] == 2]['sigmagnr'] ** 2).mean())
        sigmnr_g = np.sqrt((df_valid[df_valid['fid'] == 1]['sigmagnr'] ** 2).mean())



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

        
    
    else:
        print("there is no Upperlimits")

        combined_df = df_valid[columns_to_keep].copy()

# # 7) Combine Upper with valid 

    combined_df.sort_index(inplace=True)


# # 8) Data by days 

    combined_df['mjd'] = (combined_df['jd'] - 2400000.5).astype(int)

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
    df_extended['source'] = 1


#If this condition is true, it indicates that there is missing data.
    if (df_mod.shape[0] < (max_mjd -min_mjd )*2):        
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
        df_new['source'] = 0

        df_extended = pd.concat([df_extended, df_new.reset_index()], ignore_index=True)
        
    df_extended['objectId'] = Id 

    df_extended.sort_values(by='mjd',   inplace=True)
    df_extended.reset_index(drop= True, inplace=True)
    
    return pdf_selectionne, df_extended


start_time = time.time()



df_anomaly = pd.DataFrame(columns=['objectId', 'candid', 'jd'])
df_anomaly2 = pd.DataFrame(columns=['objectId', 'df'])
# can be optimized by removing the function !
results=[]
results2=[]
for Id in unique_ids[:100]:
    Anomaly, df_anm = function_FN(Id)
    #print(Id)
    #df_anomaly2.loc[len(df_anomaly2)] = [Id, Anomaly.candid, Anomaly.candidate['jd'], df_anm.to_dict()]

    # Append the results to the list
    results.append([Id, Anomaly.candid, Anomaly.candidate['jd']])
    results2.append(df_anm)

# Create a DataFrame from the list of results
df_anomaly = pd.DataFrame(results, columns=['objectId', 'candid', 'jd'])
df_anomaly2 = pd.concat(results2, ignore_index=True)
df_merged = pd.merge(df_anomaly, df_anomaly2, on='objectId', how='inner')

# Write DataFrame to HDF5 with compression
df_anomaly.to_hdf('data1.h5', key='df', mode='w', complib='zlib', complevel=9)
df_anomaly2.to_hdf('data2.h5', key='df', mode='w', complib='zlib', complevel=9)
df_merged.to_hdf('data3.h5', key='df', mode='w', complib='zlib', complevel=9)


df_anomaly.to_parquet('df_anomaly.parquet')
df_anomaly2.to_parquet('df_anomaly2.parquet')
df_merged.to_parquet('df_merged.parquet')

#df_hdf = pd.read_hdf('data2.h5', key='df')



end_time = time.time()
elapsed_time = end_time - start_time
print("Temps écoulé:", elapsed_time, "secondes")


# In[3]

import time

compression_options = ['gzip', 'snappy', 'brotli', 'lz4']  

for compression in compression_options:
    start_time = time.time()
    
    df_anomaly.to_parquet(f'df_anomaly_{compression}.parquet', compression=compression)
    df_anomaly2.to_parquet(f'df_anomaly2_{compression}.parquet', compression=compression)
    df_merged.to_parquet(f'df_merged_{compression}.parquet', compression=compression)
    
    end_time = time.time()
    
    write_time = end_time - start_time
    
    print(f"Files saved with {compression} compression in {write_time} seconds")
