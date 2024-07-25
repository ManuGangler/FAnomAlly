#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:24:11 2024

@author: mohamadjouni
"""

import pandas as pd
import numpy as np
#from fink_utils.photometry.conversion import apparent_flux

from fink_utils.photometry.utils import is_source_behind
from FAnomAly.flux import flux_nr
from FAnomAly.flux import apparent_flux_New

from FAnomAly.Weighted_Mean import Weighted_Mean_general
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
kernel =  ConstantKernel(constant_value=1.e-6, constant_value_bounds=(1e-12, 1)) * RBF(length_scale=5.0, length_scale_bounds=(5e-1, 5e2))

import warnings
import matplotlib.pyplot as plt
colordic = {1: 'C0', 2: 'C1'}



def Gaussian_Process(df,flux= 'dc_flux',sigflux= 'dc_sigflux'):
    mean_prediction = {}
    std_prediction = {}
    #plt.figure()
    for filt in np.unique(df.fid):
        X_train=df[df.fid==filt].mjd.values.reshape(-1, 1)
        Y_train=np.squeeze(df[df.fid==filt][f'{flux}'].values)
        Sig_train = np.squeeze(df[df.fid==filt][f'{sigflux}'].values)
        X=np.arange(min_mjd,max_mjd+1,1).reshape(-1, 1)


        if len(X_train) == 1 : 
            # If there's only one training point, create a series of mean predictions and high error bars
            #print(Y_train)
            mean_prediction[filt] = np.full((max_mjd - min_mjd + 1,), float(Y_train))
            std_prediction[filt] = np.full((max_mjd - min_mjd + 1,), 1 * float(Sig_train))
            
        else :
            
            gaussian_process = GaussianProcessRegressor( kernel=kernel, alpha=Sig_train**2, n_restarts_optimizer=9 )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                gaussian_process.fit(X_train, Y_train)
            mean_prediction[filt], std_prediction[filt] = gaussian_process.predict(X, return_std=True)
        
        """plt.errorbar(
            X_train, Y_train, Sig_train,
            ls='', 
            marker='o',
            color=colordic[filt], 
            label=f"{filt} valid difference flux"
        )
    
        plt.plot(
            X,mean_prediction[filt], c=colordic[filt], 
            label=f"{filt} GP prediction"
        )
    
        plt.fill_between(
            X[:,0],mean_prediction[filt]+std_prediction[filt], mean_prediction[filt]-std_prediction[filt],color=colordic[filt], alpha=0.3
        )
        plt.show()"""
    return mean_prediction, std_prediction




def function_FN(Id, first_day, last_day):    
    pdf_filter_by_shared_Id = pdf.loc[pdf['objectId'] == Id]

    candidate_df = pdf_filter_by_shared_Id['candidate'].apply(pd.Series)

    index_max_jd = candidate_df.sort_values(by= 'jd').index[-1]

    pdf_last_alert = pdf_filter_by_shared_Id.loc[index_max_jd]

    pdf_selectionne_cand = pdf_last_alert['prv_candidates'] 

    keys = pdf_selectionne_cand[0].keys()
    latest_cand = {key: pdf_last_alert['candidate'][key] for key in keys if key in pdf_last_alert['candidate']}

    liste_dicts = list(pdf_selectionne_cand)
    liste_dicts.append(latest_cand)
    df = pd.DataFrame(liste_dicts)

    maskValid = (df['rb'] > 0.55) & (df['nbad'] == 0)
    df_valid = df[maskValid].sort_values('jd')
    
    
    df_valid['is_Source'] = is_source_behind(df_valid['distnr'])


    dc_flux, dc_sigflux = np.transpose(
            [
                apparent_flux_New(*args, jansky=True) for args in zip(
                    df_valid['magpsf'].astype(float).values,
                    df_valid['sigmapsf'].astype(float).values,
                    df_valid['magnr'].astype(float).values,
                    df_valid['sigmagnr'].astype(float).values,
                    df_valid['isdiffpos'].values,
                    df_valid['is_Source'].astype(bool).values
    
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

      
    
    df_valid['mjd'] = (df_valid['jd'] - 2400000.5).astype(int)



    is_Source_by_fid = df_valid.groupby(['fid', 'is_Source']).size().unstack(fill_value=0).sort_index()

    # # Determine which count is highest
    is_Source_by_fid['Highest_bool'] = is_Source_by_fid.idxmax(axis=1)
    min_mjd = int(first_day - 2400000.5) #df_by_days['mjd'].min()
    max_mjd = int(last_day  - 2400000.5)
    

    df_valid['source'] = 1
 
    if len(df_valid[df_valid['fid'] == 1]) ==0 : 

    
         new_rows = pd.DataFrame({'fid': [1, 1],
                             'mjd': [min_mjd, max_mjd],
                             'dc_flux': [0, 0],
                             'dc_sigflux': [0, 0],
                             'nr_flux' : [0,0],
                             'nr_sigflux':[0,0],
                             'source' : [0,0],
                             'is_Source': [is_Source_by_fid['Highest_bool'].iloc[0],is_Source_by_fid['Highest_bool'].iloc[0]]

                            })

         df_valid = pd.concat([df_valid, new_rows], ignore_index=True)

    elif (len(df_valid[df_valid['fid'] == 2]) ==0) : 

          new_rows = pd.DataFrame({'fid': [2, 2],
                             'mjd': [min_mjd, max_mjd],
                             'dc_flux': [0, 0],
                             'dc_sigflux': [0, 0],
                             'nr_flux' : [0,0],
                             'nr_sigflux':[0,0],
                             'source' : [0,0],
                             'is_Source': [is_Source_by_fid['Highest_bool'].iloc[0],is_Source_by_fid['Highest_bool'].iloc[0]]

                           })

          df_valid = pd.concat([df_valid, new_rows], ignore_index=True)
          
    
    
                        
    
    
    df_valid['is_valid'] = True
   
    columns_to_keep = ['mjd', 'fid','dc_flux', 'dc_sigflux','nr_flux', 'nr_sigflux','source']

    combined_df = df_valid[columns_to_keep].copy()


    combined_df.sort_index(inplace=True)


    df_group = combined_df.groupby(['mjd','fid'])



    df_by_days_dc = df_group.apply(Weighted_Mean_general, flux_col='dc_flux', sigflux_col='dc_sigflux')#.reset_index()
    
    
    df_by_days_nr = df_group.apply(Weighted_Mean_general, flux_col='nr_flux', sigflux_col='nr_sigflux')#.reset_index()
    
    df_by_days_source = df_group['source'].apply(lambda x: x.value_counts().idxmax())
    
    # Merge the results based on 'mjd' and 'fid'
    df_by_days = pd.merge(df_by_days_dc, df_by_days_nr, on=['mjd', 'fid'], suffixes=('_dc', '_nr'))

    df_by_days = pd.merge(df_by_days, df_by_days_source, on=['mjd', 'fid'])
    df_by_days.reset_index(inplace=True)

    
    dc_flux = {}
    dc_sigflux_pred = {}
    
    
    dc_flux_pred,dc_sigflux_pred = Gaussian_Process(df_by_days, "dc_flux","dc_sigflux")

    all_days = pd.DataFrame({'mjd': range(min_mjd, max_mjd + 1)})

    df_extended = df_by_days
    #df_extended['source'] = 1
    ########################################### We don't really need all these lines, we can simplify them
    there_missing_data = (df_by_days.shape[0] < (max_mjd -min_mjd + 1)*2)
    if there_missing_data:
     for filt in np.unique(df_extended['fid']):
        mask = df_extended['fid'] == filt
        data_days = df_extended[mask]['mjd']

        
        missing_days = ~all_days['mjd'].isin(data_days)
        df_missing_days = all_days[missing_days].copy()  # Ensure a copy to avoid chain indexing issues
        
        true_indices = missing_days[missing_days].index.tolist()
        
  
        df_missing_days['dc_flux'] = dc_flux_pred[filt][true_indices]
        df_missing_days['dc_sigflux'] = dc_sigflux_pred[filt][true_indices]
        
        df_missing_days['fid'] = filt
        df_missing_days['source'] = 0
    
        # Append the missing data to df_extended
        df_extended = pd.concat([df_extended, df_missing_days], ignore_index=True, sort=False)
    
    columns_to_keep = ['mjd', 'fid','dc_flux', 'dc_sigflux','source']

    df_extended = df_extended[columns_to_keep].sort_values(by=['mjd', 'fid'])
    
    df_extended['objectId'] = Id 


    #df_extended.sort_values(by=['mjd','fid'],   inplace=True)
    df_extended.reset_index(drop= True, inplace=True)
        
    return pdf_last_alert, df_extended




def main():    
    global X, min_mjd, max_mjd, pdf
    pdf = pd.read_parquet('../../ftransfer_ztf_2024-02-01_689626')
            
    jd_series = pdf.candidate.apply(lambda a:a['jd'])
    last_day = jd_series.max()
    first_day = np.min(pdf.prv_candidates.apply(lambda a: a[0]['jd']))
    
    
    min_mjd = int(first_day - 2400000.5) #df_by_days['mjd'].min()
    max_mjd = int(last_day  - 2400000.5)
    
    X=np.arange(min_mjd,max_mjd+1,1).reshape(-1, 1)


    unique_ids = pdf['objectId'].unique().tolist()
    
    with open("Classifications_arch/unique_ids.txt", 'w') as file: # you only need to do that once
        # Write each unique ID to a separate line
        for id in unique_ids:
          file.write(f"{id}\n")
          
    start_time = time.time()

    df_anomaly = pd.DataFrame(columns=['objectId', 'candid', 'jd'])
    df_anomaly2 = pd.DataFrame(columns=['objectId', 'df'])

    results=[]
    results2=[]
    

    for Id in unique_ids[:50]:
       #print(Id)

       Anomaly, df_anm = function_FN(Id,first_day,last_day)
 
       results.append([Id, Anomaly.candid, Anomaly.candidate['jd']])
       results2.append(df_anm)

    df_anomaly = pd.DataFrame(results, columns=['objectId', 'candid', 'jd'])
    df_anomaly2 = pd.concat(results2, ignore_index=True)
    df_merged = pd.merge(df_anomaly, df_anomaly2, on='objectId', how='inner')


    df_merged.to_parquet('df_test.parquet', compression='gzip')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Temps écoulé:", elapsed_time, "secondes")




if __name__ == "__main__":
    main()


