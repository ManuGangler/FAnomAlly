#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from fink_utils.photometry.conversion import apparent_flux
from fink_utils.photometry.conversion import dc_mag
from fink_utils.photometry.utils import is_source_behind
from FAnomAly.flux import flux_nr
from FAnomAly.flux import apparent_flux_Upper
from FAnomAly.Weighted_Mean import Weighted_Mean_general
from FAnomAly.Weighted_Mean import Weighted_Mean_all
import time


def function_FN(Id):    
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

    isSource = is_source_behind(df_valid['distnr'].values[0])

    if isSource:
      #print('It looks like there is a source behind. Lets compute the DC magnitude instead.')
    
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
      #print('No source found -- keeping PSF fit magnitude')
      df_valid['mag_dc'] = df_valid['magpsf']
      df_valid['err_dc'] = df_valid['sigmapsf']



    ref_value_r = np.sqrt((df_valid[df_valid['fid'] == 2]['magnr'] ** 2).mean())# Quadratic Mean
    ref_value_g = np.sqrt((df_valid[df_valid['fid'] == 1]['magnr'] ** 2).mean())


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

    df_Upper = df[maskUpper].sort_values('jd')


    columns_to_keep = ['jd', 'fid','dc_flux', 'dc_sigflux', 'nr_flux', 'nr_sigflux', 'source']

    df_valid['source'] = 1
    if len(df_valid[df_valid['fid'] == 1]) ==0 :
         df_Upper.drop(df_Upper[df_Upper['fid'] == 1].index, inplace=True)
    
         new_rows = pd.DataFrame({'fid': [1, 1],
                             'jd': [df_valid['jd'].min(), df_valid['jd'].max()],
                             'dc_flux': [0, 0],
                             'dc_sigflux': [0, 0],
                             'nr_flux' : [0,0],
                             'nr_sigflux':[0,0],
                             'source' : [0,0]
                            })

         df_valid = pd.concat([df_valid, new_rows], ignore_index=True)

    elif len(df_valid[df_valid['fid'] == 2]) ==0 : 
          df_Upper.drop(df_Upper[df_Upper['fid'] == 2].index, inplace=True)
    
        
          new_rows = pd.DataFrame({'fid': [2, 2],
                             'jd': [df_valid['jd'].min(), df_valid['jd'].max()],
                             'dc_flux': [0, 0],
                             'dc_sigflux': [0, 0],
                             'nr_flux' : [0,0],
                             'nr_sigflux':[0,0],
                             'source' : [0,0]

                           })

          df_valid = pd.concat([df_valid, new_rows], ignore_index=True)

    there_upper = (len(df_Upper)>0)

    if there_upper :
        df_Upper['source'] = 1

  
        mean_sigmnr_g = np.sqrt((df_valid[df_valid['fid'] == 1]['sigmagnr'] ** 2).mean())
        mean_sigmnr_r = np.sqrt((df_valid[df_valid['fid'] == 2]['sigmagnr'] ** 2).mean())
    



        dc_flux, dc_sigflux,nr_sigflux = np.transpose(
        [
            apparent_flux_Upper(*args, ref_value_r, ref_value_g, mean_sigmnr_r,  mean_sigmnr_g, jansky=True) for args in zip(
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
        #print("there is no Upperlimits")

        combined_df = df_valid[columns_to_keep].copy()


    combined_df.sort_index(inplace=True)
    combined_df['mjd'] = (combined_df['jd'] - 2400000.5).astype(int)


    df_group = combined_df.groupby(['mjd','fid'])





    df_by_days = pd.DataFrame()
    df_by_days = df_group.apply(Weighted_Mean_general, flux_col='dc_flux', sigflux_col='dc_sigflux')



    df_by_days[['nr_flux', 'nr_sigflux']] = df_group.apply(Weighted_Mean_general, flux_col='nr_flux',sigflux_col='nr_sigflux')

    df_by_days['source'] = df_group.apply(lambda group: group['source'].iloc[0])

    df_by_days.reset_index(inplace=True)

    min_mjd = df_by_days['mjd'].min()
    max_mjd = df_by_days['mjd'].max()

    all_days = pd.DataFrame({'mjd': range(min_mjd, max_mjd + 1)})

    df_extended = df_by_days
    #df_extended['source'] = 1

    there_missing_data = (df_by_days.shape[0] < (max_mjd -min_mjd + 1)*2)
    if there_missing_data:        
     for filt in np.unique(df_extended['fid']):

        mask = df_extended['fid'] == filt
        sub = df_extended[mask]
        data_days = df_extended[mask]['mjd']

        missing_days = all_days[~all_days['mjd'].isin(data_days)]
    
    
        df_predic = pd.DataFrame(index=missing_days['mjd'])
    
        dc_flux ,nr_flux = Weighted_Mean_all(sub)
        
        dc_sigflux = sub['dc_flux'].std()
        nr_sigflux = sub['nr_flux'].std()
        

        
        df_predic[['fid','dc_flux', 'dc_sigflux' ,'nr_flux' ,'nr_sigflux']] = [filt,dc_flux, dc_sigflux ,nr_flux ,nr_sigflux]
        df_predic['source'] = 0

        df_extended = pd.concat([df_extended, df_predic.reset_index()], ignore_index=True)
        
    df_extended['objectId'] = Id 

    df_extended.sort_values(by=['mjd','fid'],   inplace=True)
    df_extended.reset_index(drop= True, inplace=True)
    
    return pdf_last_alert, df_extended




def main():    
    #global pdf
    #pdf = pd.read_parquet('../../ftransfer_ztf_2024-02-01_689626')

    unique_ids = pdf['objectId'].unique().tolist()

    start_time = time.time()

    df_anomaly = pd.DataFrame(columns=['objectId', 'candid', 'jd'])
    df_anomaly2 = pd.DataFrame(columns=['objectId', 'df'])

    results=[]
    results2=[]

    for Id in unique_ids[:100]:
       #print(Id)

       Anomaly, df_anm = function_FN(Id)
 
       results.append([Id, Anomaly.candid, Anomaly.candidate['jd']])
       results2.append(df_anm)

    df_anomaly = pd.DataFrame(results, columns=['objectId', 'candid', 'jd'])
    df_anomaly2 = pd.concat(results2, ignore_index=True)
    df_merged = pd.merge(df_anomaly, df_anomaly2, on='objectId', how='inner')

    df_merged.to_parquet('df_merged1.parquet', compression='gzip')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Temps écoulé:", elapsed_time, "secondes")

    print(df_merged)


if __name__ == "__main__":
    main()

