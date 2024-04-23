#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from fink_utils.photometry.conversion import apparent_flux
#from fink_utils.photometry.conversion import dc_mag
from fink_utils.photometry.utils import is_source_behind
from FAnomAly.flux import flux_nr
from FAnomAly.flux import apparent_flux_Upper
from FAnomAly.flux import apparent_flux_New
from FAnomAly.dc_mag import dc_mag
from FAnomAly.Weighted_Mean import Weighted_Mean_general
from FAnomAly.Weighted_Mean import Weighted_Mean_all
import time

    
def get_interval(target_mjd,fid, min_mjd, max_mjd,df_valid):
    lower_bound = min_mjd
    upper_bound = max_mjd
#     print(lower_bound,upper_bound )
    for index, row in df_valid[df_valid['fid'] == fid].iterrows():
        if row['mjd'] < target_mjd:
            lower_bound = row['mjd']
        else:
            upper_bound = row['mjd']
            break
    return lower_bound,upper_bound 

def get_pos_neg(x,magpsf, magnr): # change the name ! 
    #A = 2 ## here for pos flux, =1/2 for neg 
    if (x == 't') or (x == '1'):
        return magpsf < (magnr+0)
    # maskneg = (df['isdiffpos'] == 'f') | (df['isdiffpos'] == '0')
    return magpsf > (magnr+2.5*np.log10(2))

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

      
    


    maskUpper = pd.isna(df['magpsf'])

    df_Upper = df[maskUpper].sort_values('jd')
    
    
    
    df_valid['mjd'] = (df_valid['jd'] - 2400000.5).astype(int)
    df_Upper['mjd'] = (df_Upper['jd'] - 2400000.5).astype(int)



    columns_to_keep = ['mjd', 'fid','dc_flux', 'dc_sigflux', 'nr_flux', 'nr_sigflux', 'source','is_valid']

    is_Source_by_fid = df_valid.groupby(['fid', 'is_Source']).size().unstack(fill_value=0).sort_index()

    # # Determine which count is highest
    is_Source_by_fid['Highest_bool'] = is_Source_by_fid.idxmax(axis=1)
    min_mjd = int(first_day - 2400000.5) #df_by_days['mjd'].min()
    max_mjd = int(last_day  - 2400000.5)
    

    df_valid['source'] = 1
 
    if len(df_valid[df_valid['fid'] == 1]) ==0 :
         df_Upper.drop(df_Upper[df_Upper['fid'] == 1].index, inplace=True)
    
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

    elif len(df_valid[df_valid['fid'] == 2]) ==0 : 
          df_Upper.drop(df_Upper[df_Upper['fid'] == 2].index, inplace=True)
    
        
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
          
    
            
    for filt in np.unique(df_valid['fid']):
        df_fid = df_valid[df_valid['fid'] == filt]
        min_mjd2 = df_fid['mjd'].min()
        max_mjd2 = df_fid['mjd'].max()
    
        
        for index, row in df_Upper[df_Upper['fid'] == filt][['mjd']].iterrows():
            # If valid data and an upper limit are both present in a given day, 
            # we only consider the valid data, (we drope upperlim)
            if len(df_fid[df_fid['mjd'] == row['mjd']]) > 0 :
                df_Upper.drop(index, inplace=True)

            
            elif row['mjd'] < min_mjd2 : #or row['mjd'] > max_mjd: 
                 df_min = df_fid[(df_fid['mjd'] == min_mjd2)]
                 idx = df_min['magpsf'].idxmax()
                 x = df_min['isdiffpos'].loc[idx]
                 magpsf_down, magnr_down = df_min[['magpsf','magnr']].loc[idx]
                 if get_pos_neg(x,magpsf_down, magnr_down): 
                     df_Upper.drop(index, inplace=True)
    
            elif row['mjd'] > max_mjd2 : #or row['mjd'] > max_mjd: 
                 df_max = df_fid[(df_fid['mjd'] == max_mjd2)]
                 idx = df_max['magpsf'].idxmax()
                 x = df_max['isdiffpos'].loc[idx]
                 magpsf_up, magnr_up = df_max[['magpsf','magnr']].loc[idx]
                 if get_pos_neg(x,magpsf_up, magnr_up): 
                     df_Upper.drop(index, inplace=True)
    
    
            else:
                lower_bound,upper_bound = get_interval(row['mjd'],filt, min_mjd2, max_mjd2, df_valid)
                # print(filt, row['mjd'], lower_bound, upper_bound)
                      
                df_down = df_fid[(df_fid['mjd'] == lower_bound)]
                idx_down = df_down['magpsf'].idxmax()
                x_down = df_down['isdiffpos'].loc[idx_down]
                magnr_down, magpsf_down = df_down[['magnr','magpsf']].loc[idx_down]
    
    
                df_up = df_fid[(df_fid['mjd'] == upper_bound)]
                idx_up = df_up['magpsf'].idxmax()
                x_up = df_up['isdiffpos'].loc[idx_up]
                magnr_up, magpsf_up = df_up[['magnr','magpsf']].loc[idx_up]
                
                
                #print(filt, row['mjd'], lower_bound, upper_bound, magpsf_down, magpsf_up, magnr_down, magnr_up)
                if get_pos_neg(x_down, magpsf_down, magnr_down) and get_pos_neg(x_up, magpsf_up, magnr_up):
                    #print("A>1")
                    df_Upper.drop(index, inplace=True)
    
    
                        
    
    there_upper = (len(df_Upper)>0)
    
    df_valid['is_valid'] = True
   
    if there_upper :
        df_Upper['is_valid'] = False

        is_Source_by_fid = df_valid.groupby(['fid', 'is_Source']).size().unstack(fill_value=0).sort_index()
        
        # Determine which count is highest
        is_Source_by_fid['Highest_bool'] = is_Source_by_fid.idxmax(axis=1)
        
        
        df_Upper['source'] = 1
        
        
        if is_Source_by_fid['Highest_bool'].iloc[0]: 
            ref_value_g = np.sqrt((df_valid[df_valid['fid'] == 1]['magnr'] ** 2).mean())
            mean_sigmnr_g = np.sqrt((df_valid[df_valid['fid'] == 1]['sigmagnr'] ** 2).mean())
        else:
            ref_value_g = np.inf
            mean_sigmnr_g = 0
    
            
        if is_Source_by_fid['Highest_bool'].iloc[1]: 
            ref_value_r = np.sqrt((df_valid[df_valid['fid'] == 2]['magnr'] ** 2).mean())# Quadratic Mean
            mean_sigmnr_r = np.sqrt((df_valid[df_valid['fid'] == 2]['sigmagnr'] ** 2).mean())
        else:
            ref_value_r = np.inf
            mean_sigmnr_r = 0
                
        
  


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


    df_group = combined_df.groupby(['mjd','fid'])





    df_by_days = pd.DataFrame()
    df_by_days = df_group.apply(Weighted_Mean_general, flux_col='dc_flux', sigflux_col='dc_sigflux')



    df_by_days[['nr_flux', 'nr_sigflux']] = df_group.apply(Weighted_Mean_general, flux_col='nr_flux',sigflux_col='nr_sigflux')

    df_by_days['source'] = df_group.apply(lambda group: group['source'].iloc[0])
    df_by_days['is_valid'] = df_group.apply(lambda group: group['is_valid'].iloc[0])
    df_by_days.reset_index(inplace=True)

    
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
    
    jd_series = pdf.candidate.apply(pd.Series)['jd']

    last_day = jd_series.max()
    # this is not the perfect way to verify the min but we suppose that 'jd' is sorted
    first_day = last_day
    for k in range(len(pdf)):
        if first_day > pdf.prv_candidates[k][0]['jd']:
            first_day = pdf.prv_candidates[k][0]['jd']
        
    

    unique_ids = pdf['objectId'].unique().tolist()

    start_time = time.time()

    df_anomaly = pd.DataFrame(columns=['objectId', 'candid', 'jd'])
    df_anomaly2 = pd.DataFrame(columns=['objectId', 'df'])

    results=[]
    results2=[]
    

    for Id in unique_ids[:100]:
       #print(Id)

       Anomaly, df_anm = function_FN(Id,first_day,last_day)
 
       results.append([Id, Anomaly.candid, Anomaly.candidate['jd']])
       results2.append(df_anm)

    df_anomaly = pd.DataFrame(results, columns=['objectId', 'candid', 'jd'])
    df_anomaly2 = pd.concat(results2, ignore_index=True)
    df_merged = pd.merge(df_anomaly, df_anomaly2, on='objectId', how='inner')


    df_merged.to_parquet('df_after_upper_conds.parquet', compression='gzip')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Temps écoulé:", elapsed_time, "secondes")

    #print(df_merged)

import pstats

import cProfile
if __name__ == "__main__":
    main()
    #cProfile.run('main()', sort='cumulative')
    #cProfile.run('main()', 'profile_stats')
    #stats = pstats.Stats('profile_stats')
    #stats.strip_dirs().sort_stats(-1).print_stats()



