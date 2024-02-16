#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from typing import Tuple

def flux_nr(
    magnr: float,
    sigmagnr: float,
    jansky: bool = True

) -> Tuple[float, float]:
    """
    
    Parameters
    ---------

    Returns
    --------
    dc_flux: float
        Apparent flux
    dc_sigflux: float
        Error on apparent flux
    """

    nr_flux = 10 ** (-0.4 * magnr)
    
    nr_sigflux = np.log(10) * 0.4 * nr_flux *sigmagnr # dc_flux ! 
    
    if jansky:
        nr_flux *= 3631
        nr_sigflux *= 3631

    return nr_flux, nr_sigflux



def apparent_flux_Upper(
    diffmaglim: float,
    fid: int,
    ref_r: float,
    ref_g: float,
    sigmnr_r: float,
    sigmnr_g: float,
    jansky: bool = True

) -> Tuple[float, float]:
    """
    
    Parameters
    ---------

    Returns
    --------
    dc_flux: float
        Apparent flux
    dc_sigflux: float
        Error on apparent flux
    """
    

    if (fid == 1):
        mnr = ref_g
        sigmnr = sigmnr_g
        
        sig_c = (10 ** (-0.4 * diffmaglim))/np.sqrt(3) * 1.9973023835917523 # (rescale factor ! )

    if (fid == 2):
        mnr = ref_r
        sigmnr = sigmnr_r
        
        sig_c = (10 ** (-0.4 * diffmaglim))/np.sqrt(3) * 4.654601721082196 # (rescale factor ! )


    dc_flux = 10 ** (-0.4 * mnr)
    sig_flux_nr = np.log(10) * 0.4 * (10 ** (-0.4 * mnr)) *sigmnr # dc_flux ! 
    
    dc_sigflux = np.sqrt(sig_c**2 + sig_flux_nr**2)

    if jansky:
        dc_flux *= 3631
        dc_sigflux *= 3631
        sig_flux_nr *= 3631

    return dc_flux, dc_sigflux, sig_flux_nr
