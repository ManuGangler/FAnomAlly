#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import Tuple

def apparent_flux(
    magpsf: float,
    sigmapsf: float,
    magnr: float,
    sigmagnr: float,
    isdiffpos: int,
    jansky: bool = True
) -> Tuple[float, float]:
    """Compute apparent flux from difference magnitude supplied by ZTF
    Implemented according to p.107 of the ZTF Science Data System Explanatory Supplement
    https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf

    Parameters
    ---------
    magpsf,sigmapsf; floats
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr: floats
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    isdiffpos: str
        t or 1 => candidate is from positive (sci minus ref) subtraction;
        f or 0 => candidate is from negative (ref minus sci) subtraction
    jansky: bool
        If True, normalise units to Jansky. Default is True.

    Returns
    --------
    dc_flux: float
        Apparent flux
    dc_sigflux: float
        Error on apparent flux
    """
    if magpsf is None or magnr < 0:
        return float("Nan"), float("Nan")

    difference_flux = 10 ** (-0.4 * magpsf)
    difference_sigflux = (sigmapsf / 1.0857) * difference_flux

    ref_flux = 10 ** (-0.4 * magnr)
    ref_sigflux = (sigmagnr / 1.0857) * ref_flux

    # add or subract difference flux based on isdiffpos
    if (isdiffpos == 't') or (isdiffpos == '1'):
        dc_flux = ref_flux + difference_flux
    else:
        dc_flux = ref_flux - difference_flux

    # assumes errors are independent. Maybe too conservative.
    dc_sigflux = np.sqrt(difference_sigflux**2 + ref_sigflux**2)

    if jansky:
        dc_flux *= 3631
        dc_sigflux *= 3631

    return dc_flux, dc_sigflux



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
    sigmnr_mean_r: float,
    sigmnr_mean_g: float,
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
        sigmnr = sigmnr_mean_g
        
        sig_c = (10 ** (-0.4 * diffmaglim))/np.sqrt(3) * 1.9973023835917523 # (rescale factor ! )

    if (fid == 2):
        mnr = ref_r
        sigmnr = sigmnr_mean_r
        
        sig_c = (10 ** (-0.4 * diffmaglim))/np.sqrt(3) * 4.654601721082196 # (rescale factor ! )


    dc_flux = 10 ** (-0.4 * mnr)
    sig_flux_nr = np.log(10) * 0.4 * (10 ** (-0.4 * mnr)) *sigmnr # dc_flux ! 
    
    dc_sigflux = np.sqrt(sig_c**2 + sig_flux_nr**2)

    if jansky:
        dc_flux *= 3631
        dc_sigflux *= 3631
        sig_flux_nr *= 3631

    return dc_flux, dc_sigflux, sig_flux_nr
