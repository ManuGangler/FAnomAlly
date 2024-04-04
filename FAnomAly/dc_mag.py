#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:36:39 2024

@author: mohamadjouni
"""
import numpy as np
from typing import Tuple
from flux import apparent_flux

def dc_mag(
    magpsf: float,
    sigmapsf: float,
    magnr: float,
    sigmagnr: float,
    isdiffpos: int,
    is_Source: bool = True
) -> Tuple[float, float]:
    """Compute apparent magnitude from difference magnitude supplied by ZTF
    Implemented according to p.107 of the ZTF Science Data System Explanatory Supplement
    https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf

    Parameters
    ----------
    magpsf,sigmapsf
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    isdiffpos
        t or 1 => candidate is from positive (sci minus ref) subtraction
        f or 0 => candidate is from negative (ref minus sci) subtraction

    Returns
    --------
    dc_mag: float
        Apparent magnitude
    dc_sigmag: float
        Error on apparent magnitude
    """
    
    
    if is_Source: 
    
        dc_flux, dc_sigflux = apparent_flux(
            magpsf, sigmapsf, magnr, sigmagnr, isdiffpos, jansky=False
        )

        # apparent mag and its error from fluxes
        dc_mag = -2.5 * np.log10(dc_flux)
        dc_sigmag = dc_sigflux / dc_flux * 1.0857
        
        return dc_mag, dc_sigmag

    return magpsf, sigmapsf