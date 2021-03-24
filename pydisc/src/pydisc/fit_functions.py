# -*- coding: utf-8 -*-
"""
This provides a set of functions to fit profiles and maps
"""
import numpy as np
from scipy.optimize import leastsq
from scipy.odr import Model, ODR, RealData

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

list_functions = ['plummer', 'pplummer', 'expodisc']

def plummer(r, rho0=1., rp=1.):
    """Derive a plummer sphere on r"""
    return rho0 / (1. + (r / rp)**2)**2.50

def pplummer(r, d0=1., rp=1.):
    """Derive a plummer sphere on r"""
    return d0 / (1. + (r / rp)**2)**2

def expodisc(r, d0=1., rd=1.):
    return d0 * np.exp(-r / rd)

def fit_disc_sphe(r, flux, fdisc="expodisc", fsph="pplummer"):
    """Fit the spheroid on top of a disc

    Args:
        r:
        flux:

    Returns:

    """
    if fdisc not in list_functions:
        print("ERROR[fit_spheroid] function {} was not found".format(fdisc))
        return
    func_disc = globals()[fdisc]
    if fsph not in list_functions:
        print("ERROR[fit_spheroid] function {} was not found".format(fsph))
        return
    func_sphe = globals()[fsph]

    sel_good = (flux > 0.)
    I_max = np.max(flux[sel_good], axis=None)
    rmax = np.max(r[sel_good], axis=None)

    # Initial values
    Imax_sphe, Rsphe = I_max * 0.8, 0.1 * rmax
    Imax_disc, Rdisc = I_max * 0.2,  0.5 * rmax

    def total(x, par):
        return spheroid(x, par) + disc(x, par)

    def spheroid(x, par):
        return func_sphe(x, par[0], par[1])

    def disc(x, par):
        return func_disc(x, par[2], par[3])

    def residuals(par, y, x):
        """Residual function for the fit

        Args:
            par:
            y:
            x:
        Return:
            err
        """
        return y - total(x, par)

    # Initial conditions
    par0 = [Imax_sphe, Rsphe, Imax_disc, Rdisc]

    return leastsq(residuals, par0, args=(flux[sel_good], r[sel_good]), maxfev=10000), \
           spheroid, disc, total

def linear_fit(ab, x):
    """Linear function of x
    Args:
        ab: [a, b]
            Constant and slope
        x: array or float
            Input data

    Returns:
        a + b * x
    """
    return ab[0] + ab[1] * x

def fit_slope(X, Y, eX=None, eY=None):
    """Will fit a Y vs X with errors using the scipy
    ODR functionalities and a linear fitting function

    Args:
        X:
        Y: input arrays
        eX:
        eY: input uncertaintites [None]

    Returns:
        odr output
    """

    # First filtering the Nan
    good = ~np.isnan(X)
    if eX is None: eX_good = None
    else: eX_good = eX[good]
    if eY is None: eY_good = None
    else: eY_good = eY[good]

    linear = Model(linear_fit)
    odr_data = RealData(X[good], Y[good], sx=eX_good, sy=eY_good)
    odr_run = ODR(odr_data, linear, beta0=[2., 1.])
    return odr_run.run()
