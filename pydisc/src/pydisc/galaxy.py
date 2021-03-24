# -*- coding: utf-8 -*-
"""
This is the main class of the package - disc - gathering
data, attributes and functions defining a disc model.

This module include computation for torques, Tremaine-Weinberg
method, in-plane velocities (Maciejewski et al. method).
"""

from numpy import deg2rad, rad2deg, cos, sin, arctan, tan, pi
from astropy import units as u
from . import local_units as lu

from . import transform, misc_functions

class Galaxy(object):
    """
    Attributes
    ----------
    distance
    pc_per_arcsec
    inclin
    PAnodes
    PAbar
    """
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:
                distance
                inclin
                PAnodes
                PAbar
        """

        # Distance in Mpc
        self.distance = kwargs.pop('distance', 10.)
        # Inclination in degrees
        self.inclin = kwargs.pop("inclination", 60.)
        # PA of the line of nodes
        self.PAnodes = kwargs.pop("PAnodes", 0.)
        # PA of the bar
        self.PAbar = kwargs.pop("PAbar", 0.)

    @property
    def pc_per_arcsec(self):
        """Return the scale pc per arcsecond"""
        return misc_functions.get_pc_per_arcsec(self.distance)

    @property
    def _eq_pc_per_arcsec(self):
        return [(u.pc, u.arcsec,
                 lambda x: x / self.pc_per_arcsec,
                 lambda x: self.pc_per_arcsec * x)]

    def convert_xyunit(self, xyunit, newunit=u.kpc):
        return lu.get_conversion_factor(xyunit, newunit,
                                        equiv=self._eq_pc_per_arcsec)

    def pc_per_xyunit(self, xyunit):
        """pc per unitXY
        """
        return self.convert_xyunit(xyunit, u.pc)[0]

    def pc_per_pixel(self, xyunit, pixel_scale):
        """pc per pixel
        """
        return self.pc_per_xyunit(xyunit) * pixel_scale

    @property
    def inclin(self) :
        return self._inclin

    @inclin.setter
    def inclin(self, inclin) :
        self._inclin = inclin
        self._inclin_rad = deg2rad(inclin)
        self._mat_inc = transform.set_stretchmatrix(coefY=1. / cos(self._inclin_rad))

    @property
    def PAnodes(self) :
        return self._PAnodes

    @PAnodes.setter
    def PAnodes(self, PAnodes) :
        self._PAnodes = PAnodes
        self._PAnodes_rad = deg2rad(PAnodes)
        self._mat_lon = transform.set_rotmatrix(self._PAnodes_rad + pi / 2.)

    @property
    def PAbar(self) :
        return self._PAbar

    @PAbar.setter
    def PAbar(self, PAbar) :
        self._PAbar = PAbar
        self._PAbar_rad = deg2rad(PAbar)
        self._PAbar_lon = PAbar - self.PAnodes
        self._PAbar_lon_rad = deg2rad(self._PAbar_lon)
        self._PAbar_londep_rad = arctan(tan(self._PAbar_lon_rad) / cos(self._inclin_rad))
        self._PAbar_londep = rad2deg(self._PAbar_londep_rad)
        self._mat_bar = transform.set_rotmatrix(self._PAbar_rad + pi / 2.)
        self._mat_bardep = transform.set_rotmatrix(self._PAbar_londep_rad)
