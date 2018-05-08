
#import shmarray
import warnings
import datetime
import time
import tempfile
import shutil
from glob import glob
import sys, time, os
import os, os.path, sys, configparser, collections
import importlib.util
from joblib import Parallel, delayed
import multiprocessing

import dill as pickle
import joblib


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from jsonpath_rw import jsonpath, parse
from copy import deepcopy

from tqdm import tqdm_notebook
from tqdm import tqdm

import scipy.misc
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.io import loadmat, savemat
from scipy.integrate import cumtrapz, simps
from scipy.optimize import curve_fit, root
from scipy.interpolate import UnivariateSpline

import numpy as np
from numpy import sin, cos, tan, exp, log, sqrt



from skimage.filters import threshold_otsu
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import median




class Model (object):
    def __init__ (self, nCores = None):

        self.debugLevel = 5
        # nCores
        if nCores is None:
            self.nCores = multiprocessing.cpu_count()
        else:
            self.nCores = nCores


    def log (self, s, l = 5):
        if self.debugLevel >= l:
            print (s)

    def error (self, s):
        print ("ERROR:", s)
        exit(-1)


    def ext_tofts_integral (self, t, Kt=0.1, ve=0.2, vp=0.1, uniform_sampling=True):
        """ Extended Tofts Model, with time t in min.
            Works when t_dce = t_aif only and t is uniformly spaced.
        """
        Cp = self.Cp
        nt = len(t)
        Ct = np.zeros(nt)
        for k in range(nt):
            if uniform_sampling:
                tmp = cumtrapz(exp(-Kt*(t[k] - t[:k+1])/ve)*Cp[:k+1],
                               t[:k+1], initial=0.0) + vp * Cp[:k+1]
                Ct[k] = tmp[-1]
            else:
                Ct[k] = simps(exp(-Kt*(t[k] - t[:k+1])/ve)*Cp[:k+1],
                              t[:k+1]) + vp * Cp[:k+1]
        return Ct*Kt



    def tofts_integral (self, t, Kt=0.1, ve=0.2, uniform_sampling=False):
        ''' Standard Tofts Model, with time t in min.
            Current works only when AIF and DCE data are sampled on
            same grid.  '''
        Cp = self.Cp
        nt = len(t)
        Ct = np.zeros(nt)
        for k in range(nt):
            if uniform_sampling:
                tmp = cumtrapz(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1], t[:k+1], initial=0.0)
                Ct[k] = tmp[-1]
                #Ct[k] = simps(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1], dx=t[1]-t[0])
            else:
                try:
                    tmp = exp(-Kt/ve)
                except:
                    tmp = 1
                Ct[k] = simps(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1], x=t[:k+1])
        return Kt*Ct


    def fitVoxel (self, s, x, y):
        # only for masked pixels
        #print (mask.shape)
        if self.mask [s, x, y] == True:
            try:
                #print(Ct[:,s,x,y])
                #self.F = fit_func
                popt, pcov = curve_fit (self.fit_func, self.relativeVolumeTimes, self.Ct[:, s, x, y], p0 = self.initPoint)
                #print(".")
            except (ValueError, RuntimeError) as e:
                #print("A")
                popt = self.popt_default
                pcov = self.pcov_default
            #stop("A")
            self.Ktrans[s, x, y] = popt[0]
            self.ve[s, x, y] = popt[1]
            try:
                self.Ktrans_cov[s, x, y] = pcov[0,0]
                self.ve_cov[s, x, y] = pcov[1,1]
            except TypeError:
                None #print idx, popt, pcov
            if self.model == "Extended Tofts-Kety":
                self.vp [s, x, y] = popt[2]
                self.vp_cov [s, x, y] = pcov[2,2]



    def fit (self, model = "Tofts-Kety", relativeVolumeTimes = None, Ct = None, Cp = None, slices = None, mask = None, plot_each_fit = False):
        """
        Solve tissue model for each voxel and return parameter maps.

        more detailed.

        Parameters
        ----------
            Cp : (N,) array_like or None
                Cp/AIF. If None given, a modeled AIF curve will be resampled to the
                relative Times of the dynamics.
            Ct: tissue concentration of CA, expected to be N x Ndyn
            t: time samples, assumed to be the same for Ct and Cp
            extended: if True, use Extended Tofts-Kety model.
            idxs: indices of ROI to fit

        See Also
        --------



        Notes
        -----


        Examples
        --------


        """

        self.log ("Computing perfusion maps.")

        # TODO: other checks

        # convert mask to indicies, if necessary
        if len(mask.shape) != 3:
            self.error ("Mask must have the same shape as a single volume.")
        self.mask = mask

        if relativeVolumeTimes is None:
            self.error ("Timings are necessary.")

        self.Cp = Cp
        self.Ct = Ct
        self.model = model
        self.relativeVolumeTimes = relativeVolumeTimes

        # maps are over all volumes, so slices x space coordinates
        nSlices = Ct.shape[1]
        mapShape = (nSlices,) + Ct.shape[2:4]
        #print(mapShape)
        self.Ktrans = np.zeros ( mapShape )
        self.ve = np.zeros ( mapShape )
        self.Ktrans_cov = np.zeros ( mapShape )
        self.ve_cov = np.zeros ( mapShape )


        if model == "Extended Tofts-Kety":
            self.log ('Using Extended Tofts-Kety')
            self.fit_func = self.ext_tofts_integral
            self.initPoint = [0.01, 0.01, 0.01]
            self.popt_default = [-1,-1,-1]
            self.pcov_default = np.ones ((3,3))

        if model == "Tofts-Kety":
            self.log ('Using Standard Tofts-Kety')
            self.vp = np.zeros (mapShape)
            self.vp_cov= np.zeros (mapShape)
            self.fit_func = self.tofts_integral
            self.initPoint = [0.01, 0.01]
            self.popt_default = [-1,-1]
            self.pcov_default = np.ones((2,2))


        # slice by slice
        #import shmarray HAB NICHT NUR EINS.
        for s in tqdm(slices):
            # threading cannot be used, because of GIL and whatever, it just stalls.
            # max_nbytes = 1 should force the workers to use a shared memory, so they all write
            # to the same numpy array. else, they write in their own copy, which gets discarded
            # by the end of the worker, and we get back a black image.
            results = joblib.Parallel(n_jobs=self.nCores, max_nbytes = 1, backend="multiprocessing")(delayed(self.fitVoxel)(s, x, y) for y in range (Ct.shape[3]) for x in range (Ct.shape[2]))
            #results = [self.fitVoxel(s, x, y) for y in range (Ct.shape[3]) for x in range (Ct.shape[2])]


        if self.model == "Extended Tofts-Kety":
            return self.Ktrans, self.ve, self.vp

        if self.model == "Tofts-Kety":
            return self.Ktrans, self.ve

        self.error ("Unknown model selected!")
        pass


#
