
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

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from jsonpath_rw import jsonpath, parse

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

import SimpleITK as sitk
#import dicom

from skimage.filters import threshold_otsu
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import median


# own libraries
from . import DICOM
from . import helpers
from . import Model


class KMDynamics (DICOM):
    def __init__ (self, nCores = None):
        super(KMDynamics, self).__init__()

        # nCores
        if nCores is None:
            nCores = multiprocessing.cpu_count()

        # debug parameter
        self.debugLevel = 'INFO'

        # phyiscal parameters
        self.TR = None
        self.flipAngle = None

        self.sliceLocations = None

        # this is the volume as numpy array
        self.vol = None

        # contains the time for each slice over all volumes, e.g. slice 0 might have time 20171131 012345.650000
        self.seriesTimes = None

        # contains the volume id for each silce, e.g. slice 0 might be volume 13 etc.
        self.seriesVolIDs = None

        # contains the time for each volume, i.e. if there are 30 volumes, we have 30 time stamps
        self.volumeTimes = None

        # names of volumes are just their times, so this could be just replaced by volumeTimes.
        self.volumeNames = None

        # a 2D array (nVolumes x nSlices) that contains the relative timings of the ordered slices,
        # e.g. volume 4 might have [0, 33, 66, 99, ...] ms
        self.relativeTimes = None

        # relative times of volumes, so volume 0 will be at time 0, volume 1 maybe at time 13, etc..
        self.relativeVolumeTimes = None

        # contrast agent relaxivitiy
        self.relaxivity = 4.5
        pass



    def setRelaxitivity (self, relaxivity):
        self.relaxivity = relaxivity


    def getRelaxitivity (self):
        return self.relaxivity


    def mapIndexToVolume (self, n):
        """ Given an index N get the name (i.e. the creation date) of the
            corresponding volume.
        """
        return self.volumeNames[n]


    def mapVolumeToIndex (self, name):
        """ Given a name find the corresponding index. As we assume that the
            volumes are ordered by time (as we name them by the acqusitiontime),
            this index is the time index for the volume.
        """
        return self.volumeNames.index(name)


    def createVolumes (self):
        """ Extract time and dates of all slices and create volume IDs.
            Also extract informations that are important for us, e.g. flipAngle and TR.
            Note: TR will be normalized to seconds by dividing it by 1000 if its larger than 1.0.
            Necessary for all subsequent operations.
        """
        self.acquisitionTime =  [self.extractMetaTag ("AcquisitionTime", imageNumber = r) for r in range(len(self.series_file_names))]
        self.acquisitionDate =  [self.extractMetaTag ("AcquisitionDate", imageNumber = r) for r in range(len(self.series_file_names))]
        self.acquisitionNumber = [self.extractMetaTag ("AcquisitionNumber", imageNumber = r) for r in range(len(self.series_file_names))]
        self.contentTime = [self.extractMetaTag ("ContentTime", imageNumber = r) for r in range(len(self.series_file_names))]
        self.contentDate = [self.extractMetaTag ("ContentDate", imageNumber = r) for r in range(len(self.series_file_names))]
        self.sliceLocations = [self.extractMetaTag ("SliceLocation", imageNumber = r) for r in range(len(self.series_file_names))]

        # for qiba we indeed need to look at the times
        # for our own example acquisitionNumber seemed to be ok too.
        #self.seriesVolIDs = set(self.acquisitionNumber )

        # can assume here that the volumes can be ordered by time
        self.seriesTimes = [x[1].replace(' ','') + " " + x[0].replace(' ','') for x in zip(self.acquisitionTime, self.acquisitionDate)]
        self.seriesTimes = [time.mktime( datetime.datetime.strptime(x, "%Y%m%d %H%M%S.%f").timetuple() ) for x in self.seriesTimes]

        # for now volumes are just named by their times
        self.volumeNames = sorted(list(set(self.seriesTimes)))
        self.volumeTimes = self.volumeNames
        self.relativeVolumeTimes = [t-self.volumeTimes[0] for t in self.volumeTimes]

        self.seriesVolIDs = [self.volumeNames.index(t) for t in self.seriesTimes]

        try:
            TR = self.extractMetaTag ("RepetitionTime", imageNumber = 0)
            self.TR = float(self.extractMetaTag ("RepetitionTime", imageNumber = 0))
        except:
            self.TR = None
            self.TR = 15/1000


        if self.TR == None:
            self.error ("There is no TR meta tag in this KM Dynamics.")
        # TR gets normalized to seconds
        if self.TR > 1.0:
            self.TR = self.TR/1000

        self.flipAngle = self.extractMetaTag ("FlipAngle", imageNumber = 0)
        if self.flipAngle == None:
            self.error ("There is no FlipAngle meta tag in this KM Dynamics.")
        self.flipAngle = np.pi/180.0*float(self.flipAngle)
        return self.seriesVolIDs


    def getVolumeIdx (self, N):
        """ Get the internal indices of all slices for volume with index N.
            The slices are sorted in time.
            Note: Indexing starts with 0.
        """
        #
        if N in range(len(self.volumeTimes)):
            gpIdx = [i for i, r in enumerate(self.seriesVolIDs) if r == N]
            gpAcNum = [self.contentTime[g] for g in gpIdx]
            sortedGroupIndices = list(zip(*sorted(zip(gpAcNum, gpIdx))))[1]
        else:
            self.error ("Unknown volume")
        return sortedGroupIndices


    def checkVolumes (self):
        """ Make sure that all volumes have the same number of slices.
        """
        nSlices = None
        self.createVolumes()
        for N in self.seriesVolIDs:
            sortedGroupIndices = self.getVolumeIdx (N)
            if nSlices == None:
                nSlices = len(sortedGroupIndices)
            else:
                if nSlices != len(sortedGroupIndices):
                    self.error ("Slices have not the same size")
        return nSlices


    def getNSlices (self):
        return self.checkVolumes ()


    def getNVolumes (self):
        self.createVolumes ()
        return len(self.volumeNames)


    def getVolumeShape (self):
        return self.sitk_ndarray.shape[1:]


    def getVolumeAsArray (self, N):
        """ Given an index, we retrieve the corresponding volume as a numpy array.
        """
        self.createVolumes()

        # reset times?
        if N in range(self.getNVolumes()):
            sortedGroupIndices = self.getVolumeIdx (N)

            tmpT = [self.contentTime[x] for x in sortedGroupIndices]
            tmpD = [self.contentDate[x] for x in sortedGroupIndices]

            useSliceLocation = True
            if useSliceLocation == True:
                baseTime = self.volumeTimes[N]
                #print (baseTime)
                tmpTimes = [self.sliceLocations[x] for x in sortedGroupIndices]
                tmpTimes = [float(z) for z in tmpTimes]
                output = [0] * len (tmpTimes)
                for i, x in enumerate(sorted(range(len(tmpTimes)), key=lambda y:tmpTimes[y])):
                    output[x] = i
                relTimes = output
                 #absTimes = [t + baseTime for t in output ]
                sortedGroupIndices = [sortedGroupIndices[z] for z in relTimes]

                tmpZ = list(zip(tmpD, tmpT))
                absTimes = [datetime.datetime.strptime((x[0] + '.' + x[1]).replace(' ',''), "%Y%m%d.%H%M%S.%f") for x in tmpZ]
            else:
                # make seconds out of the dates
                tmpZ = list(zip(tmpD, tmpT))

                absTimes = [datetime.datetime.strptime((x[0] + '.' + x[1]).replace(' ',''), "%Y%m%d.%H%M%S.%f") for x in tmpZ]
                relTimes = [(z-absTimes[0]) for z in absTimes]
                relTimes = [((q.seconds + q.days * 24 * 3600) * 10**6) + q.microseconds for q in relTimes]

            if not hasattr(self, 'relativeTimes') or self.relativeTimes is None:
                self.relativeTimes = np.zeros( (len(self.volumeTimes), len(sortedGroupIndices)), dtype = np.int64 )
            self.relativeTimes [N, :] = relTimes

            volumeAsArray = np.zeros( (len(sortedGroupIndices),) + self.sitk_ndarray.shape[1:])

            for i, v in enumerate(sortedGroupIndices):
                volumeAsArray [i, :, :] = self.sitk_ndarray [v, :, :]
        else:
            self.error("Index out of bounds. There are only " + str(len(self.volumeNames)) + " volumes.")
        return volumeAsArray


    def setVolumeAsArray (self, N, volumeAsArray):
        self.createVolumes()
        if N in range(self.getNVolumes()):
            gpIdx = [i for i, r in enumerate(self.acquisitionNumber) if int(r) == N]
            gpAcNum = [self.contentTime[g] for g in gpIdx]
            sortedGroupIndices = list(zip(*sorted(zip(gpAcNum, gpIdx))))[1]

            for i, v in enumerate(sortedGroupIndices):
                self.sitk_ndarray [v, :, :] = volumeAsArray [i, :, :]
            #p = mosaic(volumeAsArray)
            #scipy.misc.imsave(str(N) + "_mosaic.jpg", p.astype(np.uint8))
        else:
            self.error("Index out of bounds. There are only " + str(len(self.volumeNames)) + " volumes.")
        return True


    def getAsArray (self):
        """ Get the whole 4d volume as numpy array, slice by slice.
        """

        if self.vol is None:

            # TODO: need to coregister first, but for now only check if slide numbers are ok or not
            nSlices  = self.getNSlices()
            nVolumes = self.getNVolumes()
            volShape = self.getVolumeShape ()
            volArr = np.zeros ( (nVolumes, nSlices, ) + volShape)
            #print (volArr.shape)

            # make sure the volumes are numbered from 1..#Volumes
            # vset = set([x-1 for x in self.seriesVolIDs])
            # nset = set(range(len(self.seriesVolIDs)))
            # if len(vset-nset) != 0 or len(nset-vset) != 0:
            #     self.error ("Volume IDs are not consecutive and/or do not start with 1")

            for i in range(nVolumes):
                tmp = self.getVolumeAsArray (i)
                #print (tmp.shape)
                volArr [i, :, :, :] = tmp
            self.vol = volArr.copy()
        return self.vol



    def setAsArray (self, volArr ):
        nSlices  = self.getNSlices()
        nVolumes = self.getNVolumes()
        volSize = self.getVolumeShape ()

        volShape = ((nVolumes, nSlices, ) + volSize)

        if volArr .shape != volShape:
            self.error ("Volume shape does not fit.")

        for i, N in enumerate(self.seriesVolIDs):
            self.setVolumeAsArray (N, volArr [i, :, :, :])
        return volArr



    def plotSlices (self, times = 0, slices = None, cmap = 'nipy_spectral', vmin = None, vmax = None):
        # make sure we have the whole volume
        if not hasattr(self, 'vol'):
            self.vol = self.getAsArray ()

        ## FIXME: make vol computed only once everywhere

        nSlices = self.getNSlices()
        nVolumes = self.getNVolumes()

        slices = helpers.getSliceList (slices, nSlices)
        times = helpers.getSliceList (times, nVolumes)
        for s in slices:
            for t in times:
                print ("Time:" , str(t), ", Slice:", str(s))
                plt.figure()
                plt.imshow (self.vol [t, s, :, :], interpolation='nearest', cmap = cmap, vmin = vmin, vmax = vmax)



    def getAveragedSignalCurve (self, mask = 'SNR'):
        # make sure we have the whole volume
        vol = self.getAsArray ()

        # make sure we have timings
        if not hasattr(self, 'relativeVolumeTimes') or self.relativeVolumeTimes is None:
            self.error ("No relative timings found. Internal error.")

        nSlices = self.getNSlices()
        nVolumes = self.getNVolumes()


        # take average over all points?
        curveVol = None
        if mask is None:
            curveVol = vol[:,:,:,:]
            pass
        else:
            if mask == 'SNR':
                # get SNR mask
                if not hasattr(self, 'SNRMap'):
                    self.computeSNRMask ()
                mask = self.SNRMap

                # how to speed up?
                curveVol = vol.copy()
                nanMask = mask.copy()
                nanMask = np.asarray(nanMask, dtype=float)
                nanMask[nanMask == 0] = np.NaN

                for v in range (nVolumes):
                    curveVol [v, :, :, :] = vol[v, :, :, :]*nanMask
                self.curveVol = curveVol


        # compute averaged curve
        #print(curveVol.shape)
        curve = np.nanmean (curveVol, axis=1)
        #print(curve.shape)
        curve = np.nanmean (curve, axis=1)
        #print(curve.shape)
        curve = np.nanmean (curve, axis=1)
        #print(curve.shape)

        # both should have th same size, just make sure
        #curve = vol [:, slice, x, y]
        if len(self.relativeVolumeTimes) != curve.shape[0]:
            self.error ("Relativ times do not fit to the curves. Internal error.")
        return curve



    def plotAveragedSignalCurve (self, mask = 'SNR'):
        curve = self.getAveragedSignalCurve (mask)
        f = plt.figure()
        plt.plot (self.relativeVolumeTimes, curve, 'yo-')
        return f



    def getSignalCurve (self, slice, r, c, smooth = 10):
        # make sure we have the whole volume
        vol = self.getAsArray ()

        # make sure the slice is only one of it
        if not isinstance(slice, int):
            self.log ("Can only plot exactly one signal curve")

        # make sure we have timings
        if self.relativeVolumeTimes is None:
            self.error ("No relative timings found. Internal error.")

        m,n,p,q = np.shape(vol)

        if r < 0 or r >= p or c < 0 or c >= q:
            self.error ("The selected point is outside the image array.")

        # plot an average curve
        if smooth > 0:
            curveVol = vol[:,slice,(r-smooth):(r+smooth),(c-smooth):(c+smooth)]
            curve = np.mean (curveVol, axis=1)
            curve = np.mean (curve, axis=1)
        else:
            curve = vol[:,slice,r,c]

        print(curve.shape)

        # both should have th same size, just make sure
        if len(self.relativeVolumeTimes) != curve.shape[0]:
            self.error ("Relativ times do not fit to the curves. Internal error.")

        return curve



    def plotSignalCurve (self, slice, r, c, smooth = 10):
        # make sure we have the whole volume
        p = self.getSignalCurve (slice, r, c, smooth = smooth)
        f = plt.figure()
        plt.plot (self.relativeVolumeTimes, p, 'yo-')
        #return f


    def computeR10From (self, f, slices = None):
        # TODO: check if f is a flipAngle..

        R10_, S0_ = f.computeR10Image (slices = slices)
        self.R10_ = R10_.sitk_ndarray.copy()
        self.S0_ = S0_.sitk_ndarray.copy()
        return R10_, S0_



    def computeiAUCG (self, slices = None, duration = 60, mask = 'SNR', denoise = None):
        ''' mask: None=Dont use any, SNR = use SNR, else mask = binary volume
        '''
        vol = self.getAsArray()

        # make sure we have relativ timings
        if self.relativeVolumeTimes is None:
            self.error ("No relative timings found. Internal error.")

        # find the start point--
        # mask the points where the signal is valid
        if mask == 'SNR':
            if not hasattr(self, 'SNRMap'):
                self.computeSNRMask ()
            mask = self.SNRMap.copy()

        if mask is None:
            mask = np.ones(vol.shape[1:])

        # final check
        if mask.shape != vol[0,:,:,:].shape:
            self.error ("Mask size must be the same size as any volume (e.g. v[0,:,:,:]).")

        nSlices  = self.getNSlices()
        nVolumes = self.getNVolumes()

        # average over all curves with "good" voxels
        p = self.getAveragedSignalCurve (mask = None)

        # fit a polynomial
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spl = UnivariateSpline (self.relativeVolumeTimes, p, k = 4)

        # find roots.
        ## FIXME: here assume that the curve first drops then increases till the max
        xpts = spl.derivative().roots()
        ypts = spl(xpts)

        # expect only two roots for now
        if len(xpts) != 2:
            self.error ("Internal error: please fix the fitting.")
        if xpts[0] > xpts[1]:
            self.error ("Internal error: order of extremal points not correct.")

        # if plot fit == True:
        #        ...

        # determine cut point
        A = (xpts[0], ypts[0])
        B = (xpts[1], ypts[1])
        b = np.mean(spl (self.relativeVolumeTimes[0:3]))
        C = (0,b)
        D = (100000,b) # arbitrary, its a constant line anyway

        # copied from stack overflow
        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
               raise Exception('lines do not intersect')

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return x, y

        cutpt = line_intersection((A, B), (C, D))

        # find cut point
        startTime = cutpt[0]
        self.log ("Cutpoint is " + str(startTime))

        # convert slice
        slices = helpers.getSliceList (slices, nSlices)

        if not hasattr(self, 'iAUCG'):
            self.iAUCG = np.zeros (vol.shape[1:])

        # only for the good voxels!
        for s in slices:
            for x in range(vol.shape[2]):
                for y in range(vol.shape[3]):
                    if mask [s, x, y] == True:
                        fullCurve = vol[:, s, x, y]

                        if len(self.relativeVolumeTimes) != fullCurve.shape[0]:
                            self.error ("Internal error: Voxel curve size does not fit.")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            vCurve = UnivariateSpline (self.relativeVolumeTimes, fullCurve, k = 4, s = fullCurve.shape[0]/4)

                        # now integrate
                        i = vCurve.integral (startTime, startTime + duration)

                        plotEachLine = False
                        if plotEachLine == True:
                            plt.figure()
                            plt.plot (self.relativeVolumeTimes, vCurve(self.relativeVolumeTimes))
                            print(i)
                            time.sleep(5)

                        self.iAUCG [s, x, y] = i
                    else:
                        self.iAUCG [s, x, y] = 0

        if denoise == 'Simple':
            # is the noise really pointwise or do we have the same 'pixels' over all slices?
            # from images it seems that we can process each slice by itself.
            for s in slices:
                # generate a mask, somehow.
                # here we simply threshold.
                # the threshold should probably be based on physics, or if not, then we
                # try to determine it ourselves.
                curSlice = self.iAUCG[s,:,:].copy()
                curMask = np.zeros_like(curSlice)

                # remove top values where the gradient becomes too large
                # somehow adaptive versio;
                hv = np.sort(curSlice.flatten())
                factor = 10
                for b in range(9500, 9995, 5):
                    cut0 = hv[int((b-5)/10000*hv.shape[0])]
                    cut1 = hv[int(b/10000*hv.shape[0])]
                    if (cut1 - cut0) > cut0/factor:
                        #print ("Cut at " + str(b) + ", value; "+ str(cut0))
                        cut = cut0
                        break

                # static alternative:
                #m = np.median(curslices)
                #cut = factor*m
                curMask[curSlice<0] = 1
                curMask[curSlice>cut]= 1

                # to apply median filtering we need to make sure the image is between -1..1
                # and also we dont want to scale to our noise

                # turn into float -1..1
                m = np.median(curSlice)
                medianImg = curSlice.copy()
                medianImg = medianImg * (1-curMask) + m*curMask

                # scale, but very unfortunate, np,median yields 0..255 pixel values
                sf = max (np.max(medianImg), abs(np.min(medianImg)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmp  = median(medianImg/sf, disk(5))/255*sf
                #imshow(tmp[:,:])

                newSlice = curSlice * (1-curMask) + tmp*curMask
                #imshow(new, vmin=3000, vmax=35000)
                self.iAUCG[s,:,:] = newSlice

        return self.iAUCG.copy()



    # the R map has the same size as the signal
    def dce_to_r1eff(self, signal, S0_, R10_):
        #print ('converting DCE signal to effective R1')

        #assert(self.flipAngle > 0.0)
        if self.TR < 0.0:
            self.error  ("TR is smaller than zero, not possible.")

        if self.TR > 1.0:
            self.log ("TR is larger than 1.0. Normalizing it to seconds!")
            self.TR = self.TR/1000

        # compute normalized signal
        s_ = signal/S0_

        # helper
        E0 = exp(-R10_ * self.TR)

        th = self.flipAngle
        #print(s_.shape)
        #print(E0.shape)

        N = 1.0 - s_ + s_*E0 - E0 * cos(th)
        D = (1.0 - s_*cos(th) + s_*E0*cos(th) - E0*cos(th))

        E = N/D
        R = (-1.0 / self.TR) * log(E)
        return R



    def computeT1 (self, T10Image, S0Image = None, slices = None):
        vol = self.getAsArray ()
        nSlices  = self.getNSlices()

        # check that the shapes fit
        if vol.shape[1:] != T10Image.sitk_ndarray.shape:
            self.error ("Shape of T10 Image does not fit to KM Volume")

        # convert selected slice to list
        slices = helpers.getSliceList (slice, nSlices)

        if S0Image is None:
            S0mean_ = (vol[0,0,:,:]+vol[0,1,:,:])/2
        else:
            S0mean_ = S0Image.sitk_ndarray

        T10_ = T10Image.sitk_ndarray
        print (T10_.shape)
        R10_ = 1.0/T10_

        # for now we do it slice by slice
        from tqdm import tqdm_notebook

        R1_ = np.zeros( (vol.shape))
        T1_ = np.zeros( (vol.shape))
        for s in tqdm_notebook(slices):
            curR10slices = R10_ [s, :, :]
            curS0slices = S0mean_ [s, :, :]
            R1_ = self.dce_to_r1eff (vol[:, s, :, :], curS0Slice, curR10slices)
            T1_[:, s, :, :] = 1.0/R1_

        ## FIXME: return T1 as DICOM image?

        return T1_


    def createConstantR10 (self, ms = 1000):
        R10_ = np.ones (self.sitk_ndarray.shape[1:])
        self.R10_ = R10_ * ms/1000
        return True


    def computeR1 (self, R10Image, S0Image = None, slices = None):
        vol = self.getAsArray ()
        nSlices  = self.getNSlices()

        # could be DICOM, could be numpy array
        if hasattr (R10Image, 'extractMetaTag'):
            R10_ = R10Image.sitk_ndarray
        else:
            R10_ = R10Image
        self.R10_ = R10_.copy()

        # check that the shapes fit
        if vol.shape[1:] != R10_.shape:
            self.error ("Shape of R10 Image does not fit to KM Volume")

        # convert selected slice to list
        slices = helpers.getSliceList (slices, nSlices)

        if S0Image is None:
            S0mean_ = np.zeros (vol.shape[1:])
            for s in tqdm(slices):
                S0mean_[s, :, :] = (vol[s,0,:,:]+vol[s,1,:,:])/2
            print (S0mean_.shape)
        else:
            if hasattr (S0Image, 'extractMetaTag'):
                S0mean_ = S0Image.sitk_ndarray
            else:
                S0mean_ = S0Image

        # for now we do it slice by slice
        R1_ = np.zeros( (vol.shape))
        for s in tqdm(slices):
            curR10slice = R10_ [s, :, :]
            curS0slice = S0mean_ [s, :, :]
            R1_[:, s, :, :] = self.dce_to_r1eff (vol[:, s, :, :], curS0slice, curR10slice)


        ## return T1 as DICOM image, if it was one
        if hasattr (R10Image, 'extractMetaTag'):
            R10Image.sitk_ndarray = R1_
            return R10Image

        self.R1_ = R1_.copy()
        return R1_



    def computeCAConcentration (self, slices = None, Hct = None):
        self.log ("Computing CA Concentration")

        # TODO: make sure all things are in place
        self.Ct = self.R1_.copy()
        nVol = self.getNVolumes()

        for t in range(nVol):
            for s in helpers.getSliceList(slices, self.getNSlices()):
                self.Ct[t, s, :, :] = self.R1_[t, s, :, :] - self.R10_[s, :, :]

        self.Ct = self.Ct/self.relaxivity

        if Hct is not None:
            self.Ct = self.Ct/(1.0 - Hct)
        return self.Ct



    # could be moved away to helpers, as it is quite general
    def computeGeneralSNR (self, im1, im2):
        ''' Compute SNR of two images (see Dietrich et al. 2007,
            JMRI, 26, 375) '''
        #self.log ('Computing signal-to-noise ratio')

        # determine mask, otsu works in 3d too.
        th = threshold_otsu (im1)
        mask = im1 > th

        # compute SNR
        N = (im1[mask] + im2[mask]).mean()
        D = (im1[mask] - im2[mask]).std()
        SNR = N/D * 1/ sqrt(2)

        return SNR, mask


    def computeSNR (self):
        ''' Compute SNR of two images (see Dietrich et al. 2007,
            JMRI, 26, 375) '''

        # make sure we have volume
        vol = self.getAsArray()

        SNR, mask = self.computeGeneralSNR (vol[0, :, :, :], vol[1, :, :, :])
        self.SNR = SNR
        self.SNRMask = mask

        return SNR, mask



    def computeSNRMask (self):
        ''' Compute SNR of two images (see Dietrich et al. 2007,
            JMRI, 26, 375) '''
        #self.log ('Computing signal-to-noise ratio')

        # make sure we have volume
        vol = self.getAsArray()

        SNR, mask = self.computeSNR ()
        mask = vol[0, :, :, :] > (vol[0, :, :, :].max() / SNR)
        self.SNRMap = mask

        return mask



    '''
    Next an optional mask of voxels to process can be supplied in the MAT file as the
variable mask. If not supplied, an automatic mask is generated from a variation of the
signal enhancement ratio (SER, Hylton et al., 2012), defined here as the mean signal in
each voxel in the last three dynamics divided by the mean of the signal in the voxel in the
first three dynamics. (Note that this requires the acquisition of three pre-contrast time
    '''


    # computes it for all masks for now, as it is simple thresholding,
    # and this does not necessarily work with slices-- as we mask pixels
    # in the first volume by threshold*data.max, and data.max is the
    # max over the whole volume. if we do it by slice, we get a different
    # (computed) threshold.

    ## Note that this requires the acquisition of three pre-contrast time points!!

    def computeSERMask (self, th = 0.01):
        ''' Compute max signal enhancement ratio for dynamic data '''
        print ('computing signal enhancement ratios')

        # make sure we have volume
        vol = self.getAsArray()

        # check that we have at least 6 time points
        if vol.shape[0] < 6:
            self.error ("For computation of the SER mask we need at least 6 time points, the first three must be pre-contrast")

        if th < 0.0:
            self.error ("Threshold is negative.")
        th = th *vol.max()

        # get mask of 'good' pixels=first thee  vs last three, see julia paper
        # dcemri: those with data in first time volume.
        S0 = np.sum(vol [0:3, :, :, :], axis = 0)
        S1 = np.sum(vol [-3:, :, :, :], axis = 0)

        mask = S0 > th

        SER = np.zeros (S0.shape)
        SER[mask] = S1[mask]/S0[mask]

        self.maskSER = SER
        return SER



    def ext_tofts_integral (self, t, Cp, Kt=0.1, ve=0.2, vp=0.1, uniform_sampling=True):
        """ Extended Tofts Model, with time t in min.
            Works when t_dce = t_aif only and t is uniformly spaced.
        """
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



    def tofts_integral (self, t, Cp, Kt=0.1, ve=0.2, uniform_sampling=False):
        ''' Standard Tofts Model, with time t in min.
            Current works only when AIF and DCE data are sampled on
            same grid.  '''
        nt = len(t)
        Ct = np.zeros(nt)
        for k in range(nt):
            if uniform_sampling:
                tmp = cumtrapz(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1], t[:k+1], initial=0.0)
                Ct[k] = tmp[-1]
                #Ct[k] = simps(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1], dx=t[1]-t[0])
            else:
                Ct[k] = simps(exp(-(Kt/ve)*(t[k] - t[:k+1]))*Cp[:k+1], x=t[:k+1])
        return Kt*Ct


    def loadAIF (self):
        """ Load the AIF.
        """
        mat = loadmat('./data/AIF.mat')
        aif_y = mat['data']
        aif_x = mat['t']
        #plot(aif_x, aif_y)

        # fit a spline
        # use interpolated spline to resample aif to our time points
        self.AIF = InterpolatedUnivariateSpline (aif_x, aif_y, ext = 'const', k = 2)
        self.Cp = self.AIF(self.relativeVolumeTimes)
        return self.Cp.copy()


    def computePerfusionMaps (self, Ct = None, Cp = None, slices = None, mask = None,
        extended = False, plot_each_fit = False):
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

        # get volume
        vol = self.getAsArray ()
        nSlices  = self.getNSlices()

        # make sure the timings do exist already
        if self.relativeVolumeTimes is None:
            self.error ("Relative times should have been computed internally. Seems to be an internal error.")


        # resample AIF if not given
        if Cp is None:
            Cp = self.loadAIF ()
        else:
            self.Cp = Cp.copy()



        # check if all things are in place
        if hasattr( self, 'R1_') == False:
            self.log  ("No R1 map was provided. Doing this now.")
            # so compute R1 map, this in turn will check for a R10 map
            if hasattr( self, 'R10_') == False:
                self.error ("No R10 Image was given. Cannot compute this, please create it by using FlipAngleDICOM.")
            self.computeR1 (self.R10_, slices = slices)



        if Ct is None:
            if hasattr( self, 'Ct') == False:
                self.log  ("No Contrast agent concentration was computed. Doing this now.")
                self.computeCAConcentration (slices)
            Ct = self.Ct



        # get slices to compute
        slices = helpers.getSliceList (slices, nSlices)



        N = np.prod(vol.shape[1:])
        self.log("Fitting " + str(N) + " voxels.")


        nSlices = Ct.shape[1]
        mapShape = (nSlices,) + Ct.shape[2:4]


        # check if we have a map
        if mask is None:
            # do we have an own mask?
            if hasattr(self, 'mask') == False:
                mask = np.ones (mapShape) #  np.asarray(list(range(N)))
            else:
                mask = self.mask

        # convert mask to indicies, if necessary
        if len(mask.shape) != 3:
            self.error ("Mask must have the same shape as a single volume.")



        # TODO: check model type
        model = "Tofts-Kety"
        if model == "Tofts-Kety":
            self.log ("Using Tofts-Key model")
            self.m = Model.Model ()
            Ktrans, ve = self.m.fit (model = model, relativeVolumeTimes = self.relativeVolumeTimes,
                    Ct = Ct, Cp = self.Cp, slices = slices, mask = mask)
            self.Ktrans = Ktrans.copy()
            self.ve = ve.copy ()

            return self.Ktrans, self.ve

        if model == "Extended Tofts-Kety":
            self.log ("Using Extended Tofts-Key model")
            self.m = Model.Model ()
            Ktrans, ve, vp = self.m.fit (model = model, relativeVolumeTimes = self.relativeVolumeTimes,
                    Ct = Ct, Cp = self.Cp, slices = slices, mask = mask)
            self.Ktrans = Ktrans.copy()
            self.ve = ve.copy ()
            self.vp = vp.copy ()

            return self.Ktrans, self.ve, self.vp

        self.error ("Unknown model selected")



#
