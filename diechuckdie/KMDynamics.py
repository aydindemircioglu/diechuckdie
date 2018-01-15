
import datetime
from joblib import Parallel, delayed
from copy import deepcopy
from tqdm import tqdm
from scipy.integrate import cumtrapz, simps
from scipy.optimize import curve_fit, root
import tempfile
import shutil
import numpy as np
import SimpleITK as sitk
from glob import glob
import sys, time, os
import dicom
import os, os.path, sys, configparser, collections
import importlib.util
from jsonpath_rw import jsonpath, parse
import scipy.misc
from numpy import sin, cos, tan, exp, log
from tqdm import tqdm_notebook


from DICOM import DICOM


class KMDynamics (DICOM):
    def __init__ (self):
        super(KMDynamics, self).__init__()
        self.TR = None
        self.flipAngle = None
        self.absTimes = None
        self.volTimes = None
        pass

    def createVolumes (self):
        self.acquisitionTime =  [self.extractMetaTag ("AcquisitionTime", imageNumber = r) for r in range(len(self.series_file_names))]
        self.acquisitionDate =  [self.extractMetaTag ("AcquisitionDate", imageNumber = r) for r in range(len(self.series_file_names))]
        self.acquisitionNumber = [self.extractMetaTag ("AcquisitionNumber", imageNumber = r) for r in range(len(self.series_file_names))]
        self.contentTime = [self.extractMetaTag ("ContentTime", imageNumber = r) for r in range(len(self.series_file_names))]
        self.contentDate = [self.extractMetaTag ("ContentDate", imageNumber = r) for r in range(len(self.series_file_names))]
        self.volumeIDs = set(self.acquisitionNumber )
        self.volumeIDs = [int(s) for s in (self.volumeIDs)]
        self.TR = float(self.extractMetaTag ("RepetitionTime", imageNumber = 0))
        if self.TR == None:
            self.error ("There is no TR meta tag in this KM Dynamics.")
        self.flipAngle = self.extractMetaTag ("FlipAngle", imageNumber = 0)
        if self.flipAngle == None:
            self.error ("There is no FlipAngle meta tag in this KM Dynamics.")
        self.flipAngle = np.pi/180.0*float(self.flipAngle)
        return self.volumeIDs



    def getVolumeIdx (self, N):
        if N in self.volumeIDs:
            gpIdx = [i for i, r in enumerate(self.acquisitionNumber) if int(r) == N]
            gpAcNum = [self.contentTime[g] for g in gpIdx]
            sortedGroupIndices = list(zip(*sorted(zip(gpAcNum, gpIdx))))[1]
        else:
            self.error ("Unknown volume")
        return sortedGroupIndices



    def getVolumeAsArray (self, N):
        self.createVolumes()
        if self.volTimes is None:
            self.volTimes = np.zeros( (len(self.volumeIDs)), dtype = np.int64 )

        if N in self.volumeIDs:
            #print(N)
            sortedGroupIndices = self.getVolumeIdx (N)

            # take the first slice to set time of the whole volume
            t = sortedGroupIndices[0]
            dt =  self.acquisitionDate[t] + '.' + self.acquisitionTime[t].replace(' ','')
            #print(dt)
            self.volTimes[N-1] = time.mktime( datetime.datetime.strptime(dt, "%Y%m%d.%H%M%S.%f").timetuple() )

            #print ("\n" + str(N) + ' -- ' + str(self.volTimes[N-1]))
            tmpT = [self.contentTime[x] for x in sortedGroupIndices]
            tmpD = [self.contentDate[x] for x in sortedGroupIndices]

            # make seconds out of the dates
            tmpZ = list(zip(tmpD, tmpT))
            #print (tmpZ)
            absTimes = [datetime.datetime.strptime((x[0] + '.' + x[1]).replace(' ',''), "%Y%m%d.%H%M%S.%f") for x in tmpZ]
            relTimes = [(z-absTimes[0]) for z in absTimes]
            relTimes = [((q.seconds + q.days * 24 * 3600) * 10**6) + q.microseconds for q in relTimes]
            #print (relTimes)

            if self.absTimes is None:
                self.absTimes = np.zeros( (len(self.volumeIDs), len(sortedGroupIndices)), dtype = np.int64 )
            self.absTimes [N-1, :] = relTimes

            volumeAsArray = np.zeros( (len(sortedGroupIndices),) + self.sitk_ndarray.shape[1:])
            #print (volumeAsArray.shape)
            for i, v in enumerate(sortedGroupIndices):
                volumeAsArray [i, :, :] = self.sitk_ndarray [v, :, :]
            #p = mosaic(volumeAsArray)
            #scipy.misc.imsave(str(N) + "_mosaic.jpg", p.astype(np.uint8))
        return volumeAsArray


    def setVolumeAsArray (self, N, volumeAsArray):
        self.createVolumes()
        if N in self.volumeIDs:
            gpIdx = [i for i, r in enumerate(self.acquisitionNumber) if int(r) == N]
            gpAcNum = [self.contentTime[g] for g in gpIdx]
            sortedGroupIndices = list(zip(*sorted(zip(gpAcNum, gpIdx))))[1]

            for i, v in enumerate(sortedGroupIndices):
                self.sitk_ndarray [v, :, :] = volumeAsArray [i, :, :]
            #p = mosaic(volumeAsArray)
            #scipy.misc.imsave(str(N) + "_mosaic.jpg", p.astype(np.uint8))
        return True


    def checkVolumes (self):
        nSlices = None
        self.createVolumes()
        for N in self.volumeIDs:
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
        volumeIDs = self.createVolumes ()
        return len(volumeIDs)


    def getVolumeShape (self):
        return self.sitk_ndarray.shape[1:]


    def getAsArray (self):
        # TODO: need to coregister first, but for now only check if slide numbers are ok or not
        nSlices  = self.getNSlices()
        nVolumes = self.getNVolumes()
        volShape = self.getVolumeShape ()
        volArr = np.zeros ( (nVolumes, nSlices, ) + volShape)
        #print (volArr.shape)

        # make sure the volumes are numbered from 1..#Volumes
        vset = set([x-1 for x in self.volumeIDs])
        nset = set(range(len(self.volumeIDs)))
        if len(vset-nset) != 0 or len(nset-vset) != 0:
            self.error ("Volume IDs are not consecutive and/or do not start with 1")

        for i, N in enumerate(self.volumeIDs):
            tmp = self.getVolumeAsArray (i+1)
            #print (tmp.shape)
            volArr [i, :, :, :] = tmp
        return volArr



            ### FIXME: acquisitionNumber as VolumeID means they start with 0 -- need a function to map these to array index for my numpy array


    def setAsArray (self, volArr ):
        nSlices  = self.getNSlices()
        nVolumes = self.getNVolumes()
        volSize = self.getVolumeShape ()

        volShape = ((nVolumes, nSlices, ) + volSize)

        if volArr .shape != volShape:
            self.error ("Volume shape does not fit.")

        for i, N in enumerate(self.volumeIDs):
            self.setVolumeAsArray (N, volArr [i, :, :, :])
        return volArr


    # the R map has the same size as the signal
    def dce_to_r1eff(self, signal, S0_, R10_):
        #print ('converting DCE signal to effective R1')

        #assert(self.flipAngle > 0.0)
        #assert(self.TR > 0.0 and self.TR < 1.0)

        # compute normalized signal
        s_ = signal/S0_

        # helper
        E0 = exp(-R10_ * self.TR)

        th = self.flipAngle
        print(s_.shape)
        print(E0.shape)

        N = 1.0 - s_ + s_*E0 - E0 * cos(th)
        D = (1.0 - s_*cos(th) + s_*E0*cos(th) - E0*cos(th))

        E = N/D
        R = (-1.0 / self.TR) * log(E)
        return R



    def computeT1 (self, T10Image, slice = None):
        vol = self.getAsArray ()

        # check that the shapes fit
        if vol.shape[1:] != T10Image.sitk_ndarray.shape:
            self.error ("Shape of T10 Image does not fit to KM Volume")

        # generate S0mean image

        # iterate over slices
        #for s in range(vol.shape[1]):
        #    signal = vol[:,s,:,:].copy()
        #    T10_ = T10Image.sitk_image[s, :, :]
        #    R10_ = 1.0/T10_
        #    R1_[:,:,t] = self.dce_to_r1eff (signal, S0mean_, R10_)


        if slice is None:
            slice = list(range(self.nSlices))
        try:
            iterator = iter(slice)
        except TypeError:
            slice = [slice]


        # 2d
        S0mean_ = (vol[0,0,:,:]+vol[0,1,:,:])/2

        T10_ = T10Image.sitk_ndarray
        print (T10_.shape)
        R10_ = 1.0/T10_

        # for now we do it slice by slice
        from tqdm import tqdm_notebook

        R1_ = np.zeros( (vol.shape))
        T1_ = np.zeros( (R1_.shape))
        for s in tqdm_notebook(slice):
            curR10Slice = R10_ [s, :, :]
            R1_ = self.dce_to_r1eff (vol[:, s, :, :], S0mean_, curR10Slice)
            T1_ = 1.0/R1_
        return T1_




    def delme_getVolumeAsArray (self, N):

        #t = self.extractMetaTag ("ContentTime", imageNumber = r)

        # Content Time = 0008,0033 will give information about the order
        # TODO; well, so many pitfalls.

        # correct order and volumes array
        self.log ("Found " + str(len(self.series_file_names)) + " slices overall.")
        order = [0]*len(self.series_file_names)
        contentTime = dict() #np.array([0]*len(order))

        for r in range(len(order)):
            t = self.extractMetaTag ("ContentTime", imageNumber = r)
            contentTime [r] = t
        #print (contentTime)

        print (contentTime.items())
        sortedSlices = sorted(contentTime.items(), key=lambda x: x[1])
        print (sortedSlices)

        # TODO; quick hacks, just split date
        groupedSlices = [s +  tuple(s[1].split(".")) for s in sortedSlices]
        print (groupedSlices)

        groups = list(set([s[2] for s in groupedSlices] ))
        for g in groups:
            gpmembers = [s for s in sorted(groupedSlices) if s[2] == g]
            #print (gpmembers)
            print (g)
            print (len(gpmembers))
        print (groups)
        print (len(groups))
        print (len(sortedSlices))


        slices = list(set([s[1] for s in groupedSlices] ))
        print (len(slices))
