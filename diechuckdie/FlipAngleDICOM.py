

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
from numpy import sin, cos, tan, exp
from tqdm import tqdm_notebook


from DICOM import DICOM

class FlipAngleDICOM (object):
   def __init__ (self):
       self.members = []
       self.TR = None
       self.flipAngles = [] # always in RAD
       self.debugLevel = 5
       pass


   def info (self):
       print ("Flip Angle DICOM has " + str(len(self.members)) + " members.")
       print ("TR: " + str (self.TR))
       print ("Flip Angles: ")
       for f in self.flipAngles:
           print ("\t" + str(f))



   def checkNewMember (self, n):
       self.log ("Checking new member.")
       if len(self.members) != 0:
           # extract FlipAngle
           newFlipAngle = n.extractMetaTag ("FlipAngle", imageNumber = 0)
           # TODO: anything to do with it?

           # extract TR from new member
           newTR = n.extractMetaTag ("RepetitionTime", imageNumber = 0)
           if self.TR == None:
               self.error ("Members exist, but no TR.")
               return (False)
           if float(newTR) != self.TR:
               # FIXME: for now we just ignore
               self.log ("WARNING: New member has a different TR than the other mambers.\n" +
                          "Old: " + str(self.TR) + ", New: " + str(float(newTR)))
               return (True)

       return True


   # FIXME: fix logging
   def log (self, s, l = 5):
       if self.debugLevel >= l:
           print (s)

   def error (self, s):
       print ("ERROR:", s)
       exit(-1)



   def addNewMember (self, m):
       # extract FlipAngle
       flipAngle = m.extractMetaTag ("FlipAngle", imageNumber = 0)
       self.log ("Extracted flip angle " + str(flipAngle))
       self.flipAngles.append (np.pi/180.0*float(flipAngle))

       newTR = m.extractMetaTag ("RepetitionTime",  imageNumber = 0)
       self.log ("Extracted RepetitionTime " + str(newTR))
       if len(self.members) == 0:
           # if its empty we need to extract the TR
           self.TR = float(newTR)

       self.members.append (m)



   def addDICOMFromDirectory (self, dirname):
       n = DICOM()
       n.loadFromDirectory (dirname)
       if self.checkNewMember (n):
           self.addNewMember (n) # FIXME: ? .copy())
       return (True)



   def addAllDICOMFromDirectory (self, dirname):
       # glob all subdirectories or have a list of IDs
       idList = []
       for uid in idList:
           d = DICOM()
           d.getFromDirectory (uid)
           if self.checkNewMember (n):
               self.members.append (n)


   def addDICOMFromOrthanc (self, o, uuid, saveDir = None):
       n = DICOM()

       if saveDir != None:
           saveDir = os.path.join (saveDir, uuid)

       n.loadFromOrthanc (o, uuid, saveDir = saveDir)
       if self.checkNewMember (n):
           self.addNewMember (n) # FIXME: ? .copy())
       return (True)




   def setTR (self, TR):
       self.TR = TR
       pass



   def t1_signal_eqn (self, x, R10, M0):
       E10 = exp(-self.TR*R10)
       return M0*sin(x)*(1.0 - E10) / (1.0 - E10*cos(x))



   def getCoeff (self, method, t1_signal_eqn, flipAngles, y):
       if method == "curve_fit":
           try:
               popt, _ = curve_fit (t1_signal_eqn, flipAngles, y)
           except RuntimeError:
               popt = [0, 0]
           return popt
       if method == "levenberg":
           popt = root (t1_signal_eqn_root, [1, 1, 1], jac=True, method='lm')
           return popt
       if method == "naive":
           #print (y)
           xi = y/tan(flipAngles)
           yi = y/sin(flipAngles)

           popt = np.polyfit(xi, yi, 1) # returns m, b in that order
           #print(popt)
           return popt



   def fit_R10 (self, flipImages, flipAngles, TR, minMean = 0.1, method = "curve_fit"):
       inshape = flipImages.shape
       nangles = inshape[-1]

       n = np.prod(inshape[:-1])
       images = np.reshape(flipImages.copy(), (n, nangles))
       signal_scale = abs(images).max()
       images = images/signal_scale
       # TODO: check if greater than pi, if so, we have degs
       #flip_angles = pi*arange(20,0,-2)/180.0  # deg

       assert(nangles == len(flipAngles))

       R10map = np.zeros(n)
       S0map = np.zeros(n)

       for j in tqdm(range(n)):
           if minMean is not None:
               if images[j,:].mean() > minMean:
                   params = self.getCoeff (method, self.t1_signal_eqn, flipAngles, images[j,:].copy())
               else:
                   params = (0,0)
           else:
               params = self.getCoeff (method, t1_signal_eqn, flipAngles, images[j,:].copy())
           R10map[j] = params[0]
           S0map[j] = params[1]


       R10_ = np.reshape(R10map, inshape[:-1]) #*signal_scale
       S0_ = np.reshape(S0map, inshape[:-1] )
       return (R10_, S0_)




   def computeT10Image (self, slice = None, minMean = 0.05):
       # first make sure that the members all do fit somehow
       # so co-register them to get the smallest volume

       # work on copies.

       if len(self.members) == 0:
           error ("No flip angles added.")

       # create an numpy array from the member copies
       data = self.members[0].getAsArray ()
       volumes = np.zeros ( data.shape + (len(self.members),), dtype = np.float32 )
       volumes [:, :, :, 0] = data

       for m in range(1, len(self.members)):
           data = self.members[m].getAsArray ()
           volumes [:, :, :, m] = data

       # T10 Data has the same shape, but only one volume
       T10Data = np.zeros ( volumes.shape[:-1], dtype = np.float32 )

       # for each slice
       nSlices = volumes.shape[0]
       self.log ("Volume shape: " + str (volumes.shape))
       self.log ("Number of slices: " + str (nSlices))

       #nSlices = 3

       def computeSlice (s):
           timeVolForSlice = volumes[s, :, :, :]  # put the slice from all members together
           R10_, S0_ = self.fit_R10(timeVolForSlice, self.flipAngles, self.TR, minMean = minMean)

           zeroStrategy = "eps"
           if zeroStrategy == "set":
               T10_ = 1/R10_
               T10_[T10_ < 0] = 0

           if zeroStrategy == "eps":
               R10_[R10_ < 0.000001] = 0.000001
               T10_ = 1/R10_
               T10_[T10_ >= 1000000] = np.NaN
               T10_[T10_ < 0] = 0

           T10Data [s, :, :] = T10_

       if slice is None:
           slice = list(range(nSlices))
       try:
           iterator = iter(slice)
       except TypeError:
           slice = [slice]

       self.log("Computing slices " + str(list(slice)))
       [computeSlice(i) for i in tqdm(slice)]



       # create a DICOM image from the T10 data

       # as we have the same volume as any of the flip images
       # FIXME: make sure that this is the case
       # we can just take any member to modify it
       T10DICOM = DICOM()
       T10DICOM.copyFrom (self.members[0])
       T10DICOM.sitk_ndarray = T10Data
       return T10DICOM
