
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



class DICOM (object):
    def __init__ (self):
        self.debugLevel = 5
        self.sitk_ndarray = None
        pass


    def copyFrom (self, other):
        self.debugLevel = other.debugLevel
        self.sitk_image = other.sitk_image
        self.image_meta = other.image_meta
        self.sitk_ndarray = other.sitk_ndarray
        self.series_file_names = other.series_file_names
        return


    def shape (self):
        return self.sitk_ndarray.shape


    def loadFromDirectory (self, dicom_directory):
        sitkreader = sitk.ImageSeriesReader()

        # check if there is a DICOM series in the dicom_directory
        series_IDs = sitkreader.GetGDCMSeriesIDs(dicom_directory)
        if not series_IDs:
            print("ERROR: given directory \""+dicom_directory+"\" does not contain a DICOM series.")
            return (-1)

        series_file_names = sitkreader.GetGDCMSeriesFileNames(dicom_directory, series_IDs[0])
        sitkreader.SetFileNames(series_file_names)
        self.sitk_image = sitkreader.Execute()

        sitk_ndarray = sitk.GetArrayFromImage(self.sitk_image)
        z,y,x = sitk_ndarray.shape
        origin = np.array(self.sitk_image.GetOrigin())
        spacing = np.array(self.sitk_image.GetSpacing())

        # get Metadata and save as image_meta
        image_reader = sitk.ImageFileReader()
        image_reader.LoadPrivateTagsOn()
        image_meta = []
        for file_name in series_file_names:
            image_reader.SetFileName(file_name)
            img = image_reader.Execute()
            image_meta.append(img)
        self.sitk_ndarray = sitk_ndarray
        self.image_meta = image_meta
        self.series_file_names = series_file_names
        return 0


    def log (self, s, l = 5):
        if self.debugLevel >= l:
            print (s)

    def error (self, s):
        print ("ERROR:", s)
        exit(-1)


    # TODO: rename
    def extractMetaTag (self, tag, imageNumber = None):
        # mmh.
        if tag == "RepetitionTime":
            tag = "0018|0080"
        if tag == "FlipAngle":
            tag = "0018|1314"
        if tag == "ContentTime":
            tag = "0008|0033"
        if tag == "AcquisitionNumber":
            tag = "0020|0012"
        if tag == "ImageTime": # same as contenttime!
            tag = "0008|0033"
        if tag == "AcquisitionTime":  # nonsense for ordering, as it is the same for all slices
            tag = "0008|0032"
        if tag == "AcquisitionDate":  # nonsense for ordering, as it is the same for all slices
            tag = "0008|0022"
        if tag == "ContentDate":
            tag = "0008|0023"

        # check
        if imageNumber < 0 or imageNumber >= len(self.image_meta):
            self.error ("No image with number " + image + " can be found.")

        v = self.image_meta[imageNumber].GetMetaData(tag)
        return v



    # TODO: anonymize flag
    def loadFromOrthanc (self, o, seriesID, saveDir = None):
        # get temp directory
        if saveDir == None:
            downloadDir = tempfile.mkdtemp()
        else:
            downloadDir = saveDir

        self.log ("Downloading series " + str(seriesID) + " to directory " + str(downloadDir))
        o.downloadSeries (seriesID, downloadDir, True)

        self.log ("Loading series from directory " + str(downloadDir))
        self.loadFromDirectory (downloadDir)

        # if the directory was temporarily set by us, we remove it
        if saveDir == None:
            shutil.rmtree(downloadDir)



    def getAsArray (self):
        return self.sitk_ndarray


    def setFromArray(self):
        pass



    def coregisterWith (self):
        # other dicoms


        # load mask
        print ("Loading mask", maskId)
        mask = o.getSegmentationAsItk(maskId, anonymized = False)




        # resample mask to series
        rif = sitk.ResampleImageFilter()
        rif.SetReferenceImage(series)
        rif.SetOutputPixelType(mask.image.GetPixelID())
        rif.SetInterpolator(sitk.sitkNearestNeighbor)
        resMask = rif.Execute(mask.image)





    def saveToDirectory (self, destDir, seriesUID, seriesDescription, createFolder = True):
        self.log ("Saving DICOM with SeriesUID " + seriesUID + " to directory " + destDir)
        # make sure the directory exists
        if not os.path.exists(destDir):
            os.makedirs(destDir)

        # put DICOMs into subfolder names after seriesUID?
        if createFolder == True:
            destDir = os.path.join(destDir, seriesUID)
            if not os.path.exists(destDir):
                os.makedirs(destDir)

        # create writer
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        sitk_image = sitk.GetImageFromArray(self.sitk_ndarray)
        for i in range(sitk_image.GetDepth()):
            image_slice = sitk_image[:,:,i]
            original_slice = self.image_meta[i]
            # Copy the meta-data except the rescale-intercept, rescale-slope
            for k in original_slice.GetMetaDataKeys():
                if k!="0028|1052" and k!= "0028|1053":
                    image_slice.SetMetaData(k, original_slice.GetMetaData(k))
            # Set relevant keys indicating the change, modify or remove private tags as needed
            image_slice.SetMetaData("0008|0031", modification_time)
            image_slice.SetMetaData("0008|0021", modification_date)
        #  image_slice.SetMetaData("0008|0008", "DERIVED\SECONDARY")
            image_slice.SetMetaData("0008|103E", seriesDescription)
            # Anonymise PatientData
            image_slice.SetMetaData("0010|0010", seriesUID) # Patientname
            image_slice.SetMetaData("0010|0020", seriesUID) # PatientID
            image_slice.SetMetaData("0010|0030", "20000101") #Birthdate
            image_slice.SetMetaData("0010|1040", seriesUID) #Adress
            # Each of the UID components is a number (cannot start with zero) and separated by a '.'
            # We create a unique series ID using the date and time.
            image_slice.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time)
            # Write to the output directory and add the extension dcm if not there, to force writing is in DICOM format.
            writer.SetFileName(os.path.join(destDir, os.path.basename(self.series_file_names[i])) + ('' if os.path.splitext(self.series_file_names[i])[1] == '.dcm' else '.dcm'))
            writer.Execute(image_slice)
        return 0
