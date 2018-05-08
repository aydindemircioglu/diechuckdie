"""diechuckdie package -- easily handle DICOM files.
Copyright (c)

This file is part of pydicom, released under a modified MIT license.
   See the file LICENSE included with this distribution, also
   available at https://github.com/pydicom/pydicom
"""


from diechuckdie.DICOM import DICOM
from diechuckdie.FlipAngleDICOM import FlipAngleDICOM
from diechuckdie.KMDynamics import KMDynamics
from diechuckdie.Model import Model

from ._version import __version__, __version_info__


__all__ = ['DICOM',
            'FlipAngleDICOM',
            'KMDynamics',
            'Model',
            '__version__',
            '__version_info__']
