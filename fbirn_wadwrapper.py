#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
#
# Changelog:
#   20190426: Fix for matplotlib>3
#   20170502: renamed pureSNR_series to snr_series
#   20161220: remove class variables; remove testing stuff
#   20161026: added b0mapping and b1mapping tests
#   20161005: initial version, rewrite of BVN matlab code
#
# ./fbirn_wadwrapper.py -c Config/mr_philips_fbirn_b0.json -d TestSet/b0map -r results_b0map.json

from __future__ import print_function

__version__ = '20190426'
__author__ = 'aschilham, bvnierop'

import sys
import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput

if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

try:
    import pydicom as dicom
except ImportError:
    import dicom
import fbirn_lib as qc_lib
import fbirn_info as qc_info
from wad_qc.modulelibs import wadwrapper_lib

def logTag():
    return "[fbirn_wadwrapper] "

# MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

##### Real functions
def snr_series(data, results, action):
    """
    QCMR_UMCU Checks:
      B0 mapping test

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build results output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read images
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], 
                                                                  headers_only=False, logTag=logTag(), 
                                                                  skip_check=True, splitOnPosition=False)

    qclib = qc_lib.fBIRN_QC()
    cs = qc_lib.fBIRN_Struct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.verbose = True # output lot's of comments
    if 'circleradiusfactor' in params:
        qclib.usePartCircle = float(params['circleradiusfactor'])

    ## 2. Run tests
    reportkeyvals = []
    error, namevals = qclib.PureSNR(cs)
    if not error:
        for typ, name, val in namevals:
            reportkeyvals.append( (typ, name, val) )

    if error:
        raise ValueError("{} ERROR! processing error in PureSNR".format(logTag()))

    for typ, name, val in reportkeyvals:
        if typ == 'float':
            results.addFloat(name, val)
        elif typ == 'string':
            results.addString(name, val)
        elif typ == 'object':
            results.addObject(name, val)

def receiver_series(data, results, action):
    """
    QCMR_UMCU Checks:
      B0 mapping test

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build results output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read images
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], 
                                                                  headers_only=False, logTag=logTag(), 
                                                                  skip_check=True, splitOnPosition=False)

    qclib = qc_lib.fBIRN_QC()
    cs = qc_lib.fBIRN_Struct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.verbose = True # output lot's of comments
    if 'circleradiusfactor' in params:
        qclib.usePartCircle = float(params['circleradiusfactor'])

    ## 2. Run tests
    reportkeyvals = []
    error, namevals = qclib.B0Mapping(cs)
    if not error:
        for typ, name, val in namevals:
            reportkeyvals.append( (typ, name, val) )

    if error:
        raise ValueError("{} ERROR! processing error in B0Mapping".format(logTag()))

    for typ, name, val in reportkeyvals:
        if typ == 'float':
            results.addFloat(name, val)
        elif typ == 'string':
            results.addString(name, val)
        elif typ == 'object':
            results.addObject(name, val)


def b0_series(data, results, action):
    """
    QCMR_UMCU Checks:
      B0 mapping test

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build results output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read images
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], 
                                                                  headers_only=False, logTag=logTag(), 
                                                                  skip_check=True, splitOnPosition=False)

    qclib = qc_lib.fBIRN_QC()
    cs = qc_lib.fBIRN_Struct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.verbose = True # output lot's of comments
    if 'circleradiusfactor' in params:
        qclib.usePartCircle = float(params['circleradiusfactor'])

    ## 2. Run tests
    reportkeyvals = []
    error, namevals = qclib.B0Mapping(cs)
    if not error:
        for typ, name, val in namevals:
            reportkeyvals.append( (typ, name, val) )

    if error:
        raise ValueError("{} ERROR! processing error in B0Mapping".format(logTag()))

    for typ, name, val in reportkeyvals:
        if typ == 'float':
            results.addFloat(name, val)
        elif typ == 'string':
            results.addString(name, val)
        elif typ == 'object':
            results.addObject(name, val)

def b1_series(data, results, action):
    """
    QCMR_UMCU Checks:
      B1 mapping test

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build results output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read images
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], 
                                                                  headers_only=False, logTag=logTag(), 
                                                                  skip_check=True, splitOnPosition=False)

    qclib = qc_lib.fBIRN_QC()
    cs = qc_lib.fBIRN_Struct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.verbose = True # output lot's of comments
    if 'circleradiusfactor' in params:
        qclib.usePartCircle = float(params['circleradiusfactor'])

    ## 2. Run tests
    reportkeyvals = []
    error, namevals = qclib.B1Mapping(cs)
    if not error:
        for typ, name, val in namevals:
            reportkeyvals.append( (typ, name, val) )

    if error:
        raise ValueError("{} ERROR! processing error in B1Mapping".format(logTag()))

    for typ, name, val in reportkeyvals:
        if typ == 'float':
            results.addFloat(name, val)
        elif typ == 'string':
            results.addString(name, val)
        elif typ == 'object':
            results.addObject(name, val)

def qc_series(data, results, action):
    """
    QCMR_UMCU Checks: Philips fBIRN analysis reimplemented in python
      Weisskoff analysis
      Ghosting
      Residual Noise
      Temporal Fluctuation
      Spatial Noise

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build results output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read images
    dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], 
                                                                  headers_only=False, logTag=logTag(), 
                                                                  skip_check=True, splitOnPosition=False)

    qclib = qc_lib.fBIRN_QC()
    cs = qc_lib.fBIRN_Struct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.verbose = True # output lot's of comments

    ## 2. Run tests
    reportkeyvals = []
    error, namevals = qclib.QC(cs)
    if not error:
        for typ, name, val in namevals:
            reportkeyvals.append( (typ, name, val) )

    if error:
        raise ValueError("{} ERROR! processing error in QC".format(logTag()))

    for typ, name, val in reportkeyvals:
        if typ == 'float':
            results.addFloat(name, val)
        elif typ == 'string':
            results.addString(name, val)
        elif typ == 'object':
            results.addObject(name, val)

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 

def header_series(data, results, action):
    """
    Read selected dicomfields and write to IQC database

    Workflow:
        1. Run tests
        2. Build results output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    dcmInfile, pixeldataIn, dicomMode = wadwrapper_lib.prepareInput([data.series_filelist[0][0]], headers_only=True, logTag=logTag())
    cs = qc_lib.fBIRN_Struct(dcmInfile=dcmInfile, pixeldataIn=pixeldataIn, dicomMode=dicomMode)
    cs.verbose = False
    
    ## 1. run tests
    reportkeyvals = []
    dicominfo = qc_info.DICOMInfo(cs, 'dicom', 0)
    if len(dicominfo) >0:
        for di in dicominfo:
            reportkeyvals.append( (di[0],str(di[1])) )

    ## 2. Build xml/json output
    varname = 'pluginversion'
    results.addString(varname, str(qc_lib.__version__)) 

    for key,val in reportkeyvals:
        val2 = "".join([x if ord(x) < 128 else '?' for x in val]) #ignore non-ascii 
        results.addString(key, str(val2)[:min(len(str(val)),100)]) 


if __name__ == "__main__":
    data, results, config = pyWADinput()
    
    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'header_series':
            header_series(data, results, action)
        
        elif name == 'qc_series': # SeriesDescription: 'SS nosense 150FOV'
            qc_series(data, results, action)

        elif name == 'B1_series': # Series Description: 'B1map right', 'B1map left', 'B1map'
            b1_series(data, results, action)
        elif name == 'B0_series':
            b0_series(data, results, action)

        elif name == 'receiver_series':
            receiver_series(data, results, action) #3DFFE_FA1
        elif name == 'snr_series': # Series Description: '3DFFE SENSE 1 noGRRF   pause'
            snr_series(data, results, action)

    #results.limits["minlowhighmax"]["mydynamicresult"] = [1,2,3,4]

    results.write()
