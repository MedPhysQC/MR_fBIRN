# -*- coding: utf-8 -*-
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
from __future__ import print_function

"""
Changelog:
   20161220: remove testing stuff
   20161005: initial version
"""

__version__ = '20161220'
__author__ = 'aschilham, bvnierop'

# First try if we are running wad1.0, since in wad2 libs are installed system-wide
try: 
    # try local folder
    import wadwrapper_lib
except ImportError:
    # try pyWADlib from plugin.py.zip
    try: 
        from pyWADLib import wadwrapper_lib

    except ImportError: 
        # wad1.0 solutions failed, try wad2.0 from system package wad_qc
        from wad_qc.modulelibs import wadwrapper_lib


def DICOMInfo(cs, info='dicom',imslice=0):
    """
    Return some characteristic information from the dicom headers
    """
    if info == "dicom":
        dicomfields = [
            ["0010,0010", "Patients Name"], # PIQT
            ["0018,1030", "Protocol Name"],  # QA1S:MS,SE
            ["0008,0021", "Series Date"],
            ["0008,0031", "Series Time"],# no ScanTime 0008,0032 in EnhancedDicom
            ["0018,1250", "Receive Coil Name"], # Q-Body
            ["0018,1251", "Transmit Coil Name"], # B
            ["0018,0095", "Pixel Bandwidth"], # 219
            ["0018,0020", "Scanning Sequence"], # SE
            ["0018,0021", "Scanning Variant"], # SS
            ["2005,1011", "Image_Type"], # M
            ["0018,0081", "Echo Time"], # 50
            ["0020,0012", "Acquisition Number"], # 5
            ["0018,0086", "Echo Number(s)"], # 1
            ["2001,1081", "Dyn_Scan_No"], # ?1
            ["0020,0013", "Instance Number"], # 1 slice no?
            ["2001,105f,2005,1079", "Dist_sel"], # -16.32
            ["2001,1083", "Central_freq"], # 63.895241 (MHz)
            ["0018,1020", "SoftwareVersions"], 
        ]

    results = []
    for df in dicomfields:
        key = df[0]
        value = wadwrapper_lib.readDICOMtag(key, cs.dcmInfile, imslice)
        if key=="0018,1020" and len(value)>1:
            value = '_'.join(value)
        results.append( (df[1],value) )

    return results

