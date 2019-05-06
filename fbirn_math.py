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
Warning: THIS MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED! And make sure rescaling is corrected!

Changelog:
   20161005: initial version, rewrite of BVN matlab code
"""

__version__ = '20161005'
__author__ = 'aschilham, bvnierop'

try:
    import pydicom as dicom
except ImportError:
    import dicom
import numpy as np
import datetime as dt
import scipy.ndimage as scind
import matplotlib.pyplot as plt
from skimage.feature import register_translation
from skimage import transform
from skimage.measure import regionprops
import time

def RegisterImages(imIn):
    """
    Use imreg_dft to translate and rotate all images in stack imIn to imIn[0]
    Matlab uses bilinear interpolation, so order=1
    """
    #model = 'none'
    model = 'skimage'
    t0 = time.time()
    if model == 'none':
        # note: does nothing.
        # the TEMPLATE
        im0 = imIn[0].astype(float)
        result = [imIn[0].astype(float)]
        #
        for x,im in enumerate(imIn[1:]):
            result.append(im.astype(float))

    elif model == 'skimage':
        # note: does work well, needs a recent scikit-image installation, handles subpixel precision well, no rotation, very fast!
        # subpixel precision
        # the TEMPLATE
        im0 = imIn[0].astype(float)
        result = [imIn[0].astype(float)]
        with open('fbirn_dump_%s.tsv'%model,'w') as f:
            f.write('id\tshifty\tshiftx\terror\tphasediff\n')
            for x,im in enumerate(imIn[1:]):
                # print(x)
                shift, error, diffphase = register_translation(im0, im, 100) # accurate to 1/100th px
                tform = transform.SimilarityTransform(translation=[-shift[1],-shift[0]])#shift) shift needs to be (x, y)
                f.write('%d\t%.12f\t%.12f\t%.12f\t%.12f\n'%(x,shift[0],shift[1], error, diffphase))
                result.append(transform.warp(im, tform, output_shape=im0.shape, preserve_range=True))
    print('%s registering %d images in %fs'%(model, len(imIn)-1, time.time()-t0))
    return np.array(result)


def TimeVector(infolist):
    # Read acqtime tags, return list of times in seconds wrt to frame 0 (~ 0,1*TR,2*TR,...)
    times = []
    for info in infolist:
        dc      = info.AcquisitionDate 
        dc      = info.AcquisitionTime 
        timestring = info.AcquisitionTime 
        if '.' in timestring:
            t = dt.datetime.strptime(timestring, '%H%M%S.%f') #HHMMSS.FFFFFF
        else:
            t = dt.datetime.strptime(timestring, '%H%M%S') #HHMMSS.FFFFFF
        times.append( 60*(60*t.hour+t.minute)+t.second+t.microsecond/1.e6 )
        
    # time in seconds since scan start
    times = [ (t - np.min(times), i) for i,t in enumerate(times) ] # scantime in seconds (~ 0,1*TR,2*TR,...)
    
    # sort times and return indices
    times = sorted(times)
    return [t[0] for t in times], [t[1] for t in times]

def DefineHalfCircROI(mask_in, phaseEncDir, pixsizemm):
    """
    Input: mask_in: mask of sphere based on for example a thresholding
           phaseEncDir: options 'ROW', 'COL'   
    Output: mask_phase,mask_snr: mask shifted by half size im in phaseEncDir
    Note: pyqtgraph data format used, so directions of rows and colums completely opposite of expectation
    """

    # Just increase the mask to not include the phantom in measurement ROIs
    # spacing of 1 cm
    rad = int(10/pixsizemm)
    im_mask = scind.morphology.binary_dilation(mask_in, iterations=rad).astype(int)

    pixx,pixy = np.shape(mask_in)

    mask_out_col = (np.roll(im_mask, int(pixx/2.+.5), axis=0)-im_mask) > 0 # pyqtgraph = [x,y], not [y,x]
    mask_out_row = (np.roll(im_mask, int(pixy/2.+.5), axis=1)-im_mask) > 0

    # Ghost artefacts propagate in the phase enc directions, so if phaseEncDir
    # == 'COL' then we need to circshift the mask of our phantom in the
    # direction perpendicular to 'catch' possible ghosts
    if phaseEncDir == 'ROW':
        mask_phase = mask_out_col
        mask_snr   = mask_out_row
        line = list(np.sum(im_mask,axis=0)>0)
        left  = line.index(1)
        right = len(line)-1-list(reversed(line)).index(1)
        mask_snr[:, left:right+1] = 0


    elif phaseEncDir == 'COL':
        mask_snr   = mask_out_col
        mask_phase = mask_out_row
        line = list(np.sum(im_mask,axis=1)>0)
        left  = line.index(1)
        right = len(line)-1-list(reversed(line)).index(1)
        mask_snr[left:right+1, :] = 0

    if 1<0:
        plt.figure()
        plt.imshow(mask_in)
        plt.imshow(im_mask, alpha=.5)
        #plt.imshow(mask_out_col, alpha=.5)
        #plt.imshow(mask_out_row, alpha=.5)
        plt.imshow(mask_snr, alpha=.5)
        plt.show()

    return mask_phase, mask_snr

def DefineGhostBoxROI(mask_in, phaseEncDir, pixsizemm):
    """
    Make ROI boxes like Philips
    Input: mask_in: mask of sphere based on for example a thresholding
           phaseEncDir: options 'ROW', 'COL'   
    Output: mask_phase,mask_snr: boxes on edges around mask with same extend as mask; more like Philips
    Note: pyqtgraph data format used, so directions of rows and colums completely opposite of expectation
    """

    # Just increase the mask to not include the phantom in measurement ROIs
    # spacing of 8mm
    rad = int(8/pixsizemm)
    im_mask = scind.morphology.binary_dilation(mask_in, iterations=rad).astype(int)

    pixx,pixy = np.shape(mask_in)

    line = list(np.sum(im_mask,axis=0)>0)
    mask_top  = line.index(1)
    mask_bot = len(line)-1-list(reversed(line)).index(1)
    line = list(np.sum(im_mask,axis=1)>0)
    mask_left  = line.index(1)
    mask_right = len(line)-1-list(reversed(line)).index(1)
    
    mask_out_col = np.zeros(np.shape(mask_in), dtype=np.bool)
    mask_out_row = np.zeros(np.shape(mask_in), dtype=np.bool)

    mask_out_col[1:mask_left,     1+mask_top+1:mask_bot] = 1
    mask_out_col[1+mask_right:-1, 1+mask_top+1:mask_bot] = 1
    mask_out_row[1+mask_left:mask_right,  1:mask_top]    = 1
    mask_out_row[1+mask_left:mask_right,  1+mask_bot:-1] = 1
    
    # Ghost artefacts propagate in the phase enc directions, so if phaseEncDir
    # == 'COL' then we need to circshift the mask of our phantom in the
    # direction perpendicular to 'catch' possible ghosts
    if phaseEncDir == 'ROW':
        mask_phase = mask_out_col
        mask_snr   = mask_out_row
    elif phaseEncDir == 'COL':
        mask_snr   = mask_out_col
        mask_phase = mask_out_row

    return mask_phase, mask_snr

def detrend(im_in, detrending_vector):
    return [ im-d for im,d in zip(im_in, detrending_vector)]

def CircleImages(images, usePartofRadius, circles=None):
    """
    Threshold phantom images (here any non-zero values) and fit a circle around pixels found.
    Multiply found radius by arbitrary usePartofRadius to make sure to stay within
    bounds of phantom.
    Returns a list of images of circle masks.
    """
    im_circ = []
    pixx,pixy = np.shape(images[0])
    for im in images:
        mask = (im>0).astype(int)

        stats = regionprops(mask, coordinates='xy')[0] # ask for properties of first label (there is only one)
        cy,cx = stats.centroid # centroid of center of mass
        minaxislength = stats.minor_axis_length # smallest diam within mask
        # Define circle within phantom based on radius determine based on
        # largest square
        x,y = np.meshgrid( np.arange(-cx, pixx-cx), np.arange(-cy, pixy-cy) )
        r   = minaxislength/2.*usePartofRadius
        im_circ.append( np.zeros(np.shape(im), dtype=np.bool) )
        im_circ[-1][(x*x+y*y)<=r*r] = True
        if not circles is None:
            circles.append((cy+.5, cx+.5, r))
        
    return im_circ
    
