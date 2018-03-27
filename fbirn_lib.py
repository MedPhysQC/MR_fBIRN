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

% MOTIVATION
% BOLD effects of interest are small, so temporal stability during functional acquisition 
% is important. In order to accurately measure such small signal changes, an MR system must 
% have intrinsic image time series fluctuation levels much lower than these expected signal 
% changes. Quality checks allow you to assess if your data are worth being analyzed.

% usefull information
% http://sfnic.ucsf.edu/stability.html
% http://wagerlab.colorado.edu/wiki/doku.php/help/fmri_quality_control_overview
% Philips UserManual_QA_Tool

Differences between UMCU version and Philips version:
  o Philips: no drift correction of images (speed!); square Roi 1 smaller than possible to allow drifting of 1 px
    UMCU: first apply drift correction by registration of translation only (no scaling, shear, rotatation)
  o Philips: Ghost ROIs: use boxes of same extend as phantom mask
  o Philips: Residual Noise: store of max magnitude: FFT_magnitude/SV_mean
    UMCU: store freq of max amplitude, store amplitude. Philips = Num_frames*Amplitude/SV_mean
  o Philips: Temporal Noise:
     SFNR detrending over time per pixel (very slow! but now OK when restricted to box only)
    UMCU: Analyse only max roi box, detrend once for all. Validated that additional detrending has very little effect.

Changelog:
   20161220: remove class variables
   20161026: Added tests B0mapping and B1mapping
   20161025: Choices made:
               detrend Weisskoff for each roi size separately; 
               use noise corrected ghost;
               calculate ghost/snr on UNregistered images to avoid structures introduced by registration
               use theorectical weisskoff line as 1/n*meas[0] (just like others do)
   20161021: added Philips changes/runmode
   20161005: initial version, rewrite of BVN matlab code
"""

__version__ = '20161220'
__author__ = 'aschilham, bvnierop'

try:
    import pydicom as dicom
except ImportError:
    import dicom

import numpy as np
import scipy.ndimage as scind
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import fbirn_math as mmath
import time

class fBIRN_Struct:
    """
    class to hold all data and results together, to pass between functions
    """
    
    def __init__ (self, dcmInfile, pixeldataIn, dicomMode):
        # input image
        self.dcmInfile = dcmInfile
        self.pixeldataIn = pixeldataIn
        self.dicomMode = dicomMode

        self.verbose = False

        # for matlib plotting
        self.hasmadeplots = False

        self.guistuff = {} # container for gui specific stuff
    

# Define 2nd order polynomial function
def poly2(xdata, x0,x1,x2):
    """
    input: vector xdata, x
    output: vector x
    """
    return [ x0*xd*xd + x1*xd + x2 for xd in xdata ]


class fBIRN_QC:
    Rayleighcorr = 1.5267

    def __init__ (self):
        self.qcversion = __version__
        self.CALCMODE = 'Philips' # not to be selected, just for comparison
        self.CALCMODE = 'UMCU'
        self.weisskoff_theory_n = True # Philips (and others): Weisskoff to show deviation of 1/n, 
                                    #  so theoretical line is meas(1px roiwidth)/roiwidth*meas[0]

        self.usePartCircle = 0.85 # scaling factor, to make sure circle stays within phantom mask
        self.results = []  # container for all results
        self.guistuff = {} # container for gui specific stuff
        self.guimode = False # when set to True, copies of images will be made available for gui
        
    def Weisskoff(self, im_reg, halfrib, com, SV_SNR_regular):
        # Calculate the Weisskoff plot, according to Weisskoff 1996
        halfribvar   = range(halfrib+1)
        ROIlength    = [2*h+1 for h in halfribvar]

        # SV_SNR_regular is defined acc to eq. 1 in Weisskoff 1996
        Fnt          = [1./(r*SV_SNR_regular) for r in ROIlength ] #  Eq. 4 Weisskoff 1996
        
        # calculate Fn eq. 2 Weisskoff 1996
        im_x1y1 = np.zeros(np.shape(im_reg[0]), dtype=np.bool)
        pixz = len(im_reg)
        mnbar = []
        Fn = []

        dyntime = np.arange(len(im_reg), dtype=np.float) # for detrending only
        
        for n in range(len(ROIlength)):
            # start with ROI of 1x1 pixel, thus COM
            x1  = [(com[0]-halfribvar[n]),(com[0]+halfribvar[n])+1] # +1 to include upper limit
            y1  = [(com[1]-halfribvar[n]),(com[1]+halfribvar[n])+1]
            im_x1y1[x1[0]:x1[1], y1[0]:y1[1]] = True

            Mi_n  = [ np.ma.array(im, mask=~im_x1y1).mean() for im in im_reg ] # Eq. 3 Weisskoff 1996 
            mnbar = np.mean(Mi_n)                                       # Eq. 3 Weisskoff 1996 

            # Detrend with Fit raw signal curve to 2nd order poly
            x0 = [3e-5, 3e-3, np.max(Mi_n)] # start cond for lsqcurvefit
            x, pcov = curve_fit(poly2, dyntime, Mi_n, p0=x0)
            detrendedSignal = mmath.detrend(Mi_n, poly2(dyntime, *x))
            Fn.append( np.std(detrendedSignal)/mnbar ) # Eq. 2 Weisskoff 1996

            im_x1y1[x1[0]:x1[1], y1[0]:y1[1]] = False # reset mask

        if self.weisskoff_theory_n:
            Fnt = Fn[0]/Fnt[0]*np.array(Fnt)
        self.makePlots('Weisskoff', 
                       {'ROIlength': ROIlength, 'Fn':Fn, 'Fnt':Fnt})
        return Fn

    def StaticSpatialNoise(self, im_reg):
        """
        Static Spatial Noise Image
        The Static Spatial Noise Image (Even - Odd Image) is the difference, voxel by 
        voxel, between the sum over all the even-numbered images and the sum over all 
        the odd-numbered images. If the images in the time-series exhibit no drift in 
        amplitude or geometry, this image will display no structure from the phantom, 
        and the variance in this image will be a measure of the intrinsic noise.
        """
        
        pixz = len(im_reg)
        # get even and odd acquisition numbers for calc Static Spatial Noise Image 
        # to get same results as matlab (starts counting at 1), here do odd-even!
        im_SSNI = np.sum(im_reg[range(1,pixz,2)],axis=0)-np.sum(im_reg[range(0,pixz,2)],axis=0)


        self.makePlots('StaticSpatialNoise', 
                       {'im_SSNI': im_SSNI})

        # for gui only
        if self.guimode:
            self.guistuff['im_SSNI'] = np.array(im_SSNI)

        return im_SSNI
        
    def GhostSNR(self, info, im_reg, im_mask, im_mean, im_square, dynscantime, rawsignalcurve):
        # Calculate the raw signal ghost graph
        # im_mean, im_square, dynscantime, rawsignalcurve just for plot
        
        # In-plane Phase Encoding Direction: The axis of phase encoding with respect to 
        # the image. Enumerated Values: ROW = phase encoded in rows. COL = phase encoded 
        # in columns. Determines in which dir a ghost will propagate through the image
        Phdir     = info[0].InPlanePhaseEncodingDirection # (0018,1312)	
        pixsizemm = np.float(info[0].PixelSpacing[0])

        # define rois for analysis SNR and ghosting
        if self.CALCMODE == 'Philips':
            mask_phase, mask_snr = mmath.DefineGhostBoxROI(im_mask, Phdir, pixsizemm)
        else:
            mask_phase, mask_snr = mmath.DefineHalfCircROI(im_mask, Phdir, pixsizemm)
            
        # note: masked array ignores coordinates with mask == 1 (instead of selects)
        rawsignalcurve_ghost     = [ np.ma.array(im, mask=~mask_phase).mean() for im in im_reg ]
        rawsignalcurve_ghost_sd  = [ np.ma.array(im, mask=~mask_phase).std() for im in im_reg ]
        rawsignalcurve_snr       = [ np.ma.array(im, mask=~mask_snr).mean() for im in im_reg ]
        rawsignalcurve_snr_sd    = [ np.ma.array(im, mask=~mask_snr).std() for im in im_reg ]
        
        rawsignalcurve_snr_snr = [rs/sd/self.Rayleighcorr for rs,sd in zip(rawsignalcurve,rawsignalcurve_snr_sd) ]
        self.makePlots('GhostSNR', 
                       {'im_mean': im_mean, 'im_square': im_square, 'mask_phase': mask_phase, 'mask_snr':mask_snr,
                        'dynscantime': dynscantime, 
                        'rawsignalcurve_ghost':rawsignalcurve_ghost, 'rawsignalcurve_ghost_sd': rawsignalcurve_ghost_sd,
                        'rawsignalcurve_snr': rawsignalcurve_snr, 'rawsignalcurve_snr_sd': rawsignalcurve_snr_sd,
                        'rawsignalcurve_snr_snr': rawsignalcurve_snr_snr
                        })

        # for gui only
        if self.guimode:
            self.guistuff['mask_phase'] = np.array(mask_phase)
            self.guistuff['mask_snr']   = np.array(mask_snr)

        return rawsignalcurve_ghost, rawsignalcurve_ghost_sd, rawsignalcurve_snr, rawsignalcurve_snr_sd, mask_phase


    def ResidualNoise(self, info, rawsignalcurve_TFNI):
        # Calculate the residual noise graph in freq domain
        # http://www.ni.com/white-paper/4278/en/
        # A suitably scaled plot of the complex modulus of a discrete Fourier transform is commonly known as a power spectrum. 
        TR    = info[0].RepetitionTime            # 0018 0080 (ms)
        pixz = len(rawsignalcurve_TFNI)
        
        Fs = 1000./TR                         # Sampling frequency Fs(s-1), TR (ms)
        t = [ tt/Fs for tt in range(pixz-1) ] # Time vector
    
        if self.CALCMODE == 'Philips':
            P1 = np.abs(np.fft.rfft(rawsignalcurve_TFNI))/self.SV_mean

        else:
            # Calculate amplitude of largest contribution to noise
            # take FFT
            Y = np.fft.fft(rawsignalcurve_TFNI)
        
            # Compute the two-sided spectrum P2
            P2 = np.abs(Y/pixz)
        
            # Compute the one-sided spectrum P1, with the total energy in P2 conserved
            # by multiplication of half of the spectrum by 2, except for the DC
            # component
        
            P1 = 2*P2[0:int(pixz/2.+1)] # This will keep the AUC equal to that of P2!
            P1[0] = P2[0]

        rawsignalcurve_TFNI_fft = [ p*Fs/pixz for p in range(len(P1)) ]

        self.makePlots('ResidualNoise', {
            'rawsignalcurve_TFNI_fft': rawsignalcurve_TFNI_fft, 'P1': P1,
        })
            
        return rawsignalcurve_TFNI_fft, P1
    
    def TemporalNoise(self, im_reg, im_mean, xran, yran, dynscantime):
        """
        Temporal Fluctuation Noise Image, the time-series across the pixz images for 
        each voxel is detrended with a 2nd-order polynomial. The Temporal Fluctuation 
        Noise Image (SD Image) is the standard deviation of the residuals, voxel by 
        voxel, after this detrending step.
        There are several contributions to the temporal intensity fluctuations at any given voxel:
             * functional activity itself
             * electronics noise (usually white noise from receiver)
             * physiological changes (respiration & heart beat)
             * scanner noise (B0 drift due to heating, etc)
        The main source of temporal variation of average intensity will come from scanner 
        noise & physio activity (which can be recorded and subtracted).
        """
        

        # make raw signal curve within square mask im_square
        rawsignalcurve = [ np.mean(im[ xran[0]:xran[1], yran[0]:yran[1] ]) for im in im_reg ] # curve within square ROI
    
        # Fit raw signal curve to 2nd order poly
        x0 = [3e-5, 3e-3, np.max(rawsignalcurve)] # start cond for lsqcurvefit
        x, pcov = curve_fit(poly2, dynscantime, rawsignalcurve, p0=x0)
    
        im_TFNI             = mmath.detrend(im_reg, poly2(dynscantime, *x))
        rawsignalcurve_TFNI = [ np.mean(im[ xran[0]:xran[1], yran[0]:yran[1] ]) for im in im_TFNI ] # curve within square ROI
        im_TFNI_sd          = np.std(im_TFNI, axis=0) # div N-1 or N?
    
        # SFNR Image is a quotient, voxel by voxel, between the Mean Signal Image and 
        # the Temporal Fluctuation Noise Image.
        im_SFNR = im_mean/im_TFNI_sd
        im_SFNR[ im_TFNI_sd < 1.e-12 ] = 0.
        if 1<0: # wanna look like philips restricted to square?
            im_SFNR[      0:xran[0],       :       ] = 0.
            im_SFNR[xran[1]:-1     ,       :       ] = 0.
            im_SFNR[       :,             0:yran[0]] = 0.
            im_SFNR[       :,       yran[1]:-1     ] = 0.
        
        if self.CALCMODE == 'Philips':
            # detrend each pixel
            dimx, dimy = np.shape(im_mean)
            im_TFNI_sd = np.zeros( (dimx, dimy) )
            for xi in range(xran[0], xran[1]): #range(dimx):
                for yi in range(yran[0], yran[1]): #range(dimy):
                    rawsignal = [ im[xi,yi] for im in im_reg ]
                    fx0 = [3e-5, 3e-3, np.max(rawsignal)] # start cond for lsqcurvefit
                    fx, pcov = curve_fit(poly2, dynscantime, rawsignal, p0=fx0)
                    detrendsignal = mmath.detrend(rawsignal, poly2(dynscantime, *fx))
                    im_TFNI_sd[xi,yi] = np.std(detrendsignal)
            im_SFNR = im_mean/im_TFNI_sd
            im_SFNR[ im_TFNI_sd < 1.e-12 ] = 0.

        self.makePlots('TemporalNoise', {
            'dynscantime': dynscantime, 'rawsignalcurve':rawsignalcurve, 'x':x,
            'rawsignalcurve_TFNI': rawsignalcurve_TFNI,
            'im_TFNI_sd': im_TFNI_sd,
            'im_SFNR': im_SFNR})
            
        # for gui only
        if self.guimode:
            self.guistuff['im_TFNI_sd']   = np.array(im_TFNI_sd)
            self.guistuff['im_TFNI_SFNR'] = np.array(im_SFNR)

        return rawsignalcurve, x, rawsignalcurve_TFNI, im_SFNR

    def SquareInPhantom(self, im_mean):
        # Find largest square within phantom

        # mask image, define ROI for phantom, Philips uses a SI threshold
        threshold = 0.25 * np.max(im_mean)     # more or less arbitrary selection of threshold value
        im_mask   = (im_mean>threshold).astype(int)  # dim: pixx * pixy (use int else problems when rotating)
        
        # find center of mass of mask
        com = scind.measurements.center_of_mass(im_mask)
        com = [int(round(c)) for c in com]

        # define largest square within mask_beeld as ROI for further analysis
        angles = range(0,180,5) # rotate image over angles to find center of mass
        npixs  = [] 
        for ang in angles:
            # Rotate image
            im_rot = scind.interpolation.rotate(im_mask, ang) # angle in degrees
            # Calculate center of mass     
            center = scind.measurements.center_of_mass(im_rot)

            # Find first and last nonzero pixel in mask @ x-coord center of mass
            line = list(im_rot[int(round(center[0])),:])# 0 = x

            # Calculate diam (# of pixs) @ center of mass
            left  = line.index(1)
            right = len(line)-1-list(reversed(line)).index(1)
            npixs.append(right - left)

        # Calculate rib size of largest square within "circle"
        halfrib = int(np.floor(np.sqrt(1./2.)*min(npixs)/2.))

        if self.CALCMODE == 'Philips':
            halfrib -= 1 # no drift correction, so make surer the box will stay in the phantom
            
        # Make the square
        xran  = [com[0]-halfrib, com[0]+halfrib+1] # +1 to include upper limit
        yran  = [com[1]-halfrib, com[1]+halfrib+1]
        im_square = np.zeros(np.shape(im_mean))
        im_square[ xran[0]:xran[1], yran[0]:yran[1] ] = 1

        return im_square, xran, yran, halfrib, com, im_mask
    
    def QC(self, cs):
        """
        #  1. sort images on acquisition time
        #  2. remove last 1 or 2 images (last is image noise and we want an even number)
        #  3. rigid (translation, rotation) shift images to im0
        #  4. find largest square within phantom
        #  5. temporal Noise analysis
        #  6. residual noise
        #  7. Ghost and Signal-to-Noise
        #  8. Static Spatial Noise
        #  9. Weisskoff
        # 10. add results
        """
        error   = False

        info         = cs.dcmInfile._datasets # a stack of all dicom files (headers and data)
        im_unreg     = cs.pixeldataIn         # a stack of just the dicom data
        self.verbose = cs.verbose
        print('runmode:',self.CALCMODE)
        
        ## 1. sort images on acquisition time
        # Get time vector for the pixz images. It might be that these images are
        # not sorted chronologically. This function will fail if data was acquired
        # around midnight
        dynscantime, I = mmath.TimeVector(info) # sorted scantime in sec
        im_unreg       = im_unreg[I]
        info           = [info[i] for i in I]

        ## 2. remove last 1 or 2 images (last is image noise and we want an even number)
        # Ignore last 2 images (to keep even number of images, and ignore noise
        # image at temporal position 2000
        if np.mod(info[0].NumberOfTemporalPositions,2) == 1:
            removeNslices = 1
        elif np.mod(info[0].NumberOfTemporalPositions,2) == 0:
            removeNslices = 2
        im_unreg    = im_unreg[0:-removeNslices]
        dynscantime = dynscantime[0:-removeNslices]
        info        = info[0:-removeNslices]
        pixz        = len(im_unreg)

        ## 3. rigid shift images to im0
        if self.CALCMODE == 'Philips':
            im_reg = im_unreg.astype(float)
        else:
            im_reg = mmath.RegisterImages(im_unreg)
            
        ## 4. Calculate square which fits in phantom
        # mean signal image, is the simple average, voxel by voxel, across pixz images
        im_mean = np.average(im_reg,axis=0) # dim: pixx * pixy
        im_square, xran, yran, halfrib, com, im_mask = self.SquareInPhantom(im_mean)
        
        # The Mean Signal Summary Value is the average across the 15x15 square ROI 
        # placed in the center of the phantom in the Mean Signal Image.
        SV_mean = np.mean(im_mean[xran[0]:xran[1], yran[0]:yran[1]])
        self.SV_mean = SV_mean
        
        ## 5. temporal Noise analysis
        rawsignalcurve, x, rawsignalcurve_TFNI, im_SFNR = \
            self.TemporalNoise(im_reg, im_mean, xran, yran, dynscantime)

        ## 6. residual Noise
        rawsignalcurve_TFNI_fft, P1 = self.ResidualNoise(info, rawsignalcurve_TFNI)
        
        ## 7. Calculate the raw signal ghost graph
        rawsignalcurve_ghost, rawsignalcurve_ghost_sd, rawsignalcurve_snr, rawsignalcurve_snr_sd, mask_phase = \
            self.GhostSNR(info, im_unreg, im_mask, im_mean, im_square, dynscantime, rawsignalcurve)

        ## 8. Static Spatial Noise
        im_SSNI = self.StaticSpatialNoise(im_reg)

        ## 9. Weisskoff
        SV_SNR_regular = np.sum(rawsignalcurve)/(self.Rayleighcorr*np.sum(rawsignalcurve_snr_sd))
        Fn = self.Weisskoff(im_reg, halfrib, com, SV_SNR_regular)
        
        ## Registration
        self.makePlots('Registration', {
            'im_unreg': im_unreg, 'im_reg':im_reg})
        
        ## 10. Add results
        #Mean Signal Summary Value
        self.results.append( ('float', 'SV_mean', SV_mean) )
        self.results.append( ('float', 'N_dynamics', len(im_reg)) )

        # SFNR Summary Value
        # The SFNR Summary Value is the average across the 15x15 square ROI placed 
        # in the center of the phantom in the SFNR Image.
        SV_SFNR = np.mean(im_SFNR[xran[0]:xran[1], yran[0]:yran[1]])
        self.results.append( ('float', 'SV_SFNR', SV_SFNR) )

        # SNR Summary Value
        # The SNR Summary Value is (mean signal summary value)/sqrt((variance value)/pixz time points).  
        # Where, the Variance Value is the variance across the NxN square ROI 
        # placed in the center of the phantom in the Static Spatial Noise Image.
        SV_SNR_ssni = SV_mean/np.sqrt(np.var(im_SSNI[xran[0]:xran[1], yran[0]:yran[1]])/pixz)
        self.results.append( ('float', 'SV_SNR_ssni', SV_SNR_ssni) )

        # If the image homogeneity is not considered to be good, then the SNR may be derived 
        # more accurately using the following (NEMA) method. Two images should be acquired by 
        # consecutive scans with identical receiver and transmitter settings. The images should 
        # then be subtracted one from the other, to generate a third pixel-by-pixel difference 
        # image. The only difference between the two original images should be due to noise, 
        # provided the image has not suffered from ghosting or any other instability. So we now 
        # have two original images, and a subtracted image. Using either of the original images 
        # the signal (S) is again defined as the mean pixel intensity value in a ROI. The noise 
        # is the standard deviation (?) in the same ROI on the subtracted image. The signal to 
        # noise ratio is determined using SNR = ?2?S/?, where the factor of ?2 arises due to 
        # the fact that the standard deviation is derived from the subtraction image and not from 
        # the original image.
        # SNR Summary Value
        # Define SNR according to Weisskoff 1996, and taking into account that
        # noise in MR data is distributed according to a Rician distribution!
        self.results.append( ('float', 'SV_SNR_regular', SV_SNR_regular) )
        
        # Corrected ghost percentage
        SV_ghost = 100.*np.mean([(g-s)/r for g,s,r in zip(rawsignalcurve_ghost, rawsignalcurve_snr, rawsignalcurve)])
        self.results.append( ('float', 'SV_ghost', SV_ghost) )
        
        # Drift percentage, is defined based on the max. and min. value of the 2nd
        # order polynome fitted to the raw signal time curve:
        yfit = poly2(dynscantime, *x)
        SV_drift = (np.min(yfit)-np.max(yfit))/np.max(yfit)*100
        self.results.append( ('float', 'SV_drift', SV_drift) )

        # RDC Summary Value
        # The RDC Summary Value may be thought of as a measure of the size of ROI at 
        # which statistical independence of the voxels is lost, and is derived directly 
        # from the Weisskoff Plot: a log-log plot of coefficient of variation (CV) and 
        # the size of an ROI.  CV is defined as the standard deviation of a time-series 
        # divided by the mean of the time-series.  If each voxel is (relatively) 
        # independent of its neighbors, then CV for an ROI should scale inversely with 
        # the square root of the number of voxels in the ROI.  Thus, for a square NxN 
        # voxel ROI, a plot of log(CV) vs. log(N) should follow a declining straight 
        # line.  In practice, as N increases, the reduction in CV plateaus and 
        # becomes independent of N.  This occurs because system instabilities 
        # result in low-spatial-frequency image correlations, so that the statistical 
        # independence of the voxels is lost.  Define radius of decorrelation (RDC) 
        # as CV(1)/CV(Nmax), where Nmax is 15.  The RDC is the intercept between 
        # the theoretical CV(N) and the extrapolation of measured CV(Nmax).
        RDC = Fn[0]/Fn[-1]
        self.results.append( ('float', 'RDC', RDC) )
        # The interpretation of RDC depends on the max. size of the square ROI
        # used! This should be constant over time!

        self.results.append( ('float', 'MaxResNoiseAmplitude', np.max(P1)) )
        self.results.append( ('float', 'MaxResNoiseFrequency', rawsignalcurve_TFNI_fft[np.argmax(P1)]) )

        ## 11. Summary
        # Display results according to Philips GUI for data analysis
        
        print('Size square ROI:                 ',xran[1]-xran[0])
        print('Num pixs in square ROI:          ',(xran[1]-xran[0])*(yran[1]-yran[0]))
        print('Num pixs in ghost ROI:           ', np.sum(mask_phase))
        largestSTD = np.max( [ np.std(im[xran[0]:xran[1], yran[0]:yran[1]]) for im in im_reg ] ) # largest STD single pixel of rawsignalcurve
        dummy =  np.std(np.reshape(im_reg[:, xran[0]:xran[1],yran[0]:yran[1]],[pixz, (xran[1]-xran[0])*(yran[1]-yran[0])]),axis=0)#; % largest STD single pixel of rawsignalcurve

        largestSTD = np.max( np.std(np.reshape(im_reg[:, xran[0]:xran[1],yran[0]:yran[1]],[pixz, (xran[1]-xran[0])*(yran[1]-yran[0])]),axis=0))#; % largest STD single pixel of rawsignalcurve

        midSTD     = np.std( im_reg[:,com[0],com[1]] ) # in center of mass of phantom

        self.results.append( ('float', 'STD_single_pixel_pct', 100*largestSTD/np.mean(rawsignalcurve)) )
        self.results.append( ('float', 'STD_com_pixel_pct', 100*midSTD/np.mean(rawsignalcurve)) )
        self.results.append( ('float', 'STD_largest_roi_pct', 100*np.std(rawsignalcurve)/np.mean(rawsignalcurve)) )
        print('Relative STD single pixel (%):   ', self.results[-3][2])
        print('Relative STD c.o.m. pixel (%):   ', self.results[-2][2])
        print('Relative STD largest ROI (%):    ', self.results[-1][2])
        print('Radius of DeCorrelation (RDC):   ', RDC)
        print('Drift (%):                       ', SV_drift)
        print('Ghost (%):                       ', SV_ghost)
        print('SNR image(-):                    ', SV_SNR_regular)
        print('SNR summary value (-):           ', SV_SNR_ssni)
        print('SFNR summary value (-):          ', SV_SFNR)
        print('Noise spectrum peak amplitude(-):', np.max(P1)) # finally the measure we want?
        print('Noise spectrum peak frequency(-):', rawsignalcurve_TFNI_fft[np.argmax(P1)]) # finally the measure we want?
        print('SV_mean (-):                     ', self.SV_mean)
        print('Done!')

        # make some stuff accessible for gui
        # for gui only
        if self.guimode:
            self.guistuff['im_reg']  = im_reg
            self.guistuff['im_square']  = im_square
            self.guistuff['im_reg_diff']  = np.array([im_unreg[-1]-im_unreg[0], im_reg[-1]-im_reg[0]])

            cs.guistuff = self.guistuff
        
        cs.hasmadeplots = True
        if self.verbose:
            plt.show()
        return error, self.results

    def makePlots(self, what, cs):
        if what == 'GhostSNR':
            # show square within mask
            plt.figure() #1
            plt.title('Time-averaged image with ROI')
            cax = plt.imshow(np.transpose(cs['im_mean']), cmap = cm.gray) #cm.get_cmap('gray', 25))#cmap=cm.gray) #, alpha = .5
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['im_mean'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['im_square']), levels=[.5], colors='r', label='Phantom ROI')
            plt.contour(range(dimx), range(dimy), np.transpose(cs['mask_phase']), levels=[.5], colors='g', label='Phase ROI')
            plt.contour(range(dimx), range(dimy), np.transpose(cs['mask_snr']), levels=[.5], colors='b', label='SNR ROI')
            plt.plot(0,0, color='r', label='Phantom ROI') # dummy for legend
            plt.plot(0,0, color='g', label='Phase ROI') # dummy for legend
            plt.plot(0,0, color='b', label='SNR ROI') # dummy for legend
            plt.legend()
            fname = 'figure1.jpg'
            self.results.append(('object', 'ROIs', fname))
            plt.savefig(fname)

            # Raw signal graph - ghost ROI
            plt.figure() #8
            plt.title('Raw signal ghost ROI')
            plt.plot(cs['dynscantime'],cs['rawsignalcurve_ghost'],'k.', label='mean')
            plt.plot(cs['dynscantime'],cs['rawsignalcurve_ghost_sd'],'r.', label='stdev')
            plt.legend()
            plt.xlabel('Acquisition time (s)')
            fname = 'figure8.jpg'
            self.results.append(('object', 'GhostROI', fname))
            plt.savefig(fname)

            plt.figure() #8a
            plt.title('Corrected signal ghost ROI')
            plt.plot(cs['dynscantime'],[g-s for g,s in zip(cs['rawsignalcurve_ghost'], cs['rawsignalcurve_snr'])],'k.')
            plt.xlabel('Acquisition time (s)')
            fname = 'figure8a.jpg'
            self.results.append(('object', 'CorrectedGhostROI', fname))
            plt.savefig(fname)
        
            # Raw signal graph - SNR ROI
            plt.figure() #9
            plt.title('Raw signal SNR ROI')
            plt.plot(cs['dynscantime'],cs['rawsignalcurve_snr'],'k.', label='mean')
            plt.plot(cs['dynscantime'],cs['rawsignalcurve_snr_sd'],'r.', label='stdev')
            plt.legend()
            plt.xlabel('Acquisition time (s)')
            fname = 'figure9.jpg'
            self.results.append(('object', 'SNRROI', fname))
            plt.savefig(fname)

            # SNR curve
            plt.figure() #10
            plt.title('SNR graph')
            plt.plot(cs['dynscantime'], cs['rawsignalcurve_snr_snr'],'k.')
            plt.xlabel('Acquisition time (s)')
            fname = 'figure10.jpg'
            self.results.append(('object', 'SNR', fname))
            plt.savefig(fname)

        elif what == 'TemporalNoise':
            plt.figure() #2
            plt.title('Raw signal time curve')
            plt.plot(cs['dynscantime'],cs['rawsignalcurve'], 'k.')
            plt.plot(cs['dynscantime'],poly2(cs['dynscantime'], *(cs['x'])), 'r-')
            plt.xlabel('Acquisition time [s]')
            fname = 'figure2.jpg'
            self.results.append(('object', 'SignalTime', fname))
            plt.savefig(fname)

            plt.figure() #3
            plt.title('Detrended raw signal time curve')
            plt.plot(cs['dynscantime'],cs['rawsignalcurve_TFNI'], 'k.')
            plt.xlabel('Acquisition time [s]')
            fname = 'figure3.jpg'
            self.results.append(('object', 'SignalTimeDetrend', fname))
            plt.savefig(fname)

            plt.figure()#5
            #Temporal Fluctuation Noise Image
            plt.title('Temporal Fluctuation Noise Image')
            cax = plt.imshow(np.transpose(cs['im_TFNI_sd']), cmap=cm.gray,
                             vmin=np.min(cs['im_TFNI_sd']),
                             vmax=0.25*np.max(cs['im_TFNI_sd']))
            cbar = plt.colorbar(cax)
            fname = 'figure5.jpg'
            self.results.append(('object', 'TemporalNoise', fname))
            plt.savefig(fname)
        
            plt.figure()#6
            # Signal-to-Fluctuation-Noise Ratio (SFNR) image
            plt.title('Signal-to-Fluctuation-Noise Ratio (SFNR) image')
            cax = plt.imshow(np.transpose(cs['im_SFNR']), cmap=cm.gray)
            cbar = plt.colorbar(cax)
            fname = 'figure6.jpg'
            self.results.append(('object', 'SFNR', fname))
            plt.savefig(fname)

        elif what == 'ResidualNoise':
            plt.figure() #4
            # Residual noise spectrum
            plt.title('Single-sided amp. spectrum of detrended raw signal time curve')
            plt.plot(cs['rawsignalcurve_TFNI_fft'],cs['P1'])
            plt.xlabel('f (Hz)')
            plt.ylabel('|P1(f)|')
            plt.xlim(xmin=0, xmax=np.max(cs['rawsignalcurve_TFNI_fft']))
            plt.ylim(ymin=0, ymax=1)
            fname = 'figure4.jpg'
            self.results.append(('object', 'ResidualNoise', fname))
            plt.savefig(fname)

        elif what == 'StaticSpatialNoise':
            # Static Spatial Noise Image
            plt.figure() #7 
            plt.title('Static Spatial Noise Image')
            cax = plt.imshow(np.transpose(cs['im_SSNI']), cmap=cm.gray) #, alpha = .5
            cbar = plt.colorbar(cax)
            fname = 'figure7.jpg'
            self.results.append(('object', 'SSNI', fname))
            plt.savefig(fname)
 
        elif what == 'Weisskoff':
            # Weisskoff plot
            plt.figure() #11
            plt.title('Weisskoff plot')
            plt.loglog(cs['ROIlength'],100.*np.array(cs['Fn']),color='k', marker='.', linestyle='-')
            plt.loglog(cs['ROIlength'],100.*np.array(cs['Fnt']),'b')
            plt.grid(which='both')
            plt.xlabel('Square ROI width (-)')
            plt.ylabel('Relative deviation (%)')
            fname = 'figure11.jpg'
            self.results.append(('object', 'Weisskoff', fname))
            plt.savefig(fname)
        
        elif what == 'Registration':
            # Efficacy of registration
            plt.figure()
            plt.title('Unregistered')
            unreg = cs['im_unreg'][-1].astype(float)-cs['im_unreg'][0].astype(float)
            cax = plt.imshow(np.transpose(unreg), cmap = cm.gray) #cm.get_cmap('gray', 25))#cmap=cm.gray) #, alpha = .5
            cbar = plt.colorbar(cax)
            fname = 'figure20.jpg'
            self.results.append(('object', 'Unregistered', fname))
            plt.savefig(fname)
            plt.figure()
            plt.title('Registered')
            reg = cs['im_reg'][-1]-cs['im_reg'][0]
            cax = plt.imshow(np.transpose(reg), cmap = cm.gray) #cm.get_cmap('gray', 25))#cmap=cm.gray) #, alpha = .5
            cbar = plt.colorbar(cax)
            fname = 'figure21.jpg'
            self.results.append(('object', 'Registered', fname))
            plt.savefig(fname)

        elif what == 'B1mapping':
            # agreement between programmed and measured B1
            plt.figure()
            plt.plot(range(len(cs['im_mean_curve'])),cs['im_mean_curve'], color='k', marker='.', linestyle='-', label='mean')
            plt.plot(range(len(cs['im_std_curve'])),cs['im_std_curve'], color='r', marker='.', linestyle='-', label='stdev')
            plt.plot(range(len(cs['im_min_curve'])),cs['im_min_curve'], color='g', marker='.', linestyle='-', label='min')
            plt.plot(range(len(cs['im_max_curve'])),cs['im_max_curve'], color='b', marker='.', linestyle='-', label='max')
            plt.legend()
            plt.xlabel('Frame')
            plt.ylabel('Flip Angle measured/programmed [%]')
            fname = 'figure20.jpg'
            self.results.append(('object', 'B1_curves', fname))
            plt.savefig(fname)
        
            # Plot edge of defined circle mask on top of image
            plt.figure()
            plt.title('Analysis B1 map')
            cax = plt.imshow(np.transpose(cs['image_frame']), cmap = cm.gray) #cm.get_cmap('gray', 25))#cmap=cm.gray) #, alpha = .5
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['image_frame'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['image_circle']), levels=[.5], colors='g', label='Phantom ROI')
            plt.plot(0,0, color='g', label='Phantom ROI') # dummy for legend
            plt.legend()
            fname = 'figure21.jpg'
            self.results.append(('object', 'B1_analysis', fname))
            plt.savefig(fname)
            
        elif what == 'B0mapping':
            # agreement between programmed and measured B0
            plt.figure()
            plt.plot(range(len(cs['im_mean_curve'])),cs['im_mean_curve'], color='k', marker='.', linestyle='-', label='mean')
            plt.plot(range(len(cs['im_std_curve'])),cs['im_std_curve'], color='r', marker='.', linestyle='-', label='stdev')
            plt.plot(range(len(cs['im_min_curve'])),cs['im_min_curve'], color='g', marker='.', linestyle='-', label='min')
            plt.plot(range(len(cs['im_max_curve'])),cs['im_max_curve'], color='b', marker='.', linestyle='-', label='max')
            plt.legend()
            plt.xlabel('Frame')
            plt.ylabel('B0 measured/programmed [ppm]')
            fname = 'figure41.jpg'
            self.results.append(('object', 'B0_curves', fname))
            plt.savefig(fname)
        
            # Plot edge of defined circle mask on top of image
            plt.figure()
            plt.title('Analysis B0 map - PPM')
            cax = plt.imshow(np.transpose(cs['image_frame']), cmap = cm.gray) #cm.get_cmap('gray', 25))#cmap=cm.gray) #, alpha = .5
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['image_frame'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['image_circle']), levels=[.5], colors='g', label='Phantom ROI')
            plt.plot(0,0, color='g', label='Phantom ROI') # dummy for legend
            plt.legend()
            fname = 'figure42.jpg'
            self.results.append(('object', 'B0_analysis', fname))
            plt.savefig(fname)
        elif what == 'PureSNR':
            plt.figure()
            plt.title('PureSNR')
            #plt.plot(range(len(cs['im_mean_curve'])),cs['im_mean_curve'], color='k', marker='.', linestyle='-', label='mean')
            plt.plot(range(len(cs['im_std_curve'])),cs['im_std_curve'], color='r', marker='.', linestyle='-', label='stdev')
            plt.plot(range(len(cs['snr_slice'])),cs['snr_slice'], color='g', marker='.', linestyle='-', label='SNR')
            plt.legend()
            plt.xlabel('Frame')
            fname = 'figure50.jpg'
            self.results.append(('object', 'PureSNR_curves', fname))
            plt.savefig(fname)

            # Plot edge of defined circle mask on top of image
            plt.figure()
            plt.title('PureSNR Signal')
            cax = plt.imshow(np.transpose(cs['imSignal_frame']), cmap = cm.gray) 
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['imSignal_frame'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['imSignal_circle']), levels=[.5], colors='g', label='Phantom ROI')
            plt.plot(0,0, color='g', label='Phantom ROI') # dummy for legend
            plt.legend()
            fname = 'figure51.jpg'
            self.results.append(('object', 'PureSNR_Signal', fname))
            plt.savefig(fname)

            plt.figure()
            plt.title('PureSNR Noise')
            cax = plt.imshow(np.transpose(cs['imNoise_frame']), cmap = cm.gray) 
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['imNoise_frame'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['imNoise_circle']), levels=[.5], colors='g', label='Phantom ROI')
            plt.plot(0,0, color='g', label='Phantom ROI') # dummy for legend
            plt.legend()
            fname = 'figure52.jpg'
            self.results.append(('object', 'PureSNR_Noise', fname))
            plt.savefig(fname)

        elif what == 'Receivemapping':
            # Plot edge of defined circle mask on top of image
            plt.figure()
            plt.title('Receive: Normalized B1 map')
            cax = plt.imshow(np.transpose(cs['im_B1_frame']), cmap = cm.gray) 
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['im_B1_frame'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['circle_frame']), levels=[.5], colors='g', label='Phantom ROI')
            plt.plot(0,0, color='g', label='Phantom ROI') # dummy for legend
            plt.legend()
            fname = 'figure61.jpg'
            self.results.append(('object', 'Receive_B1norm', fname))
            plt.savefig(fname)
            
            plt.figure()
            plt.title('Receive: Normalized FA map')
            cax = plt.imshow(np.transpose(cs['im_FA_frame']), cmap = cm.gray) 
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['im_FA_frame'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['circle_frame']), levels=[.5], colors='g', label='Phantom ROI')
            plt.plot(0,0, color='g', label='Phantom ROI') # dummy for legend
            plt.legend()
            fname = 'figure62.jpg'
            self.results.append(('object', 'Receive_FAnorm', fname))
            plt.savefig(fname)

            plt.figure()
            plt.title('Receive: Receive map')
            cax = plt.imshow(np.transpose(cs['im_Receive_frame']), cmap = cm.gray) 
            cbar = plt.colorbar(cax)
            dimx,dimy = np.shape(cs['im_Receive_frame'])
            plt.contour(range(dimx), range(dimy), np.transpose(cs['circle_frame']), levels=[.5], colors='g', label='Phantom ROI')
            plt.plot(0,0, color='g', label='Phantom ROI') # dummy for legend
            plt.legend()
            fname = 'figure63.jpg'
            self.results.append(('object', 'Receive_map', fname))
            plt.savefig(fname)

            
    def B1Mapping(self, cs):
        """
        Determine average nominal flip angle as measured/programmed in pct
        """
        error   = False
        
        # B1 map images contain the nominal flip angle expressed as percentage 
        # of the flip angle chosen by the operator

        info         = cs.dcmInfile._datasets # a stack of all dicom files (headers and data)
        images       = cs.pixeldataIn         # a stack of just the dicom data
        self.verbose = cs.verbose

        # Data contains images with the B1 map, magnitude images, ... . Select 
        # B1maps, ignore other data. This should result in 1/4 of the loaded
        # images!

        # filter on ImageType,'ORIGINAL\PRIMARY\M_B1\M\B1' Alternatively, filter on ScanningSequence 'B1'
        desc = np.array([ '\\'.join(im.ImageType) for im in info ])
        images = images[ desc == 'ORIGINAL\\PRIMARY\\M_B1\\M\\B1' ]

        guicircles = [] # (x,y,rad) for gui
        im_circ   = mmath.CircleImages(images, self.usePartCircle, guicircles)

        # Calculate mean nominal flip angle and STDEV in all nSlices ROIs
        im_mean_curve = [ np.ma.array(im, mask=~mask).mean() for im,mask in zip(images, im_circ) ] 
        im_std_curve  = [ np.ma.array(im, mask=~mask).std() for im,mask in zip(images, im_circ) ] 
        im_min_curve  = [ np.ma.array(im, mask=~mask).min() for im,mask in zip(images, im_circ) ] 
        im_max_curve  = [ np.ma.array(im, mask=~mask).max() for im,mask in zip(images, im_circ) ] 

        print('Mu:    ', im_mean_curve)
        print('Sigma: ', im_std_curve)

        # add overall results
        ma = np.ma.array(images, mask=~np.array(im_circ))
        self.results.append( ('float', 'NominalB1_mean_ppm', ma.mean()) )
        self.results.append( ('float', 'NominalB1_std_ppm', ma.std()) )
        ma_max = ma.max()
        ma_min = ma.min()
        self.results.append( ('float', 'NominalB1_min_ppm', ma_min) )
        self.results.append( ('float', 'NominalB1_max_ppm', ma_max) )
        self.results.append( ('float', 'NominalB1_nonuniformity_pct', 100.*(ma_max-ma_min)/(ma_max+ma_min)) )

        frame = 0 # just select a frame for plotting
        self.makePlots('B1mapping', {
            'im_mean_curve':im_mean_curve,
            'im_std_curve':im_std_curve,
            'im_min_curve':im_min_curve,
            'im_max_curve':im_max_curve,
            'image_frame':images[frame],
            'image_circle':im_circ[frame]
        })
        cs.hasmadeplots = True
        if self.verbose:
            plt.show()

        # make some stuff accessible for gui
        if self.guimode:
            cs.guistuff['b1_images']  = np.array(images)
            cs.guistuff['b1_circles'] = guicircles

        print('Done!')

        return error, self.results

    def B0Mapping(self, cs):
        """
        Determine average nominal B0 as measured/programmed in ppm.

        Ref: Advances in Concurrent Motion and Field-Inhomogeneity Correction in Functional MRI, 
        Teck Beng Desmond Yeo, The University of Michigan, 2008, PhD thesis, Chapter 3.2
        """
        error   = False
        
        # B1 map images contain the nominal flip angle expressed as percentage 
        # of the flip angle chosen by the operator

        info         = cs.dcmInfile._datasets # a stack of all dicom files (headers and data)
        images       = cs.pixeldataIn         # a stack of just the dicom data
        self.verbose = cs.verbose

        # Data contains images with the magnitude images, and the B0map. Select 
        # B0maps, ignore other data. This should result in 1/2 of the loaded
        # images!

        # filter on ImageType,'ORIGINAL\PRIMARY\B0 MAP\B0\UNSPECIFIED' Alternatively, filter on ScanningSequence 'B0'?
        desc = np.array([ '\\'.join(im.ImageType) for im in info ])
        keeptype = 'ORIGINAL\\PRIMARY\\B0 MAP\\B0\\UNSPECIFIED'
        imagesHz   = images[ desc == keeptype ] # Hz images
        imagesMagn = images[ desc != keeptype ] # Magnitude images
        infoHz = [] # ignore info of magn images
        for im in info:
            if '\\'.join(im.ImageType) == keeptype:
                infoHz.append(im)

        guicircles = [] # (x,y,rad) for gui
        im_circ   = mmath.CircleImages(imagesMagn, self.usePartCircle, guicircles)

        # Calculate mean nominal B0 and STDEV in all nSlices ROIs
        im_PPM        = [ im/i.ImagingFrequency for im,i in zip(imagesHz,infoHz) ] # ppm; freq in MHz
        im_mean_curve = [ np.ma.array(im, mask=~mask).mean() for im,mask in zip(im_PPM, im_circ) ] 
        im_min_curve  = [ np.ma.array(im, mask=~mask).min() for im,mask in zip(im_PPM, im_circ) ] 
        im_max_curve  = [ np.ma.array(im, mask=~mask).max() for im,mask in zip(im_PPM, im_circ) ] 
        im_std_curve  = [ np.ma.array(im, mask=~mask).std() for im,mask in zip(im_PPM, im_circ) ] 
        
        print('Mu:    ', im_mean_curve)
        print('Sigma: ', im_std_curve)
        print('Min:   ', im_min_curve)
        print('Max:   ', im_max_curve)

        # add overall results
        ma = np.ma.array(im_PPM, mask=~np.array(im_circ))
        self.results.append( ('float', 'NominalB0_mean_ppm', ma.mean()) )
        self.results.append( ('float', 'NominalB0_std_ppm', ma.std()) )
        self.results.append( ('float', 'NominalB0_min_ppm', ma.min()) )
        self.results.append( ('float', 'NominalB0_max_ppm', ma.max()) )

        frame = 0 # just select a frame for plotting
        self.makePlots('B0mapping', {
            'im_mean_curve':im_mean_curve,
            'im_std_curve':im_std_curve,
            'im_min_curve':im_min_curve,
            'im_max_curve':im_max_curve,
            'image_frame':im_PPM[frame],
            'image_circle':im_circ[frame]
        })
        cs.hasmadeplots = True
        if self.verbose:
            plt.show()

        # make some stuff accessible for gui
        if self.guimode:
            cs.guistuff['b0_im_PPM']  = np.array(im_PPM)
            cs.guistuff['b0_circles'] = guicircles
            cs.guistuff['b0_im_circ']  = np.array(im_circ)
        
        print('Done!')
            
        return error, self.results
    
    def PureSNR(self, cs):
        """
        Determine SNR from acquisition without Gradient and without RF.

        """
        error   = False
        
        # Images of single slice

        info         = cs.dcmInfile._datasets # a stack of all dicom files (headers and data)
        images       = cs.pixeldataIn         # a stack of just the dicom data
        self.verbose = cs.verbose

        # Data contains images with the magnitude images, and the phase images. 
        # Select only phase images.
        desc = np.array([ '\\'.join(im.ImageType) for im in info ])
        keeptype = 'ORIGINAL\\PRIMARY\\M_FFE\\M\\FFE'
        images = images[ desc == keeptype ] # Magnitude images
        info = np.array(info)[ desc == keeptype ]

        # split in signal and noise images
        desc = np.array([ im.TemporalPositionIdentifier for im in info ])
        imagesSignal = images[ desc == 1 ] # first images
        imagesNoise  = images[ desc == 2 ] # images after pauze

        guicirclesSignal = [] # (x,y,rad) for gui
        im_circSignal    = np.array(mmath.CircleImages(imagesSignal, self.usePartCircle, guicirclesSignal))

        guicirclesNoise = [] # (x,y,rad) for gui
        im_circNoise    = np.array(mmath.CircleImages(imagesNoise, self.usePartCircle, guicirclesNoise))
        

        # calculate SNR over each ROI, and overall
        im_mean_curve = np.array( [ np.ma.array(im, mask=~mask).mean() for im,mask in zip(imagesSignal, im_circSignal) ] )
        im_std_curve  = np.array( [ np.ma.array(im, mask=~mask).std() for im,mask in zip(imagesNoise, im_circNoise) ] )
        snr_slice = im_mean_curve/(self.Rayleighcorr*im_std_curve)
        
        snr_overall =  np.ma.array(imagesSignal, mask=~im_circSignal).mean()/(self.Rayleighcorr*np.ma.array(imagesNoise, mask=~im_circNoise).std())
        frame       = int(len(snr_slice)/2.+.5)# frame number to show in results, of middle frame
        snr_middle  = snr_slice[frame]

        # add results: snr in middle frame, snr_overall
        self.results.append( ('float', 'PureSNR_overall', snr_overall) )
        self.results.append( ('float', 'PureSNR_middle', snr_middle) )

        self.makePlots('PureSNR', {
            'im_mean_curve':im_mean_curve,
            'im_std_curve':im_std_curve,
            'snr_slice':snr_slice,
            'imNoise_frame':imagesNoise[frame],
            'imSignal_frame':imagesSignal[frame],
            'imNoise_circle':im_circNoise[frame],
            'imSignal_circle':im_circSignal[frame],
        })
        cs.hasmadeplots = True
        if self.verbose:
            plt.show()

        # make some stuff accessible for gui
        if self.guimode:
            cs.guistuff['PureSNR_imagesNoise']  = np.array(imagesNoise)
            cs.guistuff['PureSNR_maskNoise'] = np.array(im_circNoise)
            cs.guistuff['PureSNR_imagesSignal']  = np.array(imagesSignal)
            cs.guistuff['PureSNR_maskSignal'] = np.array(im_circSignal)
            cs.guistuff['PureSNR_circlesNoise'] = np.array(guicirclesNoise)
            cs.guistuff['PureSNR_circlesSignal'] = np.array(guicirclesSignal)
        
        print('Done!')
            
        return error, self.results
    
    def ReceiveMapping(self, csFA, csB1):
        """
        Determine receive mapping from Tx*Rx/Tx
        Needs 2 scans: 3DFFE_FA1 (small flipangle) and B1map.
        """
        error   = False
        
        # B1 map images contain the nominal flip angle expressed as percentage 
        # of the flip angle chosen by the operator

        infoB1       = csB1.dcmInfile._datasets # a stack of all dicom files (headers and data)
        imagesB1     = csB1.pixeldataIn         # a stack of just the dicom data
        self.verbose = csFA.verbose

        # Data contains images with the B1 map, magnitude images, ... . Select 
        # B1maps, ignore other data. This should result in 1/4 of the loaded
        # images!

        # filter on ImageType,'ORIGINAL\PRIMARY\M_B1\M\B1' Alternatively, filter on ScanningSequence 'B1'
        desc = np.array([ '\\'.join(im.ImageType) for im in infoB1 ])
        keep = 'ORIGINAL\\PRIMARY\\M_B1\\M\\B1'
        imagesB1 = imagesB1[ desc == keep ]
        infoB1 = np.array(infoB1)[ desc == keep ]

        # now repeat for FA images
        infoFA       = csFA.dcmInfile._datasets 
        imagesFA     = csFA.pixeldataIn 

        # select only images of FA with a matching slicepos in B1
        # Slice Location: (0020,1041). Relative position of the image plane expressed in mm. 
        # SliceLocation may differ on a sub mm scale, so just pick closest one
        sliceLocB1 = [ im.SliceLocation for im in infoB1 ]
        sliceLocFA = [ im.SliceLocation for im in infoFA ]
        selectionFA = []
        for p0 in sliceLocB1:
            selectionFA.append(  np.argmin(np.array( [ abs(p1-p0) for p1 in sliceLocFA])) )
        infoFA = np.array(infoFA)[selectionFA]
        imagesFA = imagesFA[selectionFA]

        guicirclesFA = [] # (x,y,rad) for gui
        im_circFA   = np.array(mmath.CircleImages(imagesFA, self.usePartCircle, guicirclesFA))
        
        # normalize FA and B1 on mean within mask
        meanB1 = np.ma.array(imagesB1, mask=~im_circFA).mean()
        meanFA = np.ma.array(imagesFA, mask=~im_circFA).mean()

        imagesB1 /= meanB1
        imagesFA /= meanFA
 
        # Divide both (normalized) datasets to make a map of the receive field : Tx*Rx / Tx = Rx
        im_Receive = imagesFA/imagesB1*im_circFA
        im_Receive[imagesB1<1e-12] = 0

        # Calculate output parameters: mean, std, (max-min)/(max+min). These are
        # all dimensionless parameters from which we will have to learn what their
        # future value might be
        mask_Receive = im_Receive>0
        ma_Receive = np.ma.array(im_Receive, mask=~mask_Receive)
        receive_mean = ma_Receive.mean()
        receive_std  = ma_Receive.std()
        receive_min  = ma_Receive.min()
        receive_max  = ma_Receive.max()
        receive_nonuni = (receive_max-receive_min)/(receive_max+receive_min)

        self.results.append( ('float', 'receive_mean', receive_mean) )
        self.results.append( ('float', 'receive_std', receive_std) )
        self.results.append( ('float', 'receive_min', receive_min) )
        self.results.append( ('float', 'receive_max', receive_max) )
        self.results.append( ('float', 'receive_nonuni_pct', 100.*receive_nonuni) )

        frame = 0 # just select a frame for plotting
        self.makePlots('Receivemapping', {
            'im_B1_frame':imagesB1[frame],
            'im_FA_frame':imagesFA[frame],
            'im_Receive_frame':im_Receive[frame],
            'circle_frame':im_circFA[frame],
            })

        csFA.hasmadeplots = True
        if self.verbose:
            plt.show()

        # make some stuff accessible for gui
        if self.guimode:
            csFA.guistuff['Receive_B1_norm'] = np.array(imagesB1)
            csFA.guistuff['Receive_FA_norm'] = np.array(imagesFA)
            csFA.guistuff['Receive_FA_mask'] = np.array(im_circFA)
            csFA.guistuff['Receive_FA_circles'] = guicirclesFA
            csFA.guistuff['Receive_map']  = np.array(im_Receive)

        print('Done!')

        return error, self.results

