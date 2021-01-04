 # -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:10:29 2021

Author: Ryan Agius

Functions for implementing the edge detection scheme first proposed by Zhang and Bao [1]. 
Modified for use with pywt's SWT2 transform and employs double thresholding similar to canny to improve noise resilience and revovery of weak edges.

Portions of code adapted from scikit-image's implementation of the canny edge detector;
    
    Title: canny.py - Canny Edge detector
    Author: Lee Kamentsky
    Date: 11/02/2020
    Code version: 0.17.2
    Availability: https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py 


[1] Zhang, L. and Bao, P., 2002. Edge detection by scale multiplication in wavelet domain. Pattern Recognition Letters, 23(14), pp.1771-1784.
"""

import numpy as np
from pywt import swt2, Wavelet
from scipy.ndimage import generate_binary_structure, binary_erosion, label
from scipy import ndimage as ndi

def wavelet_edge_detector(image, start_level=0, levels=2, wavelet='rbio3.1', c=0.15, noise_var=40, t1=1, t2=2, dbl_th=True):

    """
    Extracts the edge local maxima of the passed image using the product of two
    consecutive stationary wavelet coefficients
    -----------
    image : 2D array
        Input image, grayscale
    start_level : int
        Initial coefficient scale level to be extracted by the SWT
    levels : int
        number of levels to consider, must be even
    wavelet : string
        Name of wavelet as listed by pywt.wavelist()
    c : float
        Multiplier for calculating the threshold
    noise_var : float
        Estimate of the Gaussian Noise variance present in the image
    t1 : float
        Threshold multiplier for the lower threshold
    t2 : float
        Threshold multiplier for the lower threshold
    Returns
    -------
    local_maxima : 2D array 
        local maxima extracted by the local maxima method
    edge_mask : 2D array 
        Binary array marking edges present in the local maxima
    -----
    
    """
    
    assert(levels%2 == 0)
    
    #calculate the maximum level to decompose the image with
    max_level = start_level+levels

    #Decompse the image to its detail coefficients using the 2D SWT
    coeffs = swt2(image, wavelet=wavelet, level=max_level,
                  start_level=start_level, norm=False,
                  trim_approx=True)
    
    #create empty arrays to store the detail coefficients
    #algoritmhs only require Horizontal and Vertical details, so Diagonal is not calculated
    coeff_arr_H = np.empty((image.shape + (max_level-start_level,)))
    coeff_arr_V = np.empty((image.shape + (max_level-start_level,)))

    #offset the coefficients based on the decomposition scale
    for i in range(max_level-start_level):
        coeff_arr_H[:,:,i]  = np.roll(coeffs[-1-i][0], 2**(i+start_level))
        coeff_arr_V[:,:,i]  = np.roll(coeffs[-1-i][1], 2**(i+start_level))

    #Get the Horizontal and Vertical products; the magnitude gradient matrices
    Mdx = np.prod(coeff_arr_H, axis=2)
    Mdy = np.prod(coeff_arr_V, axis=2)

    #Remove negative coefficients, as these are solely due to noise
    pts_Mdx_plus = (Mdx >= 0)
    Mdx = pts_Mdx_plus * Mdx

    pts_Mdy_plus = (Mdy >= 0)
    Mdy = pts_Mdy_plus * Mdy
    
    #Get the angle gradient matrices
    Adx = np.sign(coeff_arr_H[:,:,1])*np.sqrt(Mdx)
    Ady = np.sign(coeff_arr_V[:,:,1])*np.sqrt(Mdy)
    
    #Obtain the local modulus maximum in the direction of the normal of the edge
    local_maxima = local_modulus_maxima(Adx, Ady, Mdx, Mdy)
    
    if dbl_th:
        #Perform double thresholding and return the edge mask
        edge_mask = dbl_thresholding_ZhangBao(local_maxima, wavelet=wavelet,
                                              start_level=start_level,
                                              c=c, noise_var=noise_var,
                                              t1=t1, t2=t2)
    else:
        edge_mask = None

    return local_maxima, edge_mask
    

def local_modulus_maxima(Adx, Ady, Mdx, Mdy, mask=None):
    
    """ 
    Code adapted from scikit-image's canny implementation for faster execution
    
    Title: canny.py - Canny Edge detector
    Author: Lee Kamentsky
    Date: 11/02/2020
    Code version: 0.17.2
    Availability: https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py 
    """
    
    
    """Fast computation of the local maxima using custom gradient and angle matrices
    Parameters
    -----------
    Adx : 2D array
        Gradient array along axis 0 (Horizontal Detail Coefficients) to be used
        for calculating the normal to the edges
    Ady : 2D array
        Gradient array along axis 1 (Vertical Detail Coefficients) to be used
        for calculating the normal to the edges
    Mdx : 2D array
        Gradient array along axis 0 (Horizontal Detail Coefficients) to be used
        for calculating the value of the edges
    Mdy : 2D array
        Gradient array along axis 1 (Vertical Detail Coefficients) to be used
        for calculating the value of the edges
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.

    Returns
    -------
    output : 2D array 
        The local maxima
    -----
    
    The steps of the algorithm are as follows:
    * Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.
    """
    #
    # The steps involved:
    #
    # * Find the normal to the edge at each point using the arctangent of the
    #   ratio of the Y sobel over the X sobel - pragmatically, we can
    #   look at the signs of X and Y and the relative magnitude of X vs Y
    #   to sort the points into 4 categories: horizontal, vertical,
    #   diagonal and antidiagonal.
    #
    # * Look in the normal and reverse directions to see if the values
    #   in either of those directions are greater than the point in question.
    #   Use interpolation to get a mix of points instead of picking the one
    #   that's the closest to the normal.
    #
    assert (Mdx.shape == Mdy.shape)
    assert (Mdx.shape == Adx.shape)
    assert (Adx.shape == Ady.shape)

    if mask is None:
        mask = np.ones(Mdx.shape, dtype=bool)

    jsobel = Ady
    isobel = Adx
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)
    magnitude = np.hypot(Mdx, Mdy)
    
    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    s = generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)
    #
    #--------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(Mdx.shape)
    #----- 0 to 45 degrees ------
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    return local_maxima * magnitude

def dbl_thresholding_ZhangBao(local_maxima,  start_level=0, wavelet='rbio3.1', c=20, noise_var=1, t1=1, t2=2):
    """ 
    Portions of code adapted from scikit-image's canny implementation for faster execution
    
    Title: canny.py - Canny Edge detector
    Author: Lee Kamentsky
    Date: 11/02/2020
    Code version: 0.17.2
    Availability: https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py 
    """
    
    """Performs double thresholding based the wavelet energy and noise variance values
    Parameters
    -----------
    local_maxima : 2D array
        Local maxima extracted by the local maxima method, same shape as input image
    wavelet : string
        Name of wavelet as listed by pywt.wavelist()
    start_level : int
        Initial coefficient scale level to be extracted by the SWT
    c : float
        Multiplier for calculating the threshold
    noise_var : float
        Estimate of the Gaussian Noise variance present in the image
    t1 : float
        Threshold multiplier for the lower threshold
    t2 : float
        Threshold multiplier for the lower threshold
    Returns
    -------
    edge_mask : 2D array 
        Binary array marking edges present in the local maxima
    -----
    
    """  
    #
    #---- Create two masks at the two thresholds.
    #
    
    # * Label all points above the high threshold as edges.
    # * Recursively label any point above the low threshold that is 8-connected
    #   to a labeled point as an edge.
    #
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    #
    
    #First lower threshold is the same as in Zhang and Bao's paper
    #Set to remove the majority of the noise present
    
    #threshold = c * energy of wavelet at scale j, energy at scale j+1,
    #noise_var, scaled noise_var
    
    #get wavelet coefficients
    w = Wavelet(wavelet)
    
    if w.orthogonal:
        (_, psi_d1, _)  = w.wavefun(level=start_level+1)
        (_, psi_d2, _)  = w.wavefun(level=start_level+2)
    else:
        (_, psi_d1, _, _, _)  = w.wavefun(level=start_level+1)
        (_, psi_d2, _, _, _)  = w.wavefun(level=start_level+2)
        
    #compute their enegries (in reality, square root of energy)
    energy_psi_d1 = np.sqrt(np.sum(psi_d1**2))
    energy_psi_d2 = np.sqrt(np.sum(psi_d2**2))
    
    #add zeros to psi_d1 to compute the next variable
    psi_d1_up = psi_d1.repeat(2)
    psi_d1_up[1::2] = 0
    
    if wavelet == 'haar':
        psi_d1_up = psi_d1_up[1:-1]
    
    #get the sigma_i value
    sigma_i_sq = 2*np.sum((psi_d1_up/energy_psi_d1 + psi_d2/energy_psi_d2)**2)
    
    t = c * energy_psi_d1 * energy_psi_d2 * noise_var * sigma_i_sq
    T_low = t*t1
    T_high = t*t2
    
    high_mask = (local_maxima >= T_high)
    low_mask = (local_maxima >= T_low)
    
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels, np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    
    return output_mask

#run demo
if __name__ == "__main__":
    import cv2 as cv
    
    lvl = 0
    c = 0.345
    t1 = 1.0
    t2 = 2.75
    noise_var = 7237.754103671255
        
    cv.namedWindow('Camera Capture', cv.WINDOW_NORMAL)
    cv.namedWindow('Product Local Maxima - Haar Wavelet', cv.WINDOW_NORMAL)
    cv.namedWindow('Product Local Maxima - Reverse Biorthogonal 3.1 Wavelet', cv.WINDOW_NORMAL)
    cv.namedWindow('Edges - Haar Wavelet', cv.WINDOW_NORMAL)
    cv.namedWindow('Edges - Reverse Biorthogonal 3.1 Wavelet', cv.WINDOW_NORMAL)
    cv.namedWindow('Overlay - Haar Wavelet', cv.WINDOW_NORMAL)
    cv.namedWindow('Overlay - Reverse Biorthogonal 3.1 Wavelet', cv.WINDOW_NORMAL)
    
    image = cv.imread('test_images/USAF.tiff', cv.IMREAD_GRAYSCALE)
    
    #convert image from 8-bit to 12-bit, same as camera depth
    image = image.astype(np.float)
    image = image * 4095/256
    image = image.astype(np.uint16)
    
    #find local maxima and edges using the Haar wavelet
    local_maxima_hr, edges_hr = wavelet_edge_detector(image, start_level=lvl,
                                            wavelet='haar',c=c,
                                            noise_var=noise_var, t1=t1, t2=t2)
    
    local_maxima_hr = local_maxima_hr / np.max(local_maxima_hr) * 65535
    local_maxima_hr = local_maxima_hr.astype(np.uint16)
    
    edges_hr = edges_hr * np.ones(edges_hr.shape) * 65535
    edges_hr = edges_hr.astype(np.uint16)
    
    comb_hr = np.zeros((image.shape + (3,)))
    comb_hr[:,:,0] = image / 4096
    comb_hr[:,:,1] = comb_hr[:,:,0]
    comb_hr[:,:,2] = comb_hr[:,:,0]
    
    comb_hr[:,:,2] += (edges_hr/65535)
    
    comb_hr[:,:,2]  = np.clip(comb_hr[:,:,2], 0, 1)
    
    #find local maxima and edges using the Reverse Biorthogonal 3.1 wavelet
    local_maxima_rb, edges_rb = wavelet_edge_detector(image, start_level=lvl,
                                            wavelet='rbio3.1',c=c,
                                            noise_var=noise_var, t1=t1, t2=t2)
    
    local_maxima_rb = local_maxima_rb / np.max(local_maxima_rb) * 65535
    local_maxima_rb = local_maxima_rb.astype(np.uint16)
    
    edges_rb = edges_rb * np.ones(edges_rb.shape) * 65535
    edges_rb = edges_rb.astype(np.uint16)
    
    comb_rb = np.zeros((image.shape + (3,)))
    comb_rb[:,:,0] = image / 4096
    comb_rb[:,:,1] = comb_rb[:,:,0]
    comb_rb[:,:,2] = comb_rb[:,:,0]
    
    comb_rb[:,:,2] += (edges_rb/65535)
    
    comb_rb[:,:,2]  = np.clip(comb_rb[:,:,2], 0, 1)
    
    image = image.astype(np.float)
    image = image * 65535/4096
    image = image.astype(np.uint16)
    
    try:
        while True:
            cv.imshow('Camera Capture', image)
            cv.imshow('Product Local Maxima - Haar Wavelet', local_maxima_hr)
            cv.imshow('Product Local Maxima - Reverse Biorthogonal 3.1 Wavelet', local_maxima_rb)
            cv.imshow('Edges - Haar Wavelet', edges_hr)
            cv.imshow('Edges - Reverse Biorthogonal 3.1 Wavelet', edges_rb)
            cv.imshow('Overlay - Haar Wavelet', comb_hr)
            cv.imshow('Overlay - Reverse Biorthogonal 3.1 Wavelet', comb_rb)
            
            cv.waitKey(1)
            
    except KeyboardInterrupt:
        cv.destroyAllWindows()
    
    
    
