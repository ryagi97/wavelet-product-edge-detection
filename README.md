# wavelet-product-edge-detection
A python implementation of Zhang and Bao's wavelet domain scale multiplication edge detector. 

## Longer Description
Functions for implementing the edge detection scheme first proposed by Zhang and Bao [1]. 
Modified for use with pywt's SWT2 transform and employs double thresholding similar to canny to improve noise resilience and revovery of weak edges.

Portions of code adapted from scikit-image's implementation of the canny edge detector;
    
    Title: canny.py - Canny Edge detector
    Author: Lee Kamentsky
    Date: 11/02/2020
    Code version: 0.17.2
    Availability: https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/_canny.py 


[1] Zhang, L. and Bao, P., 2002. Edge detection by scale multiplication in wavelet domain. Pattern Recognition Letters, 23(14), pp.1771-1784.
