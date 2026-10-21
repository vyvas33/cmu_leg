#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm as np_norm
from collections import deque
from math import sqrt, isfinite


######################################################################
class ZUPT_AdaptiveGLRT:
    """
    General Likelihood Ratio Threshold Detector of stationary IMU

    (follows Skog et al, Zero-Velocity Detection—An Algorithm 
     Evaluation, IEEE Trans Biomed Eng, 2010 and
     Tao et al, An enhanced foot‐mounted PDR method with adaptive ZUPT
     and multi‐sensors fusion for seamless pedestrian navigation, 
     GPS Solutions, 2022 for the test threshold adaptation)

     For general introduction to GLRT, consult "Fundamentals of 
     Statistical Signal Processing" Volumes I (Estimation Theory, 
     1993) and II (Detection Theory, 1998) by Steven M. Kay

    Inputs:
        sigma_acc, sigma_gyr: standard deviations of acceleration
            and gyroscope measurements
        hist_len: Number of samples in test sequence. Should be 
            chosen such that the corresponding time interval of the 
            sequence is about 150ms.
        adapt_hist_len: Number of samples in test function history
            used for determining min and max test function values. 
            Should be chosen such that the corresponding time interval
            is several seconds to cover a few locomotion steps.
        min_offset: Offset from minimum of test function, defining
            current detection threshold (= min + offset * (max - min))

    Output:
        is_stationary: Flag indicating whether (TRUE) or not (FALSE)
            the IMU is stationary.                        
    """

    __G = 9.81  # [m/s^2] gravitational acceleration

    # PERFORMANCE NOTE (this class runs once per ESKF tick, in the 500 Hz
    # control loop, and measured 23% of ESKF.update on the Jetson):
    #
    # The windows are CIRCULAR ndarrays, not deques of little (3,) arrays. A
    # deque forces numpy to rebuild an (N,3) array on every element-wise
    # expression touching it -- `np.sum(hist, axis=0)`, `hist - g_prj` and
    # `norm(hist, axis=1)` each paid that conversion separately, three times a
    # call. Writing one row into a preallocated array costs nothing and the
    # expressions then run straight off the buffer.
    #
    # Every statistic here is a SUM over the whole window, so it is invariant
    # to row order and the ring's rotating start point changes no result.
    #
    # The adaptation history is an ndarray for the same reason: `min()`/`max()`
    # over a 1000-element deque is 2000 interpreter-level comparisons per tick
    # to track the extremes of a window that moves by one sample. ndarray
    # .min()/.max() are the same two passes in C and are exact regardless of
    # order -- with ONE behavioural difference around nan, handled at the write
    # site below.

    def __init__(self, hist_len, adapt_hist_len, sigma_acc, sigma_gyr,
                 min_offset):
        self.__N = hist_len
        self.__var_acc = sigma_acc ** 2
        self.__var_gyr = sigma_gyr ** 2

        self.__acc_hist = np.zeros((hist_len, 3))
        self.__gyr_hist = np.zeros((hist_len, 3))
        self.__i = 0                      # next window slot to overwrite

        self.__offset = min_offset
        self._thrshld = 0.0  # initialize threshold to zero
        self._test_fcn_hist = np.zeros(adapt_hist_len)
        self.__j = 0                      # next adaptation slot

        self.is_stationary = False

    def update(self, acc, gyr):
        N = self.__N
        acc_hist = self.__acc_hist
        gyr_hist = self.__gyr_hist

        # update rolling histories (ring write, no reallocation)
        i = self.__i
        acc_hist[i] = acc
        gyr_hist[i] = gyr
        self.__i = (i + 1) % N

        # compute current acc and gyr terms.
        # sum_k ||x_k||^2 is written as (X*X).sum() rather than
        # sum(norm(X, axis=1)**2): same quantity, but it skips the sqrt that
        # the square immediately undoes, and keeps the reduction in C instead
        # of iterating the rows with the builtin sum().
        acc_mean_vec = acc_hist.sum(axis=0) / N
        # norm==0 while the window is still all zeros -> nan, exactly as the
        # np.linalg.norm version did (eskf_replay_validation suppresses it).
        g_prj = self.__G * acc_mean_vec / sqrt(acc_mean_vec @ acc_mean_vec)  # <1>
        dev = acc_hist - g_prj
        acc_term = (dev * dev).sum()
        gyr_term = (gyr_hist * gyr_hist).sum()

        # perform general likelihood ratio test
        test_fcn = 1 / self.__N * \
            (acc_term / self.__var_acc + gyr_term / self.__var_gyr)
        self.is_stationary = bool(test_fcn < self._thrshld)

        # adapt threshold.
        #
        # Non-finite test functions are kept OUT of the history. They happen
        # for real: the window is all zeros before the IMU streams (and the
        # first rows of a logged trial are literally [0,0,0]), so the mean acc
        # vector has zero norm and g_prj comes out nan.
        #
        # This is not a new rule, it is the old one made explicit. The previous
        # implementation appended the nan to a deque and then took builtin
        # min()/max() over it -- and since every comparison against nan is
        # False, the nan was silently skipped UNLESS it happened to be the
        # first element, in which case min()/max() returned it. So the old
        # behaviour was "drop it, usually", with a position-dependent
        # exception. ndarray.min()/.max() instead propagate nan, which would
        # poison the threshold (and suppress ZUPT entirely) for the ~1000 ticks
        # the sample takes to age out. Skipping the write keeps the intended
        # behaviour, deterministically, and leaves the reduction a plain C pass.
        # The slot is still CONSUMED when the value is dropped, so the window
        # keeps its phase -- the old deque advanced on every call too, and a
        # skipped write here leaves the slot holding what a nan contributed to
        # builtin min()/max(): nothing.
        hist = self._test_fcn_hist
        j = self.__j
        if isfinite(test_fcn):
            hist[j] = test_fcn
        self.__j = (j + 1) % hist.size
        fcn_min = hist.min()
        fcn_max = hist.max()
        self._thrshld = fcn_min + self.__offset * (fcn_max - fcn_min)

        return self.is_stationary

# --------------------------------------------------------------------
# <1>: Projection of gravity along axis of mean acc vector.


######################################################################
class ZUPT_GLRT():
    """
    General Likelihood Ratio Threshold Detector of stationary IMU

    (follows Skog et al, Zero-Velocity Detection—An Algorithm 
     Evaluation, IEEE Trans Biomed Eng, 2010)

     For general introduction to GLRT, consult "Fundamentals of 
     Statistical Signal Processing" Volumes I (Estimation Theory, 
     1993) and II (Detection Theory, 1998) by Steven M. Kay

    Inputs:
        sigma_acc, sigma_gyr: standard deviations of acceleration
            and gyroscope measurements
        hist_len: Number of samples in test sequence. Should be 
            chosen such that the corresponding time interval of the 
            sequence is about 150ms.
        thrshld: detection threshold (between zero and one with higher
            threshold indicating more confident detection; in practice,
            however, threshold chosen by looking at test function and
            then picking a threshold that is reasonably close to 
            minima. Another problem that makes plain GLRT for ZUPT 
            difficult is that the minima depend on walking speed and
            gait. Therefore, ZUPT_AdaptiveGLRT is more recommended,
            and this plain ZUPT_GLRT class is retained here for 
            comparison only.
    Output:
        is_stationary: Flag indicating whether (TRUE) or not (FALSE)
            the IMU is stationary.                        
    """

    __G = 9.81   # [m/s^2] gravitational acceleration

    def __init__(self, hist_len, sigma_acc, sigma_gyr, thrshld):
        self.__N = hist_len
        self._thrshld = - 2 / self.__N * np.log(thrshld)  # <1>
        self.__var_acc = sigma_acc ** 2
        self.__var_gyr = sigma_gyr ** 2

        self.__acc_hist = deque(np.zeros((self.__N, 3)))
        self.__gyr_hist = deque(np.zeros((self.__N, 3)))

        self._test_fcn = 0.0
        self.is_stationary = False

    def update(self, acc, gyr):
        # update rolling histories
        self.__acc_hist.popleft();  self.__acc_hist.append(acc)
        self.__gyr_hist.popleft();  self.__gyr_hist.append(gyr)

        # compute current acc and gyr terms
        acc_means = np.sum(self.__acc_hist, axis=0) / self.__N
        g_prj = self.__G * acc_means / np_norm(acc_means)        # <2>
        acc_term = sum(np_norm(self.__acc_hist - g_prj, axis=1) ** 2)

        gyr_term = sum(np_norm(self.__gyr_hist, axis=1) ** 2)

        # perform general likelyhood ratio test
        self._test_fcn = 1 / self.__N * \
            (acc_term / self.__var_acc + gyr_term / self.__var_gyr)
        self.is_stationary = (self._test_fcn < self._thrshld)

        return self.is_stationary

# --------------------------------------------------------------------
# <1>: Transformed likelihood ratio threshold (compare paper).
# <2>: Projection of gravity along axis of mean acc vector.


######################################################################
class EMFU_GLRT():
    """
    General Likelihood Ratio Threshold Detector of consistent earth 
    magnetic field.

    (similar to acceleration portion of Skog et al, Zero-Velocity 
     Detection—An Algorithm Evaluation, IEEE Trans Biomed Eng, 2010)

    Inputs:
        sigma_mag: standard deviations of magnetometer measurements
        hist_len: Number of samples in test sequence. Should be 
            chosen such that the corresponding time interval of the 
            sequence is about 150ms.
        thrshld: detection threshold (between zero and one with higher
            threshold indicating more confident detection.
    Output:
        is_earth_mag: Flag indicating whether (TRUE) or not (FALSE)
            consistent earth magnetic field detected.       
    """
    _B = 0.4969  # [Gauss] strength of earth magnetic field  # <1>

    def __init__(self, hist_len, sigma_mag, thrshld):
        self.__var = sigma_mag ** 2
        self.__N = hist_len                                      # <2>
        self.__mag_hist = deque(np.zeros((self.__N, 3)))

        self._test_fcn = 0.0
        self._thrshld = - 2 / self.__N * np.log(thrshld)         # <3>
        self.is_earth_mag = False

    # perform general likelihood ratio test
    def update(self, mag):
        self.__mag_hist.popleft();  self.__mag_hist.append(mag)

        mag_means = np.sum(self.__mag_hist, axis=0) / self.__N
        prj = self._B * mag_means / np_norm(mag_means)           # <4>
        self._test_fcn = 1 / (self.__N * self.__var) * \
            sum(np_norm(self.__mag_hist - prj, axis=1) ** 2)

        self.is_earth_mag = (self._test_fcn < self._thrshld)

        return self.is_earth_mag

# --------------------------------------------------------------------
# <1>: Location dependent. Can also use IMU readings over calibration
#      phase such as first 2s to extract B.
# <2>: Number of samples in test sequence. Should be chosen such that
#      the corresponding time interval of the sequence is on the order
#      of about 150ms.
# <3>: Transformed likelihood ratio threshold (compare paper).
# <4>: Projection of field reference strength along axis of
#      sensor signal mean vector.


######################################################################
class ZARU_GLRT():
    """
    General Likelihood Ratio Threshold Detector of Zero Angular Rate

    (similar to acceleration portion of Skog et al, Zero-Velocity 
     Detection—An Algorithm Evaluation, IEEE Trans Biomed Eng, 2010)

    Inputs:
        sigma_gyr: standard deviations of gyroscope measurements
        hist_len: Number of samples in test sequence. Should be 
            chosen such that the corresponding time interval of the 
            sequence is about 150ms.
        thrshld: detection threshold (between zero and one with higher
            threshold indicating more confident detection.
    Output:
        is_zero__angular_rate: Flag indicating whether (TRUE) or 
        not (FALSE) the gyro is still.                        
    """

    def __init__(self, hist_len, sigma_gyr, thrshld):
        self.__N = hist_len
        self._thrshld = - 2 / self.__N * np.log(thrshld)  # <1>
        self.__var_gyr = sigma_gyr ** 2

        self.__gyr_hist = deque(np.zeros((self.__N, 3)))

        self._test_fcn = 0.0
        self.is_zero_rate = False

    def update(self, gyr):
        # update rolling history
        self.__gyr_hist.popleft();  self.__gyr_hist.append(gyr)

        # perform general likelihood ratio test
        term = sum(np_norm(self.__gyr_hist, axis=1) ** 2)
        self._test_fcn = 1 / self.__N * term / self.__var_gyr
        self.is_zero_rate = (self._test_fcn < self._thrshld)

        return self.is_zero_rate

# --------------------------------------------------------------------
# <1>: Transformed likelihood ratio threshold (compare paper).
