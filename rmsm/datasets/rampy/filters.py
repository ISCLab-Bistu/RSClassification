#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy import signal
import scipy.sparse as sparse
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def smooth(x, y, method="whittaker", **kwargs):
    """smooth the provided y signal (sampled on x)

    Parameters
    ----------
    x : ndarray
        Nx1 array of x values (equally spaced).
    y : ndarray
        Nx1 array of y values (equally spaced).
    method : str
        Method for smoothing the signal;
        choose between savgol (Savitzky-Golay), GCVSmoothedNSpline, MSESmoothedNSpline, DOFSmoothedNSpline, whittaker, likelihood , flat, hanning, hamming, bartlett, blackman.
    window_length : int, optional
        The length of the filter window (i.e. the number of coefficients). window_length must be a positive odd integer.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. polyorder must be less than window_length.
    Lambda : float, optional
        smoothing parameter of the Whittaker filter described in Eilers (2003). The higher the smoother the fit.
    d : int, optional
        d parameter in Whittaker filter, see Eilers (2003).
    ese_y : ndarray, optional
        errors associated with y (for the gcvspline algorithms)

    Returns
    -------
    y_smo : ndarray
        smoothed signal sampled on x.

    Notes
    -----

    Use of GCVSmoothedNSpline, MSESmoothedNSpline, DOFSmoothedNSpline requires installation of gcvspline. See gcvspline documentation.
    See also documentation for details on GCVSmoothedNSpline, MSESmoothedNSpline, DOFSmoothedNSpline.

    savgol uses the scipy.signal.savgol_filter() function.

    References
    ----------
    Eilers, P.H.C., 2003. A Perfect Smoother. Anal. Chem. 75, 3631–3636. https://doi.org/10.1021/ac034173t

    Scipy Cookbook: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=smooth

    """
    window_len = kwargs.get("window_length", 5)
    polyorder = kwargs.get("polyorder", 2)
    lam = kwargs.get("Lambda", 10.0 ** 5)
    d = kwargs.get("d", 2)
    ese_y = kwargs.get("ese_y", 1.0)

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if not method in ['GCVSmoothedNSpline', 'MSESmoothedNSpline', 'DOFSmoothedNSpline', 'whittaker', 'savgol',
                      'likelihood', 'exp', 'interpolation', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Method should be on 'GCVSmoothedNSpline','MSESmoothedNSpline','DOFSmoothedNSpline','whittaker','savgol','flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if (method == "GCVSmoothedNSpline") or (method == "MSESmoothedNSpline") or (
            method == "DOFSmoothedNSpline"):  # gcvspline methods
        try:  # we test if gcvsppipline is installed
            import gcvspline
        except ImportError:
            print('ERROR: Install gcvspline to use this smoothing mode (needs a working FORTRAN compiler).')

        w = 1.0 / (np.ones((y.shape[0], 1)) * ese_y)  # errors

        if method == "GCVSmoothedNSpline":
            flt = gcvspline.GCVSmoothedNSpline(x.reshape(-1), y.reshape(-1), w.reshape(-1))
        elif method == "MSESmoothedNSpline":
            flt = gcvspline.MSESmoothedNSpline(x.reshape(-1), y.reshape(-1), w.reshape(-1))
        elif method == "DOFSmoothedNSpline":
            flt = gcvspline.DOFSmoothedNSpline(x.reshape(-1), y.reshape(-1), w.reshape(-1))
        return flt(x)
    elif method == "whittaker":  # whittaker smoother
        return whittaker(y, Lambda=lam, d=d)
    elif method == "interpolation":
        # Create the interpolation function
        interpolation_function = interp1d(x, y, kind='cubic', fill_value=(y[0], y[-1]), bounds_error=False)

        # Denser wavelength points are generated
        new_wavelength = len(y)
        wavelength_dense = np.linspace(min(x), max(y), new_wavelength)

        # An interpolation function is used to generate a smooth curve
        intensity_smooth = interpolation_function(wavelength_dense)

        return intensity_smooth
    elif method == "savgol":  # Savtisky-Golay filter
        return scipy.signal.savgol_filter(y, window_len, polyorder)
    elif method == "likelihood":
        # Initial guess for window length and polynomial order
        initial_guess = [window_len, polyorder]

        # Use MLE to find the optimal window length and polynomial order
        result = minimize(neg_log_likelihood, initial_guess, args=(x, y), method='Nelder-Mead')
        optimal_window, optimal_order = result.x

        # Apply Savitzky-Golay smoothing with the optimal parameters
        smoothed_y = savgol_filter(y, window_length=int(optimal_window), polyorder=int(optimal_order))
        return smoothed_y
    elif method in frozenset(('flat', 'hanning', 'hamming', 'bartlett',
                              'blackman')):  # various window filters, from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html?highlight=smooth

        s = np.r_[y[window_len - 1:0:-1], y, y[-2:-window_len - 1:-1]]
        if method == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = getattr(np, method)(window_len)

        y_filt = np.convolve(w / w.sum(), s, mode='valid')
        shift = int((len(y_filt) - len(y)) / 2)

        return y_filt[shift:-shift]


def exponential_smoothing(data, alpha):
    smoothed_data = [data[0]]  # Initialize the first smooth point as the first point of the original data

    for i in range(1, len(data)):
        smoothed_point = alpha * data[i] + (1 - alpha) * smoothed_data[-1]
        smoothed_data.append(smoothed_point)

    return np.array(smoothed_data)


def whittaker(y, **kwargs):
    """smooth a signal with the Whittaker smoother

    Parameters
    ----------
    y : ndarray
        An array with the values to smooth (equally spaced).
    Lambda : float, optional
        The smoothing coefficient, the higher the smoother. Default = 10^5.

    Returns
    -------
    z : ndarray
        An array containing the smoothed values.

    References
    ----------
    P. H. C. Eilers, A Perfect Smoother. Anal. Chem. 75, 3631–3636 (2003).

    """
    # optional parameters
    lam = kwargs.get('Lambda', 1.0 * 10 ** 5)

    # starting the algorithm
    L = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    W = scipy.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = scipy.sparse.linalg.spsolve(Z, w * y)

    return z


def spectrafilter(spectre, filtertype, fq, numtaps, columns):
    """Filter specific frequencies in spectra with a butterworth filter

    Parameters
    ----------
    spectre : ndarray
        Array of X-Y values of spectra. First column is X and subsequent n columns are Y values of n spectra. (see also spectraarray function)
    filtertype : string
        type of filter; Choose between 'low', 'high', 'bandstop', 'bandpass'.
    fq : ndarray
        Frequency of the periodic signal you try to erase. If using a bandpass or band stop filter, fq must be an array containing the cutoff frequencies.
    columns : ndarray
        An array defining which columns to treat.

    Returns
    -------
    out : ndarray
        filtered signals.

    """
    out = np.zeros(spectre.shape)  # output array
    out[:, 0] = spectre[:, 0]  # record x axis

    # Butterworth band stop filter caracteristics
    a = spectre[1, 0] - spectre[0, 0]
    samplerate = 1 / a  # Hertz
    nyq_rate = samplerate / 2  # Nyquist frequency
    cutf = fq  # cutoff frequency
    # bandwidth = 0.005 # largeur filtre, for band pass/stop filters
    numtaps = 1  # filter order

    for i in range(len(columns)):
        y = spectre[:, columns[i]]
        if (filtertype == 'low') or (filtertype == 'high'):
            b, a = signal.butter(numtaps, [(cutf / nyq_rate)], btype=filtertype)
        else:
            b, a = signal.butter(numtaps, [(cutf[0] / nyq_rate), (cutf[1] / nyq_rate)], btype=filtertype)

        out[:, columns[i]] = signal.filtfilt(b, a, y)  # filter with phase shift correction

    return out


# Define a function to calculate the negative log likelihood
def neg_log_likelihood(params, x, y):
    window, order = params
    window = int(window)
    order = int(order)
    smoothed_y = savgol_filter(y, window_length=window, polyorder=order)
    residuals = y - smoothed_y
    sigma = np.std(residuals)
    return len(x) * np.log(sigma) + np.sum(residuals ** 2) / (2 * sigma ** 2)
