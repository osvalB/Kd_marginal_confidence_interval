import numpy            as np
from scipy              import stats


def calc_residual_sum_of_squares(y_exp,y_fit):

    """
    Parameters
    ----------
    y_exp : list or np.array
        Measured data.
    y_fit : list or np.array
        Fitted data.

    Returns
    -------
    np.ndarray
        Residual sum of squares between measured and fitted data.
    """

    # Convert to numpy arrays if needed
    if type(y_exp) != np.ndarray:
        y_exp = np.array(y_exp)

    if type(y_fit) != np.ndarray:
        y_fit = np.array(y_fit)

    residuals = y_exp - y_fit

    sq_residuals = residuals**2

    sum_sq_residuals = np.sum(sq_residuals)

    return sum_sq_residuals

def calc_critical_value(alpha,n,p):

    """
    Calculate the critical value for the confidence interval.
    It corresponds to the Fisher–Snedecor distribution F_df1,df2 with df1 = 1 and df2 = n - p degrees of freedom.

    Parameters
    ----------
    alpha : float
        One minus alpha gives the confidence level.
    n : int
        Number of data points.
    p : int
        Number of parameters.

    Returns
    -------

    float
        The critical value.

    """

    critical_value = stats.f.ppf(q=1-alpha, dfn=1, dfd=n-p)

    return critical_value

def calc_residual_sum_of_squares_profile(rss0,alpha,n,p):

    """
    Useful to calculate the limiting value for the given parameter p defining its profile confidence interval

    Parameters
    ----------
    rss0 : float
        Residual sum of squares when all parameters are fitted
    alpha : float
        One minus alpha gives the confidence level.
    n : int
        Number of data points.
    p : int
        Number of parameters.

    Returns
    -------
    float
        The limiting value for the residual sum of squares.

    """

    critical_value = calc_critical_value(alpha,n,p)

    return rss0 * ( 1 + critical_value / (n-p) )

def calculate_rss_limit(y_exp,y_fit,p,alpha=0.05):

    """
    Calculate the RSS limit.

    Parameters
    ----------
    y_exp : list or np.array
        Measured data.
    y_fit : list or np.array
        Fitted data, when all parameters are fitted.
    p : int
        Number of parameters.
    alpha : float, optional
        One minus alpha gives the confidence level. The default is 0.05.

    Returns
    -------
    float
        The residual sum of squares limit.

    """

    rss0 = calc_residual_sum_of_squares(y_exp,y_fit)

    # Deduce the number of data points from the length of the measured data
    n = len(y_exp)

    rss_limit = calc_residual_sum_of_squares_profile(rss0,alpha,n,p)

    return rss_limit
