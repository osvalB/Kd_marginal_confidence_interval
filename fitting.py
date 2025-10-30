import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from  stats import calc_residual_sum_of_squares, calculate_rss_limit

def calc_one_to_one_binding_signal(ligand_conc,prot_conc,Kd,free_protein_baseline,complex_baseline):

    """
    Given a certain ligand concentration, protein concentration and Kd, calculate the signal.
    The model assumes one-to-one binding P + L <-> PL

    Parameters
    ----------
    ligand_conc : np.array or float
        ligand concentrations
    prot_conc : np.array or float
        protein concentrations
    Kd : float
        Equilibrium dissociation constant
    free_protein_baseline : float
        Baseline signal for free protein
    complex_baseline : float
        Baseline signal for complex

    Returns
    -------
    np.array
        The signal for each ligand concentration and protein concentration.

    Notes
    -----
    The ligand concentration, protein concentration and Kd should be given in the same units.
    For example, everything in micromolar units.
    The baseline signals should be given in the same units as the signal.

    """

    l = ligand_conc
    p = prot_conc
    fpb = free_protein_baseline
    cpb = complex_baseline

    signal = 0.5 * ((Kd + p + l) - np.sqrt((Kd + p + l) ** 2 - 4 * p * l)) * (cpb - fpb) + fpb * p

    return signal

def get_one_to_one_fitting_function(ligand_conc,prot_conc):

    """
    Given a certain ligand concentration and protein concentration, return a function that calculates the signal.
    That function will be used in the fitting procedure.

    Parameters
    ----------
    ligand_conc : np.array or float
        ligand concentrations
    prot_conc : np.array or float
        protein concentrations

    Returns
    -------
    function
        A function that calculates the signal according to the parameters Kd, free_protein_baseline and complex_baseline

    """

    def return_fx(dummy_var,Kd,free_protein_baseline,complex_baseline):

        # dummy_var is required to make the function compatible with the curve_fit function

        return calc_one_to_one_binding_signal(ligand_conc,prot_conc,Kd,free_protein_baseline,complex_baseline)

    return return_fx

def get_one_to_one_fitting_function_fixed_Kd(ligand_conc,prot_conc,Kd):

    """
    Given a certain ligand concentration and protein concentration, return a function that calculates the signal.
    That function will be used in the fitting procedure.

    Parameters
    ----------
    ligand_conc : np.array or float
        ligand concentrations
    prot_conc : np.array or float
        protein concentrations
    Kd : float
        Equilibrium dissociation constant

    Returns
    -------
    function
        A function that calculates the signal according to the parameters free_protein_baseline and complex_baseline

    """

    def return_fx(dummy_var,free_protein_baseline,complex_baseline):

        # dummy_var is required to make the function compatible with the curve_fit function

        return calc_one_to_one_binding_signal(ligand_conc,prot_conc,Kd,free_protein_baseline,complex_baseline)

    return return_fx


def fit_one_to_one_signal(signal,ligand_conc,prot_conc):

    """
    Fit the measured data to the one-to-one binding model.

    Parameters
    ----------
    signal : np.array
        Experimental data.
    ligand_conc : np.array or float
        Ligand concentrations.
    prot_conc : np.array or float
        Protein concentrations.

    Returns
    -------
    fit, cov : tuple
        The fitted parameters and the covariance matrix.

    """

    # Use the ligand concentrations to sort the data
    if isinstance(ligand_conc,np.ndarray):
        sort_idx = np.argsort(ligand_conc)
        signal = signal[sort_idx]

    # Try to find good initial parameters
    max_lig_id = np.argmax(ligand_conc)
    min_lig_id = np.argmin(ligand_conc)

    max_lig_signal = signal[max_lig_id]
    min_lig_signal = signal[min_lig_id]

    baseline_one = np.max(signal/prot_conc)
    baseline_two = np.min(signal/prot_conc)

    if max_lig_signal > min_lig_signal:

        free_protein_baseline = baseline_two
        complex_baseline = baseline_one

    else:

        free_protein_baseline = baseline_one
        complex_baseline = baseline_two

    # Explore different Kd values
    min_lig = np.min(ligand_conc)
    max_lig = np.max(ligand_conc)

    min_lig_log = np.log10(min_lig)
    max_lig_log = np.log10(max_lig)

    Kd_guesses_log = np.linspace(min_lig_log,max_lig_log,20)

    Kd_guesses = 10**Kd_guesses_log

    rss_init = np.inf
    p0_best  = [free_protein_baseline,complex_baseline]
    Kd_best  = None

    for Kd_guess in Kd_guesses:

        fit_fx_fixed_Kd = get_one_to_one_fitting_function_fixed_Kd(ligand_conc,prot_conc,Kd_guess)

        params, cov = curve_fit(fit_fx_fixed_Kd, None , signal,p0 = p0_best)

        predicted_signal = fit_fx_fixed_Kd(None,params[0],params[1])

        rss = calc_residual_sum_of_squares(signal,predicted_signal)

        if rss < rss_init:
            rss_init = rss
            p0_best = params
            Kd_best = Kd_guess

    # Now fit all parameters
    fit_fx = get_one_to_one_fitting_function(ligand_conc,prot_conc)

    p0_best = [Kd_best] + p0_best.tolist()

    low_bounds = [1E-6,-np.inf,-np.inf]
    up_bounds  = [1E6,np.inf,np.inf]

    params, cov = curve_fit(
        fit_fx,
        None ,
        signal,
        p0 = p0_best,
        bounds = (low_bounds,up_bounds)
    )

    rel_error = np.abs(np.sqrt(np.diag(cov)) / params * 100)

    fitted_signal = fit_fx(None,*params)

    return params, rel_error, fitted_signal

def calc_rss_of_fitting_fixed_Kd(signal,ligand_conc,prot_conc,Kd,p0):

    """
    Fit the measured data to the one-to-one binding model, with a fixed Kd value.
    And then calculate the residual sum of squares.

    Parameters
    ----------
    signal : np.array
        Experimental data.
    ligand_conc : np.array or float
        Ligand concentrations.
    prot_conc : np.array or float
        Protein concentrations.
    Kd : float
        fixed Kd value.
    p0 : float
        initial guess for the free protein baseline and the complex baseline

    Returns
    -------
    float
        The residual sum of squares of the fitting performed with a fixed Kd value.
    """

    fit_fx_fixed_Kd = get_one_to_one_fitting_function_fixed_Kd(ligand_conc,prot_conc,Kd)

    params, cov = curve_fit(fit_fx_fixed_Kd, xdata=None , ydata=signal,p0 = p0)

    predicted_signal = fit_fx_fixed_Kd(None,params[0],params[1])

    rss = calc_residual_sum_of_squares(signal,predicted_signal)

    return rss


def calc_asymmetric_conf_interval(
        signal,
        fitted_signal,
        ligand_conc,
        prot_conc,
        estimated_Kd,
        estimated_free_protein_baseline,
        estimated_complex_baseline,
        p,
        alpha=0.05):

    """
    Given a fitted signal and an estimated Kd, calculate the 95% confidence interval for the Kd value.

    Parameters
    ----------
    signal : np.array
        Measured data.
    fitted_signal : np.array
        Fitted data, when all parameters are fitted.
    ligand_conc : np.array or float
        ligand concentrations.
    prot_conc : np.array or float
        protein concentrations.
    estimated_Kd : float
        Fitted Kd value.
    estimated_free_protein_baseline : float
        Fitted free protein baseline.
    estimated_complex_baseline : float
        Fitted complex baseline.
    p : int
        Number of parameters.

    Returns
    -------
    tuple
        The lower and upper bound of the 95% confidence interval.
    """

    rss_threshold = calculate_rss_limit(signal,fitted_signal,p,alpha)

    # Rescale the Kd to help the minimization procedure
    factor = 1E9

    p0 = [estimated_free_protein_baseline,estimated_complex_baseline]

    def fx_to_optimize(fixed_kd):

        fixed_kd = fixed_kd / factor

        rss = calc_rss_of_fitting_fixed_Kd(signal,ligand_conc,prot_conc,fixed_kd,p0)

        # Return the difference between the RSS with a fixed Kd value and the threshold RSS.
        return  np.abs(rss - rss_threshold)

    # We will only explore marginal confidence intervals around the estimated Kd value divided by 500 and multiplied by 500

    bounds_min = [estimated_Kd*factor / 500, estimated_Kd*factor]
    bounds_max = [estimated_Kd*factor, estimated_Kd*factor * 500]

    min_95_Kd = minimize_scalar(fx_to_optimize,bounds = bounds_min)
    max_95_Kd = minimize_scalar(fx_to_optimize,bounds = bounds_max)

    min_95_Kd = min_95_Kd.x / factor
    max_95_Kd = max_95_Kd.x / factor

    # Raise an error if the values are close to the boundaries
    if (min_95_Kd - estimated_Kd/500) / min_95_Kd < 0.01:
        raise ValueError("The lower bound of the 95% confidence interval could not be estimated")

    if (estimated_Kd*500 - max_95_Kd) / estimated_Kd*500 < 0.01:
        raise ValueError("The upper bound of the 95% confidence interval could not be estimated")

    # Verify that the factor between the higher and lower bound is less than 100
    if max_95_Kd / min_95_Kd > 100:
        raise ValueError("The lower and upper bound of the 95% confidence interval are too far apart")

    return min_95_Kd, max_95_Kd



