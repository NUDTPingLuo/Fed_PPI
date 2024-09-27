import numpy as np
from numba import njit
from scipy.stats import norm, binom
from scipy.special import expit
from scipy.optimize import brentq, minimize
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic
from sklearn.linear_model import LogisticRegression
from statistics_utils import (
    construct_weight_vector,
    safe_expit,
    safe_log1pexp,
    compute_cdf,
    compute_cdf_diff,
    linfty_dkw,
    linfty_binom,
    form_discrete_distribution,
    reshape_to_2d,
)

def rectified_p_value(
    rectifier,
    rectifier_std,
    imputed_mean,
    imputed_std,
    null=0,
    alternative="two-sided",
):
    """Computes a rectified p-value.

    Args:
        rectifier (float or ndarray): Rectifier value.
        rectifier_std (float or ndarray): Rectifier standard deviation.
        imputed_mean (float or ndarray): Imputed mean.
        imputed_std (float or ndarray): Imputed standard deviation.
        null (float, optional): Value of the null hypothesis to be tested. Defaults to `0`.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.

    Returns:
        float or ndarray: P-value.
    """
    rectified_point_estimate = imputed_mean + rectifier
    rectified_std = np.maximum(
        np.sqrt(imputed_std**2 + rectifier_std**2), 1e-16
    )
    return _zstat_generic(
        rectified_point_estimate, 0, rectified_std, alternative, null
    )[1]


"""
    MEAN ESTIMATION

"""


def ppi_mean_pointestimate(
    Y,
    Yhat,
    Yhat_unlabeled,
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lambd_optim_mode="overall",
):
    """Computes the prediction-powered point estimate of the d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the dimension of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set. Defaults to all ones vector.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set. Defaults to all ones vector.

    Returns:
        float or ndarray: Prediction-powered point estimate of the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Yhat.shape[1]

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    if lhat is None:
        ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean(0) + (
            w * (Y - Yhat)
        ).mean(0)
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
            optim_mode=lambd_optim_mode,
        )
        return ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return (w_unlabeled * lhat * Yhat_unlabeled).mean(axis=0) + (
            w * (Y - lhat * Yhat)
        ).mean(axis=0).squeeze()


def ppi_mean_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lambd_optim_mode="overall",
):
    """Computes the prediction-powered confidence interval for a d-dimensional mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Y.shape[1] if len(Y.shape) > 1 else 1

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    if lhat is None:
        ppi_pointest = ppi_mean_pointestimate(
            Y,
            Yhat,
            Yhat_unlabeled,
            lhat=1,
            w=w,
            w_unlabeled=w_unlabeled,
        )
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            clip=True,
            optim_mode=lambd_optim_mode,
        )
        return ppi_mean_ci(
            Y,
            Yhat,
            Yhat_unlabeled,
            alpha=alpha,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    ppi_pointest = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    imputed_std = (w_unlabeled * (lhat * Yhat_unlabeled)).std(0) / np.sqrt(N)
    rectifier_std = (w * (Y - lhat * Yhat)).std(0) / np.sqrt(n)
    imputed_var = imputed_std**2
    rectifier_var = rectifier_std**2

    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(imputed_var + rectifier_var),
        alpha,
        alternative,
    )

def ppi_mean_ci_FL(
    Y,
    Yhat,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lambd_optim_mode="overall",
):
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = Y.shape[1] if len(Y.shape) > 1 else 1

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)

    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)
    
    if lhat is None:
            ppi_pointest = ppi_mean_pointestimate(
                Y,
                Yhat,
                Yhat_unlabeled,
                lhat=1,
                w=w,
                w_unlabeled=w_unlabeled,
            )
            grads = w * (Y - ppi_pointest)
            grads_hat = w * (Yhat - ppi_pointest)
            grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
            inv_hessian = np.eye(d)
            lhat = _calc_lhat_glm(
                grads,
                grads_hat,
                grads_hat_unlabeled,
                inv_hessian,
                coord=None,
                clip=True,
                optim_mode=lambd_optim_mode,
            )
            return ppi_mean_ci_FL(
                Y,
                Yhat,
                Yhat_unlabeled,
                alpha=alpha,
                lhat=lhat,
                coord=coord,
                w=w,
                w_unlabeled=w_unlabeled,
            )
    
    ppi_pointest = ppi_mean_pointestimate(
        Y,
        Yhat,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    imputed_std = (w_unlabeled * (lhat * Yhat_unlabeled)).std(0)
    rectifier_std = (w * (Y - lhat * Yhat)).std(0)
    imputed_var = imputed_std**2
    rectifier_var = rectifier_std**2

    return ppi_pointest, imputed_var, rectifier_var

def combine_var(sample_mean, sample_counts, sample_variances):
    """
    计算多组数据合并后的方差。
    
    参数:
    sample_counts (list of ints): 每组数据的样本数量。
    sample_variances (list of floats): 每组数据的方差。
    
    返回:
    float: 合并后的方差。
    """
    # 初始化合并后的样本数量和方差
    combined_count = sample_counts[0]
    combined_variance = sample_variances[0]
    combined_mean = sample_mean[0]
    
    for i in range(1, len(sample_counts)):
        n1 = combined_count
        n2 = sample_counts[i]
        var1 = combined_variance
        var2 = sample_variances[i]
        mean1 = combined_mean
        mean2 = sample_mean[i]
        
        # 合并后的样本数量
        combined_count = n1 + n2
        
        # 合并后的方差计算
        # combined_variance = (
        #     ((n1-1) * var1 + (n2-1) * var2) / (combined_count -2)
        # )
        combined_variance = (
                (n1 * var1 + n2 * var2 + n1 * n2 * (combined_mean - mean2)**2 / combined_count) / combined_count
        )
        # 合并后的平均值
        combined_mean = (n1 * mean1 + n2 * mean2) / combined_count
        
    return combined_mean, combined_variance

def sample_mean(proportions, sample_list):
    mean_list = sample_list[0] * proportions[0]
    for i in range(1, len(sample_list)):
        mean_list_1 = mean_list
        mean_list_2 = sample_list[i] * proportions[i]
        mean_list = mean_list_1 + mean_list_2
    return mean_list

def ppi_mean_pval(
    Y,
    Yhat,
    Yhat_unlabeled,
    null=0,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
    lambd_optim_mode="overall",
):
    """Computes the prediction-powered p-value for a 1D mean.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        null (float): Value of the null hypothesis to be tested.
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        float or ndarray: Prediction-powered p-value for the mean.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    # w = np.ones(n) if w is None else w / w.sum() * n
    w = construct_weight_vector(n, w, vectorized=True)
    w_unlabeled = construct_weight_vector(N, w_unlabeled, vectorized=True)

    Y = reshape_to_2d(Y)
    Yhat = reshape_to_2d(Yhat)
    Yhat_unlabeled = reshape_to_2d(Yhat_unlabeled)
    d = Y.shape[1]

    if lhat is None:
        ppi_pointest = (w_unlabeled * Yhat_unlabeled).mean(0) + (
            w * (Y - Yhat)
        ).mean(0)
        grads = w * (Y - ppi_pointest)
        grads_hat = w * (Yhat - ppi_pointest)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - ppi_pointest)
        inv_hessian = np.eye(d)
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord=None,
            optim_mode=lambd_optim_mode,
        )

    return rectified_p_value(
        rectifier=(w * Y - lhat * w * Yhat).mean(0),
        rectifier_std=(w * Y - lhat * w * Yhat).std(0) / np.sqrt(n),
        imputed_mean=(w_unlabeled * lhat * Yhat_unlabeled).mean(0),
        imputed_std=(w_unlabeled * lhat * Yhat_unlabeled).std(0) / np.sqrt(N),
        null=null,
        alternative=alternative,
    )


"""
    QUANTILE ESTIMATION

"""


def _rectified_cdf(Y, Yhat, Yhat_unlabeled, grid, w=None, w_unlabeled=None):
    """Computes the rectified CDF of the data.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        grid (ndarray): Grid of values to compute the CDF at.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        ndarray: Rectified CDF of the data at the specified grid points.
    """
    w = np.ones(Y.shape[0]) if w is None else w / w.sum() * Y.shape[0]
    w_unlabeled = (
        np.ones(Yhat_unlabeled.shape[0])
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * Yhat_unlabeled.shape[0]
    )
    cdf_Yhat_unlabeled, _ = compute_cdf(Yhat_unlabeled, grid, w=w_unlabeled)
    cdf_rectifier, _ = compute_cdf_diff(Y, Yhat, grid, w=w)
    return cdf_Yhat_unlabeled + cdf_rectifier


def ppi_quantile_pointestimate(
    Y, Yhat, Yhat_unlabeled, q, exact_grid=False, w=None, w_unlabeled=None
):
    """Computes the prediction-powered point estimate of the quantile.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        q (float): Quantile to estimate.
        exact_grid (bool, optional): Whether to compute the exact solution (True) or an approximate solution based on a linearly spaced grid of 5000 values (False).
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        float: Prediction-powered point estimate of the quantile.
    """
    assert len(Y.shape) == 1
    w = np.ones(Y.shape[0]) if w is None else w / w.sum() * Y.shape[0]
    w_unlabeled = (
        np.ones(Yhat_unlabeled.shape[0])
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * Yhat_unlabeled.shape[0]
    )
    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)
    rectified_cdf = _rectified_cdf(
        Y, Yhat, Yhat_unlabeled, grid, w=w, w_unlabeled=w_unlabeled
    )
    minimizers = np.argmin(np.abs(rectified_cdf - q))
    minimizer = (
        minimizers
        if isinstance(minimizers, (int, np.int64))
        else minimizers[0]
    )
    return grid[
        minimizer
    ]  # Find the intersection of the rectified CDF and the quantile


def ppi_quantile_ci(
    Y,
    Yhat,
    Yhat_unlabeled,
    q,
    alpha=0.1,
    exact_grid=False,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for the quantile.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        q (float): Quantile to estimate. Must be in the range (0, 1).
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        exact_grid (bool, optional): Whether to use the exact grid of values or a linearly spaced grid of 5000 values.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the quantile.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    grid = np.concatenate([Y, Yhat, Yhat_unlabeled], axis=0)
    if exact_grid:
        grid = np.sort(grid)
    else:
        grid = np.linspace(grid.min(), grid.max(), 5000)
    cdf_Yhat_unlabeled, cdf_Yhat_unlabeled_std = compute_cdf(
        Yhat_unlabeled, grid, w=w_unlabeled
    )
    cdf_rectifier, cdf_rectifier_std = compute_cdf_diff(Y, Yhat, grid, w=w)
    # Calculate rectified p-value for null that the rectified cdf is equal to q
    rec_p_value = rectified_p_value(
        cdf_rectifier,
        cdf_rectifier_std / np.sqrt(n),
        cdf_Yhat_unlabeled,
        cdf_Yhat_unlabeled_std / np.sqrt(N),
        null=q,
        alternative="two-sided",
    )
    # Return the min and max values of the grid where p > alpha
    return grid[rec_p_value > alpha][[0, -1]]

def ppi_quantile_ci_FL(
    Y,
    Yhat,
    Yhat_unlabeled,
    q,
    grid,
    alpha=0.1,
    exact_grid=False,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for the quantile.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        q (float): Quantile to estimate. Must be in the range (0, 1).
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        exact_grid (bool, optional): Whether to use the exact grid of values or a linearly spaced grid of 5000 values.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the quantile.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    cdf_Yhat_unlabeled, cdf_Yhat_unlabeled_std = compute_cdf(
        Yhat_unlabeled, grid, w=w_unlabeled
    )
    cdf_rectifier, cdf_rectifier_std = compute_cdf_diff(Y, Yhat, grid, w=w)

    cdf_Yhat_unlabeled_var = cdf_Yhat_unlabeled_std**2
    cdf_rectifier_var = cdf_rectifier_std**2

    # Calculate rectified p-value for null that the rectified cdf is equal to q
    # rec_p_value = rectified_p_value(
    #     cdf_rectifier,
    #     np.sqrt(cdf_rectifier_var),
    #     cdf_Yhat_unlabeled,
    #     np.sqrt(cdf_Yhat_unlabeled_var),
    #     null=q,
    #     alternative="two-sided",
    # )
    # Return the min and max values of the grid where p > alpha
    return cdf_Yhat_unlabeled, cdf_rectifier, cdf_Yhat_unlabeled_var, cdf_rectifier_var

"""
    ORDINARY LEAST SQUARES

"""


def _ols(X, Y, return_se=False):
    """Computes the ordinary least squares coefficients.

    Args:
        X (ndarray): Covariates.
        Y (ndarray): Labels.
        return_se (bool, optional): Whether to return the standard errors of the coefficients.

    Returns:
        theta (ndarray): Ordinary least squares estimate of the coefficients.
        se (ndarray): If return_se==True, return the standard errors of the coefficients.
    """
    regression = OLS(Y, exog=X).fit()
    theta = regression.params
    if return_se:
        return theta, regression.HC0_se
    else:
        return theta


def _wls(X, Y, w=None, return_se=False):
    """Computes the weighted least squares estimate of the coefficients.

    Args:
        X (ndarray): Covariates.
        Y (ndarray): Labels.
        w (ndarray, optional): Sample weights.
        return_se (bool, optional): Whether to return the standard errors.

    Returns:
        theta (ndarray): Weighted least squares estimate of the coefficients.
        se (ndarray): If return_se==True, returns the standard errors of the coefficients.
    """
    if w is None or np.all(w == 1):
        return _ols(X, Y, return_se=return_se)

    regression = WLS(Y, exog=X, weights=w).fit()
    theta = regression.params
    if return_se:
        return theta, regression.HC0_se
    else:
        return theta

@njit
def _ols_get_stats(
    pointest,
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    w=None,
    w_unlabeled=None,
    use_unlabeled=True,
):
    """Computes the statistics needed for the OLS-based prediction-powered inference.

    Args:
        pointest (ndarray): A point estimate of the coefficients.
        X (ndarray): Covariates for the labeled data set.
        Y (ndarray): Labels for the labeled data set.
        Yhat (ndarray): Predictions for the labeled data set.
        X_unlabeled (ndarray): Covariates for the unlabeled data set.
        Yhat_unlabeled (ndarray): Predictions for the unlabeled data set.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.
        use_unlabeled (bool, optional): Whether to use the unlabeled data set.

    Returns:
        grads (ndarray): Gradient of the loss function with respect to the coefficients.
        grads_hat (ndarray): Gradient of the loss function with respect to the coefficients, evaluated using the labeled predictions.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the coefficients, evaluated using the unlabeled predictions.
        inv_hessian (ndarray): Inverse Hessian of the loss function with respect to the coefficients.
    """
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    d = X.shape[1]
    w = np.ones(n) if w is None else w / np.sum(w) * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / np.sum(w_unlabeled) * N
    )

    hessian = np.zeros((d, d))
    grads_hat_unlabeled = np.zeros(X_unlabeled.shape)
    if use_unlabeled:
        for i in range(N):
            hessian += (
                w_unlabeled[i]
                / (N + n)
                * np.outer(X_unlabeled[i], X_unlabeled[i])
            )
            grads_hat_unlabeled[i, :] = (
                w_unlabeled[i]
                * X_unlabeled[i, :]
                * (np.dot(X_unlabeled[i, :], pointest) - Yhat_unlabeled[i])
            )

    grads = np.zeros(X.shape)
    grads_hat = np.zeros(X.shape)
    for i in range(n):
        hessian += (
            w[i] / (N + n) * np.outer(X[i], X[i])
            if use_unlabeled
            else w[i] / n * np.outer(X[i], X[i])
        )
        grads[i, :] = w[i] * X[i, :] * (np.dot(X[i, :], pointest) - Y[i])
        grads_hat[i, :] = (
            w[i] * X[i, :] * (np.dot(X[i, :], pointest) - Yhat[i])
        )

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    return grads, grads_hat, grads_hat_unlabeled, inv_hessian


def ppi_ols_pointestimate(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered point estimate of the OLS coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        ndarray: Prediction-powered point estimate of the OLS coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / np.sum(w) * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / np.sum(w_unlabeled) * N
    )
    use_unlabeled = lhat != 0

    imputed_theta = (
        _wls(X_unlabeled, Yhat_unlabeled, w=w_unlabeled)
        if lhat is None
        else _wls(X_unlabeled, lhat * Yhat_unlabeled, w=w_unlabeled)
    )
    rectifier = (
        _wls(X, Y - Yhat, w=w)
        if lhat is None
        else _wls(X, Y - lhat * Yhat, w=w)
    )
    ppi_pointest = imputed_theta + rectifier

    if lhat is None:
        grads, grads_hat, grads_hat_unlabeled, inv_hessian = _ols_get_stats(
            ppi_pointest,
            X.astype(float),
            Y,
            Yhat,
            X_unlabeled.astype(float),
            Yhat_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled,
            use_unlabeled=use_unlabeled,
        )
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord,
            clip=True,
        )
        return ppi_ols_pointestimate(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return ppi_pointest

def ppi_ols_pointestimate_FL(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):

    imputed_theta_tmp = np.linalg.pinv([X_unlabeled[0]]) @ [[Yhat_unlabeled[0]]]
    for i in range(1, len(Yhat_unlabeled)):
        imputed_theta_tmp += np.linalg.pinv([X_unlabeled[i]]) @ [[Yhat_unlabeled[i]]]
    imputed_theta = np.array([imputed_theta_tmp[0][0], imputed_theta_tmp[1][0]]) / len(Yhat_unlabeled)

    Y_rec = Yhat - Y
    rectifier = np.linalg.pinv([X[0]]) @ [[Y_rec[0]]]
    for i in range(1, len(Yhat)):
        rectifier += np.linalg.pinv([X[i]]) @ [[Y_rec[i]]]
    rectifier = np.array([rectifier[0][0], rectifier[1][0]]) / len(Yhat)

    ppi_pointest = imputed_theta - rectifier

    return ppi_pointest

def ppi_ols_true_point(
    X,
    Y,
):

    theta_tmp = np.linalg.pinv([X[0]]) @ [[Y[0]]]
    for i in range(1, len(Y)):
        theta_tmp += np.linalg.pinv([X[i]]) @ [[Y[i]]]
    theta = np.array([theta_tmp[0][0], theta_tmp[1][0]]) / len(Y)

    return theta

def ppi_ols_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for the OLS coefficients using the PPI++ algorithm from `[ADZ23] <https://arxiv.org/abs/2311.01453>`__.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the OLS coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    use_unlabeled = lhat != 0  # If lhat is 0, revert to classical estimation.

    ppi_pointest = ppi_ols_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _ols_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
        use_unlabeled=use_unlabeled,
    )

    if lhat is None:
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord,
            clip=True,
        )
        return ppi_ols_ci(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alpha,
            alternative=alternative,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    var_unlabeled = np.cov(lhat * grads_hat_unlabeled.T).reshape(d, d)

    var = np.cov(grads.T - lhat * grads_hat.T).reshape(d, d)

    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian

    ppi_pointest = ppi_ols_pointestimate_FL(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )

def ppi_ols_ci_FL(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    w=None,
    w_unlabeled=None,
):

    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    use_unlabeled = lhat != 0  # If lhat is 0, revert to classical estimation.

    ppi_pointest = ppi_ols_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )
    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _ols_get_stats(
        ppi_pointest,
        X.astype(float),
        Y,
        Yhat,
        X_unlabeled.astype(float),
        Yhat_unlabeled,
        w=w,
        w_unlabeled=w_unlabeled,
        use_unlabeled=use_unlabeled,
    )

    if lhat is None:
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            coord,
            clip=True,
        )
        return ppi_ols_ci_FL(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alpha,
            alternative=alternative,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    var_unlabeled = np.cov(lhat * grads_hat_unlabeled.T).reshape(d, d)

    var = np.cov(grads.T - lhat * grads_hat.T).reshape(d, d)

    Sigma_hat = np.diag(inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian)

    ppi_pointest = ppi_ols_pointestimate_FL(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    # return _zconfint_generic(
    #     ppi_pointest,
    #     np.sqrt(np.diag(Sigma_hat) / n),
    #     alpha=alpha,
    #     alternative=alternative,
    # )
    return ppi_pointest, var_unlabeled, var, inv_hessian
"""
    LOGISTIC REGRESSION

"""


def ppi_logistic_pointestimate(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lhat=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered point estimate of the logistic regression coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        optimizer_options (dict, optional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
        w (ndarray, optional): Sample weights for the labeled data set.
        w_unlabeled (ndarray, optional): Sample weights for the unlabeled data set.

    Returns:
        ndarray: Prediction-powered point estimate of the logistic regression coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    if optimizer_options is None:
        optimizer_options = {"ftol": 1e-15}
    if "ftol" not in optimizer_options.keys():
        optimizer_options["ftol"] = 1e-15

    # Initialize theta
    theta = (
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-15,
            fit_intercept=False,
        )
        .fit(X, Y)
        .coef_.squeeze()
    )
    # theta = np.array([1.9133574599926604e-05, 1.8665990101002545e-07])
    if len(theta.shape) == 0:
        theta = theta.reshape(1)

    lhat_curr = 1 if lhat is None else lhat

    def rectified_logistic_loss(_theta):
        rectified_logistic_loss_func = (
            lhat_curr
            / N
            * np.sum(
                w_unlabeled
                * (
                    -Yhat_unlabeled * (X_unlabeled @ _theta)
                    + safe_log1pexp(X_unlabeled @ _theta)
                )
            )
            - lhat_curr
            / n
            * np.sum(w * (-Yhat * (X @ _theta) + safe_log1pexp(X @ _theta)))
            + 1
            / n
            * np.sum(w * (-Y * (X @ _theta) + safe_log1pexp(X @ _theta)))
        )
        return rectified_logistic_loss_func

    def rectified_logistic_grad(_theta):
        return (
            lhat_curr
            / N
            * X_unlabeled.T
            @ (
                w_unlabeled
                * (safe_expit(X_unlabeled @ _theta) - Yhat_unlabeled)
            )
            - lhat_curr / n * X.T @ (w * (safe_expit(X @ _theta) - Yhat))
            + 1 / n * X.T @ (w * (safe_expit(X @ _theta) - Y))
        )

    ppi_pointest = minimize(
        rectified_logistic_loss,
        theta,
        jac=rectified_logistic_grad,
        method="L-BFGS-B",
        tol=optimizer_options["ftol"],
        options=optimizer_options,
    ).x

    if lhat is None:
        (
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
        ) = _logistic_get_stats(
            ppi_pointest,
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            w,
            w_unlabeled,
        )
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            clip=True,
        )
        return ppi_logistic_pointestimate(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            optimizer_options=optimizer_options,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )
    else:
        return ppi_pointest

def ppi_logistic_pointestimate_FL(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lhat=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    n_k = len(Yhat[0])
    n = len(Yhat) * n_k
    N_k = len(Yhat_unlabeled[0])
    N = len(Yhat_unlabeled) * N_k
    if optimizer_options is None:
        optimizer_options = {"ftol": 1e-15}
    if "ftol" not in optimizer_options.keys():
        optimizer_options["ftol"] = 1e-15

    # Initialize theta
    # theta = (
    #     LogisticRegression(
    #         penalty=None,
    #         solver="lbfgs",
    #         max_iter=10000,
    #         tol=1e-15,
    #         fit_intercept=False,
    #     )
    #     .fit(X, Y)
    #     .coef_.squeeze()
    # )
    theta = np.array([0, 0])
    if len(theta.shape) == 0:
        theta = theta.reshape(1)

    lhat_curr = 1 if lhat is None else lhat

    def rectified_logistic_loss(_theta):
        w_unlabeled = np.ones(len(Yhat_unlabeled[0]))
        w = np.ones(len(Yhat[0]))
        rectified_logistic_loss_func = (
                lhat_curr
                / N
                * np.sum(
            w_unlabeled
            * (
                    -Yhat_unlabeled[0] * (X_unlabeled[0] @ _theta)
                    + safe_log1pexp(X_unlabeled[0] @ _theta)
            )
        )
                - lhat_curr
                / n
                * np.sum(w * (-Yhat[0] * (X[0] @ _theta) + safe_log1pexp(X[0] @ _theta)))
                + 1
                / n
                * np.sum(w * (-Y[0] * (X[0] @ _theta) + safe_log1pexp(X[0] @ _theta)))
        )
        for i in range(1, len(X_unlabeled)):
            w_unlabeled = np.ones(len(Yhat_unlabeled[i]))
            w = np.ones(len(Yhat[i]))
            rectified_logistic_loss_func1 = rectified_logistic_loss_func
            rectified_logistic_loss_func2 = (
                lhat_curr
                / N
                * np.sum(
                    w_unlabeled
                    * (
                        -Yhat_unlabeled[i] * (X_unlabeled[i] @ _theta)
                        + safe_log1pexp(X_unlabeled[i] @ _theta)
                    )
                )
                - lhat_curr
                / n
                * np.sum(w * (-Yhat[i] * (X[i] @ _theta) + safe_log1pexp(X[i] @ _theta)))
                + 1
                / n
                * np.sum(w * (-Y[i] * (X[i] @ _theta) + safe_log1pexp(X[i] @ _theta)))
            )
            rectified_logistic_loss_func = rectified_logistic_loss_func1 + rectified_logistic_loss_func2
        return rectified_logistic_loss_func

    def rectified_logistic_grad(_theta):
        w_unlabeled = np.ones(len(Yhat_unlabeled[0]))
        w = np.ones(len(Yhat[0]))
        rectified_logistic_grad_func = (
            lhat_curr
            / N
            * X_unlabeled[0].T
            @ (
                w_unlabeled
                * (safe_expit(X_unlabeled[0] @ _theta) - Yhat_unlabeled[0])
            )
            - lhat_curr / n * X[0].T @ (w * (safe_expit(X[0] @ _theta) - Yhat[0]))
            + 1 / n * X[0].T @ (w * (safe_expit(X[0] @ _theta) - Y[0]))
        )
        for i in range(1, len(X_unlabeled)):
            w_unlabeled = np.ones(len(Yhat_unlabeled[i]))
            w = np.ones(len(Yhat[i]))
            rectified_logistic_grad_func1 = rectified_logistic_grad_func
            rectified_logistic_grad_func2 = (
            lhat_curr
            / N
            * X_unlabeled[i].T
            @ (
                w_unlabeled
                * (safe_expit(X_unlabeled[i] @ _theta) - Yhat_unlabeled[i])
            )
            - lhat_curr / n * X[i].T @ (w * (safe_expit(X[i] @ _theta) - Yhat[i]))
            + 1 / n * X[i].T @ (w * (safe_expit(X[i] @ _theta) - Y[i]))
        )
            rectified_logistic_grad_func = rectified_logistic_grad_func1 + rectified_logistic_grad_func2
        return rectified_logistic_grad_func

    ppi_pointest = minimize(
        rectified_logistic_loss,
        theta,
        jac=rectified_logistic_grad,
        method="L-BFGS-B",
        tol=optimizer_options["ftol"],
        options=optimizer_options,
    ).x

    return ppi_pointest

@njit
def _logistic_get_stats(
    pointest,
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    w=None,
    w_unlabeled=None,
    use_unlabeled=True,
):
    """Computes the statistics needed for the logistic regression confidence interval.

    Args:
        pointest (ndarray): Point estimate of the logistic regression coefficients.
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        w (ndarray, optional): Standard errors of the gold-standard labels.
        w_unlabeled (ndarray, optional): Standard errors of the unlabeled data.
        use_unlabeled (bool, optional): Whether to use the unlabeled data.

    Returns:
        grads (ndarray): Gradient of the loss function on the labeled data.
        grads_hat (ndarray): Gradient of the loss function on the labeled predictions.
        grads_hat_unlabeled (ndarray): Gradient of the loss function on the unlabeled predictions.
        inv_hessian (ndarray): Inverse Hessian of the loss function on the unlabeled data.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )

    mu = safe_expit(X @ pointest)
    mu_til = safe_expit(X_unlabeled @ pointest)

    hessian = np.zeros((d, d))
    grads_hat_unlabeled = np.zeros(X_unlabeled.shape)
    if use_unlabeled:
        for i in range(N):
            hessian += (
                w_unlabeled[i]
                / (N + n)
                * mu_til[i]
                * (1 - mu_til[i])
                * np.outer(X_unlabeled[i], X_unlabeled[i])
            )
            grads_hat_unlabeled[i, :] = (
                w_unlabeled[i]
                * X_unlabeled[i, :]
                * (mu_til[i] - Yhat_unlabeled[i])
            )

    grads = np.zeros(X.shape)
    grads_hat = np.zeros(X.shape)
    for i in range(n):
        hessian += (
            w[i] / (N + n) * mu[i] * (1 - mu[i]) * np.outer(X[i], X[i])
            if use_unlabeled
            else w[i] / n * mu[i] * (1 - mu[i]) * np.outer(X[i], X[i])
        )
        grads[i, :] = w[i] * X[i, :] * (mu[i] - Y[i])
        grads_hat[i, :] = w[i] * X[i, :] * (mu[i] - Yhat[i])

    inv_hessian = np.linalg.inv(hessian).reshape(d, d)
    return grads, grads_hat, grads_hat_unlabeled, inv_hessian


def ppi_logistic_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):
    """Computes the prediction-powered confidence interval for the logistic regression coefficients using the PPI++ algorithm from `[ADZ23] <https://arxiv.org/abs/2311.01453>`__.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter (see `[ADZ23] <https://arxiv.org/abs/2311.01453>`__). The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        optimizer_options (dict, ooptional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
        w (ndarray, optional): Weights for the labeled data. If None, it is set to 1.
        w_unlabeled (ndarray, optional): Weights for the unlabeled data. If None, it is set to 1.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the logistic regression coefficients.

    Notes:
        `[ADZ23] <https://arxiv.org/abs/2311.01453>`__ A. N. Angelopoulos, J. C. Duchi, and T. Zrnic. PPI++: Efficient Prediction Powered Inference. arxiv:2311.01453, 2023.
    """
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    use_unlabeled = lhat != 0

    ppi_pointest = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options=optimizer_options,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _logistic_get_stats(
        ppi_pointest,
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        w,
        w_unlabeled,
        use_unlabeled=use_unlabeled,
    )
    if lhat is None:
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            clip=True,
        )
        return ppi_logistic_ci(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alpha,
            optimizer_options=optimizer_options,
            alternative=alternative,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    var_unlabeled = np.cov(lhat * grads_hat_unlabeled.T).reshape(d, d)

    var = np.cov(grads.T - lhat * grads_hat.T).reshape(d, d)

    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian

    tmp = np.diag(Sigma_hat) / n

    return _zconfint_generic(
        ppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,
    )

def ppi_logistic_ci_FL(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    optimizer_options=None,
    w=None,
    w_unlabeled=None,
):

    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n) if w is None else w / w.sum() * n
    w_unlabeled = (
        np.ones(N)
        if w_unlabeled is None
        else w_unlabeled / w_unlabeled.sum() * N
    )
    use_unlabeled = lhat != 0

    ppi_pointest = ppi_logistic_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        optimizer_options=optimizer_options,
        lhat=lhat,
        coord=coord,
        w=w,
        w_unlabeled=w_unlabeled,
    )

    grads, grads_hat, grads_hat_unlabeled, inv_hessian = _logistic_get_stats(
        ppi_pointest,
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        w,
        w_unlabeled,
        use_unlabeled=use_unlabeled,
    )
    if lhat is None:
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            clip=True,
        )
        return ppi_logistic_ci_FL(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alpha,
            optimizer_options=optimizer_options,
            alternative=alternative,
            lhat=lhat,
            coord=coord,
            w=w,
            w_unlabeled=w_unlabeled,
        )

    var_unlabeled = np.cov(lhat * grads_hat_unlabeled.T).reshape(d, d)

    var = np.cov(grads.T - lhat * grads_hat.T).reshape(d, d)

    Sigma_hat = np.diag(inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian)
    tmp = Sigma_hat
    #
    # return _zconfint_generic(
    #     ppi_pointest,
    #     np.sqrt(np.diag(Sigma_hat) / n),
    #     alpha=alpha,
    #     alternative=alternative,
    # )
    return ppi_pointest, var_unlabeled, var, inv_hessian


def _calc_lhat_glm(
    grads,
    grads_hat,
    grads_hat_unlabeled,
    inv_hessian,
    coord=None,
    clip=False,
    optim_mode="overall",
):
    """
    Calculates the optimal value of lhat for the prediction-powered confidence interval for GLMs.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lhat`, when `optim_mode="overall"`.
        If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        clip (bool, optional): Whether to clip the value of lhat to be non-negative. Defaults to `False`.
        optim_mode (ndarray, optional): Mode for which to optimize `lhat`, either `overall` or `element`.
        If `overall`, it optimizes the total variance over all coordinates, and the function returns a scalar.
        If `element`, it optimizes the variance for each coordinate separately, and the function returns a vector.


    Returns:
        float: Optimal value of `lhat`. Lies in [0,1].
    """
    grads = reshape_to_2d(grads)
    grads_hat = reshape_to_2d(grads_hat)
    grads_hat_unlabeled = reshape_to_2d(grads_hat_unlabeled)
    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]
    if grads.shape[1] != d:
        raise ValueError(
            "Dimension mismatch between the gradient and the inverse Hessian."
        )

    grads_cent = grads - grads.mean(axis=0)
    grad_hat_cent = grads_hat - grads_hat.mean(axis=0)
    cov_grads = (1 / n) * (
        grads_cent.T @ grad_hat_cent + grad_hat_cent.T @ grads_cent
    )

    var_grads_hat = np.cov(
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    )
    var_grads_hat = var_grads_hat.reshape(d, d)

    vhat = inv_hessian if coord is None else inv_hessian[coord, coord]
    if optim_mode == "overall":
        num = (
            np.trace(vhat @ cov_grads @ vhat)
            if coord is None
            else vhat @ cov_grads @ vhat
        )
        denom = (
            2 * (1 + (n / N)) * np.trace(vhat @ var_grads_hat @ vhat)
            if coord is None
            else 2 * (1 + (n / N)) * vhat @ var_grads_hat @ vhat
        )
        lhat = num / denom
        lhat = lhat.item()
    elif optim_mode == "element":
        num = np.diag(vhat @ cov_grads @ vhat)
        denom = 2 * (1 + (n / N)) * np.diag(vhat @ var_grads_hat @ vhat)
        lhat = num / denom
    else:
        raise ValueError(
            "Invalid value for optim_mode. Must be either 'overall' or 'element'."
        )
    if clip:
        lhat = np.clip(lhat, 1e-5, 1)
    return lhat


"""
    DISCRETE DISTRIBUTION ESTIMATION UNDER LABEL SHIFT

"""


def ppi_distribution_label_shift_ci(
    Y, Yhat, Yhat_unlabeled, K, nu, alpha=0.1, delta=None, return_counts=True
):
    """Computes the prediction-powered confidence interval for nu^T f for a discrete distribution f, under label shift.

    Args:
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        K (int): Number of classes.
        nu (ndarray): Vector nu. Coordinates must be bounded within [0, 1].
        alpha (float, optional): Final error level; the confidence interval will target a coverage of 1 - alpha. Must be in (0, 1).
        delta (float, optional): Error level of the intermediate confidence interval for the mean. Must be in (0, alpha). If return_counts == False, then delta is set equal to alpha and ignored.
        return_counts (bool, optional): Whether to return the number of samples in each class as opposed to the mean.

    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for nu^T f for a discrete distribution f, under label shift.
    """
    if not return_counts:
        delta = alpha
    if delta is None:
        delta = alpha * 0.95
    # Construct the confusion matrix
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]

    # Construct column-normalized confusion matrix Ahat
    C = np.zeros((K, K)).astype(int)
    for j in range(K):
        for l in range(K):
            C[j, l] = np.bitwise_and(Yhat == j, Y == l).astype(int).sum()
    Ahat = C / C.sum(axis=0)

    # Invert Ahat
    Ahatinv = np.linalg.inv(Ahat)
    qfhat = form_discrete_distribution(Yhat_unlabeled, sorted_highlow=True)

    # Calculate the bound
    point_estimate = nu @ Ahatinv @ qfhat

    nmin = C.sum(axis=0).min()

    def invert_budget_split(budget_split):
        return np.sqrt(1 / (4 * nmin)) * (
            norm.ppf(1 - (budget_split * delta) / (2 * K**2))
            - norm.ppf((budget_split * delta) / (2 * K**2))
        ) - np.sqrt(2 / N * np.log(2 / ((1 - budget_split) * delta)))

    try:
        budget_split = brentq(invert_budget_split, 1e-9, 1 - 1e-9)
    except:
        budget_split = 0.999999
    epsilon1 = max(
        [
            linfty_binom(C.sum(axis=0)[k], K, budget_split * delta, Ahat[:, k])
            for k in range(K)
        ]
    )
    epsilon2 = linfty_dkw(N, K, (1 - budget_split) * delta)

    qyhat_lb = np.clip(point_estimate - epsilon1 - epsilon2, 0, 1)
    qyhat_ub = np.clip(point_estimate + epsilon1 + epsilon2, 0, 1)

    if return_counts:
        count_lb = int(binom.ppf((alpha - delta) / 2, N, qyhat_lb))
        count_ub = int(binom.ppf(1 - (alpha - delta) / 2, N, qyhat_ub))
        return count_lb, count_ub
    else:
        return qyhat_lb, qyhat_ub
