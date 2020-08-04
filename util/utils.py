import numpy as np
from sklearn.preprocessing import label_binarize
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.stats import norm
# from util.roc_comparison import compare_auc_delong_xu as delong
from rpy2 import robjects as robj
from rpy2.robjects.packages import importr

def get_calibration_metrics(y_true, y_prob, n_bins, bin_strategy='quantile'):

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    if bin_strategy == 'quantile':  ## equal-frequency bins 
        # Determine bin edges by distribution of data 
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif bin_strategy == 'uniform': ## equal-width bins
        bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'bin_strategy' input. bin_Strategy "
                         "must be either 'quantile' or 'uniform'.")

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])
    
    abs_error = np.abs(prob_pred - prob_true)

    expected_error = np.average(abs_error, weights=(bin_total / y_true.size)[nonzero])
    max_error = np.max(abs_error)

    return prob_true, prob_pred, expected_error, max_error

def stat_ci(scores, confidence_level=0.95):
    mean_score = np.mean(scores)
    #     sorted_scores = np.array(sorted(scores))
    #     alpha = (1.0 - confidence_level) / 2.0
    #     ci_lower = sorted_scores[int(round(alpha * len(sorted_scores)))]
    #     ci_upper = sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))]
    ci_lower, ci_upper=norm.interval(confidence_level, loc=mean_score, scale=np.std(scores,ddof=1))
    return mean_score, ci_lower, ci_upper

def stat_pval(score1, score2, test="mannwhitneyu"):
    '''
    runs test "mannwhitneyu" or "wilcoxon"
    returns stat and pvalue
    '''
    if test=="mannwhitneyu":
        s, p=mannwhitneyu(score1, score2, alternative="two-sided")
    elif test=="wilcoxon":
        try:
            s, p=wilcoxon(score1, score2, alternative="two-sided")
        except ValueError as err:
            s=p=np.nan
    return s, p

def auc_delong_test(y1, probs1, y2, probs2):
    y1=robj.FloatVector(list(y1))
    y2=robj.FloatVector(list(y2))
    probs1=robj.FloatVector(list(probs1))
    probs2=robj.FloatVector(list(probs2))

    proc=importr("pROC")
    roc1=proc.roc(y1, probs1, direction="<", ci=True)
    roc2=proc.roc(y2, probs2, direction="<", ci=True)
    res=proc.roc_test(roc1, roc2, method="delong")

    auc1=tuple(roc1.rx2("auc"))[0]
    ci1=tuple(roc1.rx2("ci"))
    auc2=tuple(roc2.rx2("auc"))[0]
    ci2=tuple(roc2.rx2("ci"))
    pval=list(res.rx2("p.value"))[0]

    ci1=(ci1[0], ci1[2])
    ci2=(ci2[0], ci2[2])

    return auc1, ci1, auc2, ci2, pval



    
    return pval