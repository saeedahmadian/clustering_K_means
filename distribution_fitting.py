from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from Prepross import read_data
import pandas as pd
from scipy.stats.mstats import chisquare
import sklearn.cluster as cl
from sklearn.mixture import GaussianMixture as GMM


demand, lmp = read_data()
data = np.concatenate([demand[:,12:14].reshape([-1,2]),lmp[:,12:14].reshape([-1,2])],axis=1)
standard = StandardScaler().fit(data)
data_st= standard.transform(data)
size=data.shape[0]

k_means= cl.KMeans(n_clusters=2,max_iter=500)
gmm_model= GMM(n_components= 2, covariance_type='full')




percentile_bins = np.linspace(0,100,51)
percentile_cutoffs = np.percentile(data_st, percentile_bins)
observed_frequency, bins = (np.histogram(data_st, bins=percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)

def distributions():
    names =['beta',
              'expon',
              'gamma',
              'lognorm',
              'norm',
              'pearson3',
              'triang',
              'uniform',
              'weibull_min',
              'weibull_max']
    return names


def p_value(data,distribution,param):
    return stat.kstest(data,distribution,args=param)[1]

total_p_value = []
ks_chi_square = []
total_pdf=[]
counter =np.linspace(-6,6,1000)
for dist_name in distributions():
    dist = getattr(stat,dist_name)
    param= dist.fit(data_st[:,0])
    total_pdf.append(dist.pdf(counter,*param[:-2],loc=param[-2],scale=param[-1]))
    total_p_value.append(p_value(data_st[:,0],dist_name,param))
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                          scale=param[-1])
    expected_frequency = []
    for bin in range(len(percentile_bins) - 1):
        expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)

    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size
    cum_expected_frequency = np.cumsum(expected_frequency)
    ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    ks_chi_square.append(ss)

results = pd.DataFrame()
results['Distribution'] = distributions()
results['chi_square'] = ks_chi_square
results['p_value'] = total_p_value
results.sort_values(['chi_square'], inplace=True)

def myplot(data,pdf,list_pdf):
    plt

a=1