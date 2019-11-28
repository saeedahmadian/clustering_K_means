from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.cluster as cl
from Prepross import read_data
import scipy.stats as stat
import statsmodels.distributions as dist

demand, lmp = read_data()

data = np.concatenate([demand[:,12:14].reshape([-1,2]),lmp[:,12:14].reshape([-1,2])],axis=1)

scaled = MinMaxScaler().fit(data).transform(data)
stand =StandardScaler().fit(data).transform(data)
colors=np.concatenate([np.array([1 for i in range(demand.shape[0])]).reshape([-1,1]),
                       np.array([10 for i in range(demand.shape[0])]).reshape([-1,1])],axis=1)

# plt.subplot(211)
# plt.scatter(scaled[:,0:2], scaled[:,2:4], c=colors)
# plt.subplot(212)
# plt.scatter(stand[:,0:2], stand[:,2:4])


import sklearn.cluster as cl

kmeans = cl.KMeans(n_clusters=2)
fit_km = kmeans.fit(demand)
predicted=fit_km.predict(demand)


desired_data=[]
regular_data =[]
tmp=demand
for i,val in enumerate(tmp):
    if predicted[i]==1:
        desired_data.append(val)
    else:
        regular_data.append(val)

from sklearn.mixture import GaussianMixture
from sklearn.mixture import gaussian_mixture

gmm=GaussianMixture(n_components=2, random_state=1)

fit_gmm= gmm.fit(data[:,0].reshape((-1,1)))
counter=np.linspace(np.min(data[:,0]),np.max(data[:,0]),2000)


labels=['Regular','Desired']
for i in range(fit_gmm.n_components):
    pdf = fit_gmm.weights_[i] * stat.norm(fit_gmm.means_[i,0], np.sqrt(fit_gmm.covariances_[i,0])).pdf(counter)
    plt.fill(counter,pdf,alpha=.9)

liklihood = np.array(list(map(lambda x: np.exp(fit_gmm.score(np.array(x).reshape(-1,1))),counter)))
plt.hist(data[:,0],bins=80,density=True,alpha=.5)
plt.plot(counter,liklihood,)
plt.show()











a=1


