import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sn

def read_data():
    list_data = ['Data.csv', 'Data2.csv', 'Data3.csv', 'Data4.csv', 'Data5.csv']
    data = []
    for i in list_data:
        data.append(pd.read_csv(i, usecols=['Date', 'Hour', 'DA_DEMD', 'DEMAND', 'DA_LMP', 'RT_LMP'],
                                header=0, index_col='Date', parse_dates=True))

    """
    columns = ['Hour', 'DA_DEMD', 'DEMAND', 'DA_LMP', 'DA_EC', 'DA_CC', 'DA_MLC',
           'RT_LMP', 'RT_EC', 'RT_CC', 'RT_MLC', 'DryBulb', 'DewPnt', 'SYSLoad', 'RegCP']
    """

    da_demand = pd.DataFrame(index=pd.date_range(start=data[0].index[0], end=data[-1].index[-1]),
                             columns=['da_demand_t_' + str(i) for i in range(1, 25)])

    rt_demand = pd.DataFrame(index=pd.date_range(start=data[0].index[0], end=data[-1].index[-1]),
                             columns=['rt_demand_t_' + str(i) for i in range(1, 25)])

    da_lmp = pd.DataFrame(index=pd.date_range(start=data[0].index[0], end=data[-1].index[-1]),
                          columns=['da_lmp_t_' + str(i) for i in range(1, 25)])

    rt_lmp = pd.DataFrame(index=pd.date_range(start=data[0].index[0], end=data[-1].index[-1]),
                          columns=['rt_lmp_t_' + str(i) for i in range(1, 25)])

    c = 0
    for df in data:
        for i in range(int(df.shape[0] / 24)):
            da_demand.loc[df.index[i * 24]] = df.loc[df.index[i * 24]].iloc[0:24, 1].values
            rt_demand.loc[df.index[i * 24]] = df.loc[df.index[i * 24]].iloc[0:24, 2].values
            da_lmp.loc[df.index[i * 24]] = df.loc[df.index[i * 24]].iloc[0:24, 3].values
            rt_lmp.loc[df.index[i * 24]] = df.loc[df.index[i * 24]].iloc[0:24, 4].values
            c = c + 1

    delta_lmp = pd.DataFrame(data=rt_lmp.values - da_lmp.values,
                             columns=['delta_lmp_t_' + str(i) for i in range(1, 25)],
                             index=rt_lmp.index)
    delta_demand = pd.DataFrame(data=rt_demand.values - da_demand.values,
                                columns=['delta_dem_t_' + str(i) for i in range(1, 25)],
                                index=rt_lmp.index)

    new_data = pd.concat([delta_demand, delta_lmp], axis=1)
    demand = np.zeros([delta_lmp.shape[0], delta_lmp.shape[1]])
    def_lmp = np.zeros([delta_lmp.shape[0], delta_lmp.shape[1]])
    for i in range(1, 25):
        demand[:, i - 1] = new_data.sort_values(by=['delta_dem_t_' + str(i)], ascending=False).iloc[:, i - 1].values
        def_lmp[:, i - 1] = new_data.sort_values(by=['delta_lmp_t_' + str(i)], ascending=False).iloc[:,
                            24 + i - 1].values

    desired_demand = demand[0:int(demand.shape[0] / 2), :]
    desired_delta = def_lmp[0:int(demand.shape[0] / 2), :]
    actual_demand = demand[int(demand.shape[0] / 2):demand.shape[0], :]
    actual_delta = def_lmp[int(demand.shape[0] / 2):demand.shape[0], :]

    return demand, def_lmp


a=1