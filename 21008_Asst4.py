import random
import numpy as np
import pandas as pd
import math
from random import shuffle
import itertools
from scipy import stats
from scipy.stats.mstats import gmean
from matplotlib import pyplot as plt


def Relativ_Entropy(fr_0, fr_1):
    m_params = 24
    l_params = 250
    bias = 1/(2*m_params*l_params)
    r_entropy = []
    assert len(fr_0) == len(fr_1)
    count_fr_data = 0
    for i in range(len(fr_0)):
        if np.isnan(fr_0[i]) or np.isnan(fr_1[i]):
            pass
        else:
            if fr_0[i] > bias:
                r_n_entropy = fr_0[i]*math.log((fr_0[i] - bias)/(fr_1[i] + bias)) - fr_0[i] + fr_1[i]
                                            
                r_entropy.append(r_n_entropy)
            else:
                r_entropy.append(fr_1[i])
            count_fr_data += 1

    return np.sum(r_entropy)/count_fr_data



def Avg_Searchtime(reaction_times, baseline):
    return np.nanmean(reaction_times) - baseline



def get_l1_distance(fr_0, fr_1):
    l1_distance = []
    assert len(fr_0) == len(fr_1)
    count_fr_data = 0
    for i in range(len(fr_0)):
        if np.isnan(fr_0[i]) or np.isnan(fr_1[i]):
            pass
        else:
            n_l1 = abs(fr_0[i] - fr_1[i])
            l1_distance.append(n_l1)
        count_fr_data += 1

    return np.sum(l1_distance)/len(fr_1)




class Gamma_Distribution_Fitting:

    def __init__(self, Searchtime_Data):
        self._Searchtime_Data = Searchtime_Data
        self.g_count = Searchtime_Data.shape[1]
        self.r_group = None
        self.Avg_list = []
        self.div_list = []
        self.shape = None
        self.rate = None

    def Shape_Params(self):
        Params_mat = np.polyfit(self.Avg_list, self.div_list, deg=1, full=True)
        self.shape = 1/math.pow(Params_mat[0][0], 2)

    def Rand_groups(self):
        r_num_list = []
        while len(r_num_list) < self.g_count//2:
            r_num = random.randint(0, self.g_count-1)
            if r_num not in r_num_list:
                r_num_list.append(r_num)
        self.r_group = r_num_list

    def Mean_Sdv(self):
        for index in self.r_group:
            S_T_Col = np.array(self._Searchtime_Data.values[:, index][2:]).astype(np.float)
            mean = np.nanmean(S_T_Col)
            stddev = math.sqrt(np.nanvar(S_T_Col))
            self.Avg_list.append(mean)
            self.div_list.append(stddev)
         

        

    def Rate_Params(self):
        o_groups = []
        l_mean = []
        l_var = []
        p_cdf = []
        for index in range(self.g_count):
            if index not in self.r_group:
                o_groups.append(index)

        for index in o_groups:
            S_T_Col = np.array(self._Searchtime_Data.values[:, index][2:]).astype(np.float)
            S_T_Col_1 = [time for time in S_T_Col if str(time) != 'nan']
            shuffle(S_T_Col_1)
            r_searchtimes = S_T_Col_1[0:len(S_T_Col_1)//2]
            p_cdf.append(S_T_Col_1[len(S_T_Col_1)//2:len(S_T_Col_1)])
            mean = np.nanmean(r_searchtimes)
            variance = np.nanvar(r_searchtimes)
            l_mean.append(mean)
            l_var.append(variance)

         

        Params_mat = np.polyfit(l_mean, l_var, deg=1, full=True)
        self.rate = 1/Params_mat[0][0]
        print('Rate parameter :', self.rate)
        print('Shape parameter :', self.shape)
        # Plot empirical gamma distribution against theoretical gamma distribution
        p_cdf = list(itertools.chain.from_iterable(p_cdf))
        s_cdf = np.sort(p_cdf)
        cdf_emp = np.arange(len(s_cdf)) / float(len(s_cdf) - 1)
        plt.plot(s_cdf, cdf_emp)
        x_gamma = np.linspace(0, s_cdf[-1], 200)
        y_gamma = stats.gamma.cdf(x_gamma, a=self.shape, scale=1/self.rate)
        plt.plot(x_gamma, y_gamma, color='r')
        plt.show()
        plt.close()

        y_pdf = stats.gamma.rvs(size=len(p_cdf), a=self.shape, scale=1 / self.rate)
        ks_test = stats.ks_2samp(s_cdf, y_pdf)
        print(' The Kolmogorov-Smirnov statistic and P_value', ks_test)

    def Plot_Mean_Sdv(self):
        axis = plt.subplot(111)
        plt.xlabel('Mean')
        plt.ylabel('Standard Deviation')
        axis.scatter(self.Avg_list, self.div_list, color='r')
        plt.show()
        plt.close()

    




class Line_Fitting:

    def __init__(self, Searchtime_Data, Firingrate_Data):
        self._Searchtime_Data = Searchtime_Data
        self._Firingrate_Data = Firingrate_Data
        self.Avg_Search_Time = []
        self.Relative_Data = []
        self.L1_Data = []
        self.Inv_Searchtime = []
        self.Entopy_Ratio = None
        self.L1_Ratio = None

    def Calculate_Searchtime_Avg(self):
        Search_Data_len = self._Searchtime_Data.shape[1]
        for i in range(Search_Data_len):
            S_T_Col = np.array(self._Searchtime_Data.values[:, i][2:]).astype(np.float)
            Searchtime_Avg =  Avg_Searchtime(S_T_Col, 328)
            self.Avg_Search_Time.append(Searchtime_Avg)
            self.Inv_Searchtime = [1000 / search_time for search_time in self.Avg_Search_Time]
        return self.Avg_Search_Time

    def Calculate_Entropy(self):
        count = 4
        col_count = 6
        for i in range(count):
            if i != 3:
                for j in range(col_count // 2):
                    col_indx = i * col_count + 2 * j
                    fr_0 = np.array(self._Firingrate_Data.values[:, col_indx][2:]).astype(np.float)
                    fr_1 = np.array(self._Firingrate_Data.values[:, col_indx + 1][2:]).astype(np.float)
                    entropy_01 =  Relativ_Entropy(fr_0, fr_1)
                    self.Relative_Data.append(entropy_01)
                    l1_01 =  get_l1_distance(fr_0, fr_1)
                    self.L1_Data.append(l1_01)

                    entropy_10 =  Relativ_Entropy(fr_1, fr_0)
                    self.Relative_Data.append(entropy_10)
                    l1_10 =  get_l1_distance(fr_1, fr_0)
                    self.L1_Data.append(l1_10)
            else:
                for j in range(3):
                    col_indx = i * col_count + 2 * j
                    fr_0 = np.array(self._Firingrate_Data.values[:, col_indx][2:]).astype(np.float)
                    fr_1 = np.array(self._Firingrate_Data.values[:, col_indx + 2][2:]).astype(np.float)
                    entropy_01_1 =  Relativ_Entropy(fr_0, fr_1)
                    l1_01_1 =  get_l1_distance(fr_0, fr_1)
                    entropy_10_1 =  Relativ_Entropy(fr_1, fr_0)
                    l1_10_1 =  get_l1_distance(fr_1, fr_0)

                    fr_0 = np.array(self._Firingrate_Data.values[:, col_indx][2:]).astype(np.float)
                    fr_1 = np.array(self._Firingrate_Data.values[:, col_indx + 3][2:]).astype(np.float)
                    entropy_01_2 =  Relativ_Entropy(fr_0, fr_1)
                    l1_01_2 =  get_l1_distance(fr_0, fr_1)
                    entropy_10_2 =  Relativ_Entropy(fr_1, fr_0)
                    l1_10_2 =  get_l1_distance(fr_1, fr_0)

                    fr_0 = np.array(self._Firingrate_Data.values[:, col_indx + 1][2:]).astype(np.float)
                    fr_1 = np.array(self._Firingrate_Data.values[:, col_indx + 2][2:]).astype(np.float)
                    entropy_01_3 =  Relativ_Entropy(fr_0, fr_1)
                    l1_01_3 =  get_l1_distance(fr_0, fr_1)
                    entropy_10_3 =  Relativ_Entropy(fr_1, fr_0)
                    l1_10_3 =  get_l1_distance(fr_1, fr_0)

                    fr_0 = np.array(self._Firingrate_Data.values[:, col_indx + 1][2:]).astype(np.float)
                    fr_1 = np.array(self._Firingrate_Data.values[:, col_indx + 3][2:]).astype(np.float)
                    entropy_01_4 =  Relativ_Entropy(fr_0, fr_1)
                    l1_01_4 =  get_l1_distance(fr_0, fr_1)
                    entropy_10_4 =  Relativ_Entropy(fr_1, fr_0)
                    l1_10_4 =  get_l1_distance(fr_1, fr_0)

                    entropy_01= np.mean([entropy_01_1, entropy_01_2, entropy_01_3,
                                                   entropy_01_4])
                    l1_01 = np.mean([l1_01_1, l1_01_2, l1_01_3, l1_01_4])
                    self.Relative_Data.append(entropy_01)
                    self.L1_Data.append(l1_01)

                    entropy_10 = np.mean([entropy_10_1, entropy_10_2, entropy_10_3,entropy_10_4])                                                   
                    l1_10 = np.mean([l1_10_1, l1_10_2, l1_10_3, l1_10_4])
                    self.Relative_Data.append(entropy_10)
                    self.L1_Data.append(l1_10)

        
        return self.Relative_Data, self.L1_Data


    def Straightline_Fitting(x, y):
        x = x[:,np.newaxis]
        i, r, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return i,r
        

    def Am_Gm_Spread(self):
        entropy_prod = np.multiply(self.Avg_Search_Time, self.Relative_Data)
         
        L1_prod = np.multiply(self.Avg_Search_Time, self.L1_Data)
         

        entropy_prod_AM= np.mean(entropy_prod)
        entropy_prod_GM = gmean(entropy_prod)
         

        L1_prod_AM = np.mean(L1_prod)
        L1_prod_GM = gmean(L1_prod)
         

        self.Entopy_Ratio = entropy_prod_AM / entropy_prod_GM
        print('The Ratio of AM and GM for relative entropy with search time: ' + str(self.Entopy_Ratio))

        self.L1_Ratio = L1_prod_AM / L1_prod_GM
        print('The Ratio of AM and GM for L1 distance with search time: ' + str(self.L1_Ratio))

        return self.Entopy_Ratio, self.L1_Ratio


    def Plot_Entropy(self):
        axis = plt.subplot(111)
        plt.xlabel('Relative Entropy distance')
        plt.gca().set_ylabel(r'$s^{-1}$')
        axis.scatter(self.Relative_Data, self.Inv_Searchtime, c='red')
        slope, r_error = Line_Fitting.Straightline_Fitting(np.array(self.Relative_Data),np.array(self.Inv_Searchtime))                                                             
        axis.plot(self.Relative_Data, slope*self.Relative_Data)
        plt.show()
        plt.close()

    def Plot_L1(self):
        axis = plt.subplot(111)
        plt.xlabel('L1 distance')
        plt.gca().set_ylabel(r'$s^{-1}$')
        axis.scatter(self.L1_Data, self.Inv_Searchtime, c='red')
        slope, r_error = Line_Fitting.Straightline_Fitting(np.array(self.L1_Data),np.array(self.Inv_Searchtime))
        axis.plot(self.L1_Data, slope * self.L1_Data)
        plt.show()
        plt.close()






if __name__ == '__main__':

    Searchtime_Data = pd.read_csv('../data/02_data_visual_neuroscience_searchtimes.csv') 
    Firingrate_Data = pd.read_csv('../data/02_data_visual_neuroscience_firingrates.csv')
    
     

    
    # First part - Fiting a straight line
    line_fit = Line_Fitting(Searchtime_Data, Firingrate_Data)
    line_fit.Calculate_Searchtime_Avg()
    line_fit.Calculate_Entropy()
    line_fit.Plot_L1()
    line_fit.Plot_Entropy()
    line_fit.Am_Gm_Spread()

    # Second part - The Gamma Distribution 
    gamma_fitter = Gamma_Distribution_Fitting(Searchtime_Data)
    gamma_fitter.Rand_groups()
    gamma_fitter.Mean_Sdv()
    gamma_fitter.Plot_Mean_Sdv()
    gamma_fitter.Shape_Params()
    gamma_fitter.Rate_Params()

