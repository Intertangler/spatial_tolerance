import os
import time
import datetime
import sys
import numpy as np
from StringIO import StringIO
import math
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage, imread
from matplotlib.transforms import Bbox
import scipy.optimize as opt
import scipy.stats as stats
import ipdb
import matplotlib.cm as cm
import pandas as pd

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

def exponential_decay_fit(x,a,c,d):
    return a*np.exp(-c*x)+ d


# FUNCTION FOR FINDING INITIAL GUESSES FOR ON AND OFF RATES
def avg_slope_guess(time_points, concentrations,run_data):

    #run_data = fetch_and_normalize_data('data_directory/' + str(run_mono[0]) + '.txt', normalize_const=19.58)

    t1 = np.arange(time_points[1], time_points[2])
    t2 = np.arange(time_points[-1], len(run_data)-1)

    run_data1 = run_data[time_points[1]:time_points[2]]
    run_data2 = run_data[time_points[-1]:-1]
    
    k1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(t1, run_data1)
    r_on_g = k1 / concentrations[1]
    r_on_err = std_err1 / concentrations[1]

    popt, pcov = opt.curve_fit(exponential_decay_fit, t2, run_data2,p0=(1, 1e-6, 1), maxfev=100000)
    r_off_g = popt[1]
    r_off_err = np.sqrt(np.diag(pcov))[1]

    # ipdb.set_trace() # avg slope guess before return
    return r_on_g, r_off_g

def get_save_directory(name):
    global today
    today = datetime.date.today()
    directory_name = str(today) + '__' + str(time.time()) + str(name)
    os.makedirs(directory_name)
    return directory_name

def get_Rmax(struct_file_path, RstrRmax_slope, RstrRmax_interc):
#     plt.figure(figsize=(10,10))
    struct_run = list(pd.read_csv(struct_file_path, sep="\t", header=None)[0])
#     # plt.subplot(521)
#     plt.scatter(range(0, len(struct_run)), struct_run, alpha = 0.5, s = 25, c = "#ff3399")
#     plt.ylabel("RU")
#     plt.ylim(-500,1000)
#     plt.savefig(output_directory+"/ raw_"+master_name+"_monoplot.svg")
#     plt.show()

#     Rstruct = np.mean(struct_run[-750:-50])-np.min(struct_run[250:-400])#np.mean(struct_run[0:250])
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,750),struct_run[-750:])
#     Rstruct = intercept
    Rstruct = intercept-np.mean(struct_run[0:200])
    Rmax = RstrRmax_slope * Rstruct + RstrRmax_interc
#     import pdb; pdb.set_trace()
#     import pdb; pdb.set_trace()
    return Rmax

def fetch_data_raw(data_file_path):
    # ipdb.set_trace()
    with open(str(data_file_path), 'r') as dataFile:
        data1 = dataFile.read()
    actual_run = np.genfromtxt(StringIO(data1), delimiter="\n")
    return actual_run

def fetch_and_normalize_data(data_file_path, normalize_const):
    # ipdb.set_trace()
    with open(str(data_file_path), 'r') as dataFile:
        data1 = dataFile.read()
    actual_run = np.genfromtxt(StringIO(data1), delimiter="\n")
    actual_run = actual_run / (normalize_const) * 1
    return actual_run

    # return np.genfromtxt(StringIO(data), delimiter='\n') / normalize_const

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
class Fitting:
    # attributes of the pattern are its name and coordinates
    def __init__(self, run_list, run_type, mono_rates, KD_biv, concentrations, time_points, scaling_constant,output_dir = "output", struct_params = None,data_dir = 'data_directory/', k_monobiv=None, k_bivmono=None):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.data_dir = data_dir
        self.run_names = run_list[:, 0]
        self.run_type = run_type
        self.transition_matrix_name = list(set(run_list[:, 1]))
        self.run_patterns_pos = np.zeros(len(run_list), dtype=int)
        for i in range(0, len(run_list)):
            for j in range(0, len(self.transition_matrix_name)):
                if run_list[i][1] == self.transition_matrix_name[j]:
                    self.run_patterns_pos[i] = j
                    break
        self.run_data = []
        self.uniformization_rate = 0                                                                                    # flagged for deletion
        if scaling_constant == None:
            self.struct_dir, self.RstrRmax_slope, self.RstrRmax_interc = struct_params
            try:
                self.scaling_constant = [get_Rmax(self.struct_dir+x+"_struct.txt", self.RstrRmax_slope, self.RstrRmax_interc) for x in self.run_names]
#                 import pdb; pdb.set_trace()
            except:
#                 import pdb; pdb.set_trace()
                raise Exception("problem loading structure-binding data - make sure prefix agrees with main sensorgram followed by '_struct.txt'")
        else:
            self.scaling_constant = scaling_constant           

        self.KD_biv = KD_biv
        self.k_on = mono_rates[0]
        self.k_off = mono_rates[1]
        if self.run_type == "biv_scale_only" and k_monobiv == None and k_bivmono == None:
            try:
                self.k_monobiv = 0.02 / self.KD_biv
                self.k_bivmono = self.KD_biv * self.k_monobiv
            except:
                raise Exception("there was a problem assigning bivalent rates from optimization parameters")
        elif self.run_type == "biv_scale_only" and k_monobiv != None and k_bivmono != None:
            try:
                self.k_monobiv = k_monobiv
                self.k_bivmono = k_bivmono
            except:
                raise Exception("there was a problem assigning bivalent rates from optimization parameters")
            
        elif self.run_type == "biv_3param":
            try:
                self.k_monobiv = k_monobiv
                self.k_bivmono = k_bivmono
            except:
                raise Exception("there was a problem assigning bivalent rates from optimization parameters")


#         import pdb
#         pdb.set_trace()
        self.concentrations = concentrations
        self.time_points = time_points



        # ipdb.set_trace()
        self.transition_matrix = []
        self.occupancy_key = []
        # self.distance_matrix = []
        # self.alpha_distance = []
        for i in range(0, len(self.transition_matrix_name)):
            self.transition_matrix.append(np.load(
                str(os.getcwd()) + '/transition_matrices/' + str(self.transition_matrix_name[i]) + '.npy'))
            self.occupancy_key.append(np.load(
                str(os.getcwd()) + '/transition_matrices/' + str(self.transition_matrix_name[i]) + 'occupancy_key.npy'))


#### functionality for importing distance from the coordinate and transition matrix data - not needed for fitting though
        # if run_type == "biv" or run_type=="biv_scale_only":
        #     for i in range(0, len(self.run_names)):
        #         self.distance_matrix.append(np.load( #fetch the matrix of distances for particular antigen configuration
        #             str(os.getcwd()) + '/transition_matrices/' + str(
        #                 run_list[i, 1]) + 'distance_transition_matrix.npy'))
        #         if run_list[i, 1] == 'bivalent_close':
        #             for j in range(0, len(self.distance_matrix[-1])):
        #                 for k in range(j, len(self.distance_matrix[-1])):
        #                     if self.distance_matrix[-1][j, k] != 0:
        #                         self.distance_matrix[-1][j, k] = run_list[i, 2]
        #                         self.distance_matrix[-1][k, j] = run_list[i, 2]
        #     self.alpha_distance = self.distance_matrix
        #     for i in range(0, len(self.run_names)):
        #         for j in range(0, len(self.distance_matrix[i])):
        #             for k in range(j, len(self.distance_matrix[i])):
        #                 self.alpha_distance[i][j, k] = alpha(self.distance_matrix[i][j, k])
        #                 self.alpha_distance[i][k, j] = self.alpha_distance[i][j, k]
        self.print_graph = False
        self.exp_nr = 0
    def set_bounds(self, bounds):
        self.minimizer_kwargs = {"bounds": bounds, "method": "L-BFGS-B"}

    def fit_annealing(self, fitting_number=1):
        self.scaling_constant = self.scaling_constant
        xopt_with_increased_data = []

        ## NUMBER OF RUNS TAKEN FOR FITTING
        #fitting_number = 2

#         variable_parameter = 0  # just so it's defined in any case

#         if self.run_type == "mono":
#             variable_parameter = np.array([self.k_on, self.k_off, self.scaling_constant])
#         elif self.run_type == "biv":
#             variable_parameter = np.array([self.KD_biv, self.scaling_constant])
#         elif self.run_type == "biv_scale_only":
#             variable_parameter = np.array([self.scaling_constant])
#         else:
#             raise Exception("error in fit_annealing: you supplied the wrong run_type")
        self.master_error = np.zeros(fitting_number)
#         log_variable_parameter = np.log10(variable_parameter)
        
        for self.exp_nr in range(0, fitting_number):
            if type(self.scaling_constant) is not list:
                scaling_constant = self.scaling_constant
            else:
                scaling_constant = self.scaling_constant[self.exp_nr]
            variable_parameter = 0  # just so it's defined in any case

            if self.run_type == "mono":
                variable_parameter = np.array([self.k_on, self.k_off, scaling_constant])
            elif self.run_type == "biv":
                variable_parameter = np.array([self.KD_biv, scaling_constant])
            elif self.run_type == "biv_3param":
                variable_parameter = np.array([self.k_monobiv,self.k_bivmono, scaling_constant])    
            elif self.run_type == "biv_scale_only":
                variable_parameter = np.array([scaling_constant])
            else:
                raise Exception("error in fit_annealing: you supplied the wrong run_type")            
#             import pdb; pdb.set_trace()
            log_variable_parameter = np.log10(variable_parameter)
            print("(fit_annealing) variable parameter: ", variable_parameter, "log10 variable parameter: ", log_variable_parameter)
#             import pdb; pdb.set_trace()
            try:
                self.run_data = fetch_and_normalize_data(self.data_dir + str(self.run_names[self.exp_nr]) + '.txt', normalize_const=1)
            except:
                ipdb.set_trace()
                raise Exception("error in fit_annealing: problem loading data from file")
                
                
            if type(self.scaling_constant) is not list: #i.e. if none was fed in, and rmax method was used to pull result
                pass
            else:
#                 print("!!!!!!!!!")
#                 import pdb; pdb.set_trace()
                self.minimizer_kwargs["bounds"] = [(x) for x in self.minimizer_kwargs["bounds"][0:-1]]+[(np.log10(scaling_constant-10),np.log10(scaling_constant+20))]
#             import pdb; pdb.set_trace()
            xopt = opt.basinhopping(self.SPR_run, log_variable_parameter, T=1., stepsize=0.1, disp=True, niter=1, minimizer_kwargs=self.minimizer_kwargs)
#             import pdb; pdb.set_trace()
            # except:import pdb;pdb.set_trace()
            # xopt = opt.minimize(self.SPR_run, log_variable_parameter, method="Nelder-Mead")
            xopt_with_increased_data.append(xopt.x)
            self.print_graph = True
            print(self.run_names[self.exp_nr],[i for i in xopt.x])
            self.SPR_run([i for i in xopt.x])
            self.print_graph = False
        print(xopt_with_increased_data)
        return xopt_with_increased_data

    def uniformize(self, trm):
        infinitessimal_generator_matrix = np.zeros((len(trm), len(trm)))
        for j in range(len(trm)):
            for m in range(len(trm)):
                if j == m:
                    infinitessimal_generator_matrix[j, m] = - np.sum(trm[j, :])
                else:
                    infinitessimal_generator_matrix[j, m] = trm[j, m]
        self.uniformization_rate = max(np.sum(trm, axis=1))
        return np.identity(len(trm)) + infinitessimal_generator_matrix / self.uniformization_rate

    def trm(self, rate_on, rate_off, rate_mono_to_double, rate_double_to_mono, pattern_nr):
        transition_rate_matrix = np.zeros((len(self.transition_matrix[pattern_nr]), len(self.transition_matrix[pattern_nr])))
        for i in range(0, len(self.transition_matrix[pattern_nr])):
            for j in range(0, len(self.transition_matrix[pattern_nr])):
                if self.transition_matrix[pattern_nr][i, j] == 0:
                    pass
                elif self.transition_matrix[pattern_nr][i, j] == 1:
                    transition_rate_matrix[i, j] = rate_on
                elif self.transition_matrix[pattern_nr][i, j] == 2:
                    transition_rate_matrix[i, j] = rate_off
                elif self.transition_matrix[pattern_nr][i, j] == 3:
                    transition_rate_matrix[i, j] = rate_mono_to_double# * self.alpha_distance[self.exp_nr][i, j]
                elif self.transition_matrix[pattern_nr][i, j] == 4:
                    transition_rate_matrix[i, j] = rate_double_to_mono
                else:
                    pass
        # if print_graph == True:
        #     print transition_rate_matrix
        #     plt.imshow(transition_rate_matrix,interpolation='nearest',cmap='jet')
        #     plt.show()
        return transition_rate_matrix


    def SPR_run(self, optimization_parameter):
        #transform the parameters back to non-log scale
        optimization_parameter = np.power(10,np.array(optimization_parameter))

        # set the variable parameter where it belongs based on what is happening
        if self.run_type == "mono":
            rate_on_0 = optimization_parameter[0]
            rate_off_0 = optimization_parameter[1]
            rate_mono_biv_0 = 0
            rate_biv_mono_0 = 0
            scaling_const = optimization_parameter[2]
        elif self.run_type == "biv":
            rate_on_0 = self.k_on
            rate_off_0 = self.k_off
            try:
                rate_mono_biv_0 = 0.02 / optimization_parameter[0]
                rate_biv_mono_0 = optimization_parameter[0] * rate_mono_biv_0
            except:
                raise Exception("there was a problem assigning bivalent rates from optimization parameters")
            scaling_const = optimization_parameter[1]
        elif self.run_type == "biv_scale_only":
            rate_on_0 = self.k_on
            rate_off_0 = self.k_off
            rate_mono_biv_0 = self.k_monobiv
            rate_biv_mono_0 = self.k_bivmono
            scaling_const = optimization_parameter[0]
        elif self.run_type == "biv_3param":
            rate_on_0 = self.k_on
            rate_off_0 = self.k_off
            rate_mono_biv_0 = optimization_parameter[0]  
            rate_biv_mono_0 = optimization_parameter[1]  
            scaling_const = optimization_parameter[2]            
        else:
            raise Exception("error in SPR_run: problem assigning rates")

        stateLabels = np.arange(len(self.transition_matrix[self.run_patterns_pos[self.exp_nr]]))

        # because it will be edited by normalization
        actual_run = self.run_data
        actual_run = actual_run / scaling_const
        timeSamples = len(actual_run)
        # probability vector initialization - this occurs outside the multiconcentration experiment loop
        timeInterval = timeSamples + 1  # seconds
        probabilityVectors = np.zeros((timeInterval, len(stateLabels)))
        probabilityVectors[0][0] = 1.0000  # because everything is starting in state 0 at the beginning of the run

        # make sure to rewrite properly
        time_points = np.append(self.time_points, timeSamples)

        # initialize the occupancy vector as well
        occupancy = np.zeros(timeInterval)
        # weighted_p_vectors = np.zeros((len(self.occupancy_key), timeInterval))
        deltat = 1.0

        for sub_run in range(0, len(time_points) - 1):
            t_lower_bound = time_points[sub_run]
            t_upper_bound = time_points[sub_run + 1]
            sub_run_concentration = self.concentrations[sub_run]

            # initialization of the transition rate matrix according to current concentration and resultant rate
            transitionRateMatrix = self.trm(rate_on_0 * sub_run_concentration, rate_off_0, rate_mono_biv_0, rate_biv_mono_0,
                                            self.run_patterns_pos[self.exp_nr])

            ########        ########        ########        ########        ########        ########        ########
            #                                            UNIFORMIZATION
            ########        ########        ########        ########        ########        ########        ########

            uniformization_matrix = self.uniformize(transitionRateMatrix)
            time_block= range(t_lower_bound, t_upper_bound)

            for t_step in time_block:
                iteration_depth = 15
                pi_i = np.zeros(len(probabilityVectors[t_step + 1]))
                # pi_i = pi_i + probabilityVectors[t_step]
                for poisson_step in range(0, iteration_depth):
                    # e^(-qt)*(qt)^i/i!
                    poisson_factor_i = np.exp(-1 * self.uniformization_rate * deltat) * (
                        math.pow(self.uniformization_rate * deltat, poisson_step)) / math.factorial(poisson_step)
                    if poisson_step == 0:
                        pi_i = pi_i + probabilityVectors[t_step]
                    else:
                        pi_i = np.dot(pi_i, uniformization_matrix)
                    probabilityVectors[t_step + 1] = probabilityVectors[t_step + 1] + pi_i * poisson_factor_i
#                 if np.sum( probabilityVectors[t_step + 1]) < 1:
#                     print("warning - leakage")
                probabilityVectors[t_step + 1] = probabilityVectors[t_step + 1]/np.sum(probabilityVectors[t_step + 1]) # renormalization - should not be necessary if no leakage though
#                     if np.sum( probabilityVectors[t_step + 1]) < 1:
#                         iteration_depth += 10
#                     else: leakage = False
                    # if t_step == 475:
                    #    print 'step: ', poisson_step, '\n ////p vector t: \n', probabilityVectors[t_step], '\n ////p vector t+1: \n', probabilityVectors[t_step+1],'\n uniformization rate q: ',uniformization_rate,'\n poisson factor: ', poisson_factor_i, '\n ::::::::inf. gen. matrix Q: \n ', infinitessimal_generator_matrix,'\n ::::::::uniformaziation matrix: \n', uniformization_matrix, '\n ::::::::transition matrix: \n', transitionRateMatrix
                for state in range(0, len(self.occupancy_key[self.run_patterns_pos[self.exp_nr]])):
                    occupancy[t_step + 1] = occupancy[t_step + 1] + probabilityVectors[t_step + 1, state] * \
                                            self.occupancy_key[self.run_patterns_pos[self.exp_nr]][state]
                    # weighted_p_vectors[state]

        #if self.print_graph:
            #ipdb.set_trace() #sse calcs come up next
        # ipdb.set_trace()
        self.residuals = (occupancy[0:-1] - actual_run)
        k = 50
        krange = range(0, k)
        total_auto_correlation = 0
        autocorrelations = np.zeros((len(self.residuals) - k + 1))
        for i in range(0, len(self.residuals) - k + 1):
            k_mer = self.residuals[i:i + k]
            autocorrelations[i] = np.sqrt(np.correlate(krange, k_mer) ** 2) / k
            # total_auto_correlation += np.abs(np.correlate(krange, k_mer))**2 / k
        # sse = (occupancy[0:-1] - actual_run)**2
        #sse = ((occupancy[0:-1] - actual_run) * ((np.linspace(1, timeSamples, timeSamples)) ** 0.1)) ** 2
        # master_error = total_auto_correlation#+np.max(autocorrelations)*len(autocorrelations)#+np.sum(sse)**.1
        master_error = np.sum(autocorrelations*np.abs(self.residuals[0:-(k-1)]))
        # print(master_error)
        # ipdb.set_trace() # This is where I plot dif
        #self.autocorrelation = np.correlate(moving_average(residuals))
        if self.print_graph:

            plt.rc('xtick', labelsize=25)
            plt.rc('ytick', labelsize=25)
            plt.rcParams.update({'font.size': 25})

            plt.figure(figsize=(10, 5))
            plt.scatter(range(len(actual_run)),actual_run, alpha = 0.5, s = 25, c = "#518394")
            plt.plot(range(0, len(occupancy)),occupancy,c="#ff3399", linewidth=2)
            plt.ylabel("$\\Phi$ $[n_{Ab}/n_{struct}]$",fontsize=25)
            plt.xlabel("time [s]",fontsize=25)
            plt.ylim(0, 2)
            plt.savefig(self.output_dir+"/SPR_curve_fitted_result_overlay"+self.run_names[self.exp_nr]+".svg")
            plt.show()
            plt.close()

            print(optimization_parameter)
            plt.figure(figsize=(10, 5))
            plt.scatter(range(0, len(actual_run)),self.residuals, alpha = 0.5, s = 10, c = "r")
            plt.ylabel("residuals $[n_{Ab}/n_{struct}]$",fontsize=25)
            plt.xlabel("time [s]",fontsize=25)
            plt.ylim(-1,1)
            plt.savefig(self.output_dir+"/SPR_curve_fit_residuals"+self.run_names[self.exp_nr]+".svg")
            plt.show()
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.stackplot(range(0, len(self.residuals) - k + 1),
                     autocorrelations * np.abs(self.residuals[0:-(k - 1)]), colors = ["r"])
            plt.ylim(0, 10)
            plt.ylabel("autocorrelation-weighted \n residuals $[n_{Ab}/n_{struct}]$",fontsize=25)
            plt.xlabel("time [s]",fontsize=25)
            plt.savefig(self.output_dir+"/autocorr_weighted_residuals" + self.run_names[self.exp_nr] + ".svg")
            plt.show()
            plt.close()

            vals = np.linspace(0, 1, len(probabilityVectors[0]))
            np.random.shuffle(vals)
            cmap_main = cm.get_cmap('coolwarm')
            cmap = cm.colors.ListedColormap(cmap_main(vals))

            state_printout_limit = 5
            # import pdb; pdb.set_trace()
            # if len(self.occupancy_key[0]) > state_printout_limit:
            end_rankings = np.argsort(probabilityVectors[-1])[::-1]
            # ranked_occupancy = self.occupancy_key[0][end_rankings]
            # ranked_final_probabilities = probabilityVectors[-1][end_rankings]
            # sorted_states = np.sort(np.array([probabilityVectors[-1], self.occupancy_key[0], np.arange(0, len(self.occupancy_key[0]),1)]),axis=1).T[::-1]
            # cmap_sorted = sorted_states.T[3]
            # import pdb; pdb.set_trace()
            # subscript = str(int(sorted_states[i][2]))
            labels = ["$\\sigma_{" + str(int(i)) + \
                      "}$: $\\phi_{"+str(int(i))+"}$ = " + str(int(self.occupancy_key[0][i]))+ \
                      ", $p^{t_{final}}_{"+str(int(i))+"}$ = "+ str(np.round(probabilityVectors[-1][i],decimals=3)) for i in end_rankings[0:state_printout_limit]]

            # ["$s_{" + str(int(sorted_states[i][2])) + "}$: $O_{" + str(int(sorted_states[i][2])) + "}$ = " + str(
            #     int(sorted_states[i][1])) + ", $p^{t_final}_{" + str(int(sorted_states[i][2])) + "}$ = " + str(
            #     np.round(sorted_states[i][0], decimals=3)) for i in sorted_states[0:state_printout_limit]]
            # else:
            #     labels = ["$s_{"+str(i)+"}$: "+str(int(self.occupancy_key[0][i])) for i in range(0, len(self.occupancy_key[0]))]
            # plt.figure(figsize=(10, 5))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stackplot(range(0, len(self.run_data) + 1), probabilityVectors.T[end_rankings], colors=cmap.colors, labels=labels)
            ax.legend(loc='upper left',fontsize=15)
            plt.ylabel("state probability \n $p_{i}$",fontsize=25)
            plt.ylim((0, 1.0))
            plt.xlabel("t (sec)",fontsize=25)
            plt.xlim((0, len(self.run_data) + 1))
            # plt.tight_layout()
            plt.savefig(self.output_dir + "/stack_probabilities" + self.run_names[self.exp_nr] + ".svg")
            plt.show()
            plt.close()




            # plt.figure(figsize=(10, 5))
            fig, ax = plt.subplots(figsize=(10, 5))
            weighted_ps = np.zeros((len(self.occupancy_key[0])),dtype=np.ndarray)
            for state in range(0,len(self.occupancy_key[0])):
                # import pdb; pdb.set_trace()
                weighted_ps[state] = probabilityVectors.T[state]*self.occupancy_key[0][state]
            occupancies = np.array([np.array(i, dtype=float) for i in weighted_ps])
            # import pdb; pdb.set_trace()
            end_rankings_weighted = np.argsort(occupancies.T[-1])[::-1]
            weighted_labels = ["$\\sigma_{" + str(int(i)) +  "}$: $\\phi_{" + str(int(i)) + "}p^{t_{final}}_{" + str(int(i)) + "}$ = " + str( np.round(probabilityVectors[-1][i] * self.occupancy_key[0][i], decimals=3)) for i in end_rankings_weighted[0:state_printout_limit]]
            # import pdb; pdb.set_trace()
            ax.stackplot(range(len(weighted_ps[0])), occupancies[end_rankings], colors=cmap.colors,labels=weighted_labels)
            # import pdb; pdb.set_trace()
            ax.legend(loc='upper left',fontsize=15)
            plt.scatter(range(len(actual_run)),actual_run, s = 30, c = "k")
            plt.ylabel("occupancy \n $p_{i}\\phi_{i} \ [n_{Ab}/n_{struct}]$",fontsize=25)
            plt.ylim((0, 2.0))
            plt.xlabel("t (sec)",fontsize=25)
            plt.xlim((0, len(self.run_data) + 1))
            plt.tight_layout()
            plt.savefig(self.output_dir + "/occupancies" + self.run_names[self.exp_nr] + ".svg")
            
            plt.show()
            plt.close()
        try:self.master_error[self.exp_nr] = master_error
        except:import pdb; pdb.set_trace()
#         import pdb; pdb.set_trace()
        return master_error


# def alpha(distance): # the function which determines the distance dependence of k mono to bi,
#     if distance == 0:
#         slope_m = 0
#         intercept_b = 0
#     elif distance < 2:
#         slope_m = .0553
#         intercept_b = -2.082e-17
#     elif distance < 2.4:
#         slope_m = .210
#         intercept_b = -.310
#     elif distance < 3.4:
#         slope_m = .805
#         intercept_b = -1.738
#     elif distance < 6.8:
#         slope_m = -.007
#         intercept_b = 1.0247
#     elif distance < 14.28:
#         slope_m = .003
#         intercept_b = .955
#     elif distance < 15.8:
#         slope_m = -.010
#         intercept_b = 1.138
#     elif distance < 16.74:
#         slope_m = -.482
#         intercept_b = 8.599
#     elif distance < 21:
#         slope_m = -.109
#         intercept_b = 2.358
#     else:
#         slope_m = 0
#         intercept_b = 0
#     a = distance*slope_m+intercept_b
#     return a
