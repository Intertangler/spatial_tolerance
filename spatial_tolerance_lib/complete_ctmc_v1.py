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


def exponential_decay_fit(x,a,c,d):
    return a*np.exp(-c*x)+ d


def get_save_directory(name):
    global today
    today = datetime.date.today()
    directory_name = str(today) + '__' + str(time.time()) + str(name)
    os.makedirs(directory_name)
    return directory_name

def compute_KD(parameters):
    """\
    compute the interconversion factor based on analytical spatial tolerance model parameters

    Parameters
    ------------
    l_t
        tension characteristic length term
    alpha_t
        tension modulus term
    l_c
        compression characteristic length term
    alpha_c
        compression modulus term
    KD2max
        max KD2 value
    distances
        range of input separation distances
        
    Returns:
    -----------
    combine
        range of KD2 values corresponding to input distances and the combine impact of compression and tension terms of spatial tolerance
    """
    l_t,alpha_t,l_c,alpha_c,KD2max,distances = parameters
    compression = KD2max*np.exp(-1*alpha_c*distances-l_c)
    tension = KD2max/(1+np.exp(-1*alpha_t*(distances-l_t)))
    
    combine = (compression+tension)
    return combine



class Simulation:
    """\
    Template for a stand-alone simulation instance.

    """
    # attributes of the pattern are its name and coordinates
    def __init__(self, transition_matrix_name, mono_rates, KD_fx, concentrations, time_points, final_time = 2869,output_dir="output",experiment_name = "ctmc",Temp = 298,kB = 1.3806e-23):
        """
        Attributes
        ------------
        transition_matrix_name
            name of the corresponding transition matrix for the pattern to be simulated
        mono_rates
            k1 and k-1 rates
        KD_fx
            analytical function parameters for bivalent interconversion
        concentrations
            list of concentrations corresponding to different injection steps during the run
        time_points
            list of time points corresponding to different injection steps during the run

        Keyword arguments
        ------------
        final_time
            final timepoint of the simulation
        output_dir
            default output directory
        experiment_name
            name of experiment
        Temp
            temperature, default set to 298 Kelvin
        kB
            Boltzmann constant        

        """
        self.Temp = Temp  # kelvin
        self.kB = kB  # J /K / particle
        self.mu_mono = 1.805e-20 #J/antibody
        self.mu_biv = 2.693e-20 #J/antibody
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.transition_matrix_name = transition_matrix_name
        self.final_time = final_time
        self.KD_fx = KD_fx
        self.uniformization_rate = 0


        self.k_on = mono_rates[0]
        self.k_off = mono_rates[1]
        # self.k_monobiv = rates[2]
        # self.k_bivmono = rates[3]

        self.concentrations = concentrations
        self.time_points = time_points

        self.master_error = 0

        self.distance_matrix = []
        self.alpha_distance = []

        self.transition_matrix = np.load(str(os.getcwd()) + '/transition_matrices/' + str(self.transition_matrix_name) + '.npy')
        self.occupancy_key = np.load(str(os.getcwd()) + '/transition_matrices/' + str(self.transition_matrix_name) + 'occupancy_key.npy')
        self.particle_count_key = np.load(str(os.getcwd()) + '/transition_matrices/' + str(self.transition_matrix_name) + 'particle_count_key.npy')
        self.distance_matrix = np.load(str(os.getcwd()) + '/transition_matrices/' + str(self.transition_matrix_name) + 'distance_transition_matrix.npy')

        self.print_graph = False

    def uniformize(self, trm):
        """
        performs transition matrix uniformization operation
        
        Parameters:
        -----------
        trm
            transition rate matrix to be uniformized
        
        """
        infinitessimal_generator_matrix = np.zeros((len(trm), len(trm)))
        for j in range(len(trm)):
            for m in range(len(trm)):
                if j == m:
                    infinitessimal_generator_matrix[j, m] = - np.sum(trm[j, :])
                else:
                    infinitessimal_generator_matrix[j, m] = trm[j, m]
        self.uniformization_rate = max(np.sum(trm, axis=1))
        return np.identity(len(trm)) + infinitessimal_generator_matrix / self.uniformization_rate

    def trm(self, rate_on, rate_off):
        """
        Assemble transition rate matrix from analytical KD2 function
        
        Parameters:
        -----------
        rate_on
            k_1 binding rate monovalent
        rate_off
            k_-1 dissociation rate monovalent
        
        Return
        -----------
        transition_rate_matrix
            a transition rate matrix populated with entries according to separation distance and input rates
        """        
        kbm = 0.19781887754
        transition_rate_matrix = np.zeros((len(self.transition_matrix), len(self.transition_matrix)))
        for i in range(0, len(self.transition_matrix)):
            for j in range(0, len(self.transition_matrix)):
                if self.transition_matrix[i, j] == 0:
                    pass
                elif self.transition_matrix[i, j] == 1:
                    transition_rate_matrix[i, j] = rate_on
                elif self.transition_matrix[i, j] == 2:
                    transition_rate_matrix[i, j] = rate_off
                elif self.transition_matrix[i, j] == 3:
                    transition_rate_matrix[i, j] = kbm/compute_KD(self.KD_fx + [self.distance_matrix[i, j]])
                elif self.transition_matrix[i, j] == 4:
                    transition_rate_matrix[i, j] = kbm
                else:
                    pass

        return transition_rate_matrix





    #### EQUILIBRUM CALCULATIONS #####

    def infinitessimal_generator_matrix(self,trm):
        """
        Assemble infinitessimal generator matrix from an input transition rate matrix.
        
        Parameters:
        -----------
        trm
            the input transition rate matrix
        
        Return
        -----------
        infinitessimal_generator_matrix
            the output infinitessimal generator matrix corresponding to the supplied rate matrix
        """            
        infinitessimal_generator_matrix = np.zeros((len(trm), len(trm)))
        for j in range(len(trm)):
            for m in range(len(trm)):
                if j == m:
                    infinitessimal_generator_matrix[j, m] = - np.sum(trm[j, :])
                else:
                    infinitessimal_generator_matrix[j, m] = trm[j, m]
        return infinitessimal_generator_matrix

    def get_stationary_dist(self, guess_p_vector, args):
        """
        Objective function used to evaluate whether stationary distribution has been found.
        
        Parameters:
        -----------
        guess_p_vector
            guess a probability distribution that will be iteratively modified until it satisfies sationarity
        args
            additional arguments - Q the infinitessimal generator matrix
            
        Return
        -----------
        stability_cond
            the objective function value to be minimized to determine stationarity
        """            
        guess_p_vector = 10**guess_p_vector
        Q = args
        stability_cond = np.sum(1+np.abs(np.dot(np.abs(np.nan_to_num(guess_p_vector)), Q)))**2
        return stability_cond

    def normalization_condition(self,guess_p_vector):
        """
        A constraint - all probabilities must sum to 1 
        
        Parameters:
        -----------
        guess_p_vector
            the supplied probability distribution
        
        Return
        -----------
        norm_cond
            objective function output that must satisfy constraint
        """    
        guess_p_vector = 10**guess_p_vector
        norm_cond = (np.sum(np.abs(guess_p_vector))-1)*100000
        return norm_cond
    def explosive_p_condition(self,guess_p_vector):
        """
        A constraint - probabilities must not explode
        
        Parameters:
        -----------
        guess_p_vector
            the supplied probability distribution
        
        Return
        -----------
        product
            objective function output that must satisfy constraint
        """            
        guess_p_vector = 10**guess_p_vector
        product = np.abs(np.prod(guess_p_vector))*10**(len(guess_p_vector))
        return product

    def ineq_constraint(self,guess_p_vector):
        """
        A constraint - finds the minimum item in the probability distribution
        
        Parameters:
        -----------
        guess_p_vector
            the supplied probability distribution
        
        Return
        -----------
        np.min(guess_p_vector)
            objective function output that must satisfy constraint
        """            
        guess_p_vector = 10**guess_p_vector
        return np.min(guess_p_vector)
    
    def no_negatives(self,input_vector):
        return np.min(input_vector)
    
    def no_neg_energies(self,x):
        return -1 * self.kB * self.Temp * np.log((x))
    
    def stationary_distribution(self, concentration, guess_init):
        """
        Numerically solve for the stationary distribution given an initial guess probability distribution and a specified antibody concentration.
        
        Parameters:
        -----------
        concentration
            stationary distribution is defined for a given concentration value
        guess_init
            initial probability distribution that will be numerically adjusted until it satisfies stationarity
        
        Return
        -----------
        10**xopt.x
            result of the optimization process - stationary distribution and its associated score
        """              
        self.concentration = concentration
        transitionRateMatrix = self.trm(self.k_on * concentration, self.k_off)
        Tx = transitionRateMatrix
        Q = self.infinitessimal_generator_matrix(Tx)

        # init[0] = 1
        cons = [{'type': 'eq', 'fun': self.normalization_condition},
                {'type':'ineq', 'fun': self.ineq_constraint},]
#                {'type':'ineq', 'fun': self.explosive_p_condition}]
#         soln = np.linalg.lstsq(Q,np.zeros(len(guess_init)),rcond=-1)
        xopt = opt.minimize(self.get_stationary_dist,
                            np.log10(guess_init),
                            args=(Q),
                            method="SLSQP",
                            constraints=cons,
                            options={'maxiter': 10000, 'ftol': 1e-25})
#         xopt = opt.basinhopping(self.get_stationary_dist, guess_init, 
#             T=1e-8, stepsize=0.05, disp=False, niter=1,interval=10,
#             minimizer_kwargs = {"args": (Q), "method": "SLSQP","constraints":cons})
#         if np.round(np.sum(xopt.x),5) != 1:

        return 10**xopt.x#soln[3]#

#backup partition function wouth chem potentials
#     def compare_equilibrium_ps(self, exp_energies):
#         #     exp_energies = np.exp(-1*guess_energies/(kB * Temp))
#         exp_energies = np.abs(exp_energies)
        
#         partition_function = np.sum(exp_energies)
#         calcd_ps = exp_energies / partition_function
#         errors = np.abs(calcd_ps - self.equilibrium_p_dist)
#         sse = np.sum(errors)
#         return sse

#     def solve_partition_function(self, equilibrium_p_dist,verbose=True):

#         self.equilibrium_p_dist = equilibrium_p_dist
#         guess_energies = equilibrium_p_dist * 10 ** -20
#         exp_energies = np.exp(-1 * guess_energies / (self.kB * self.Temp))

#         # initial guess status
#         partition_function = np.sum(exp_energies)
#         calcd_ps = exp_energies / partition_function
#         errors = (calcd_ps - equilibrium_p_dist) ** 2
#         sse = np.sum(errors)
#         cons = [{'type':'ineq', 'fun': self.ineq_constraint}]
#         xopt = opt.minimize(self.compare_equilibrium_ps, exp_energies, method='Nelder-Mead', jac=None, hess=None,
#                             hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
# #         xopt = opt.basinhopping(self.compare_equilibrium_ps, exp_energies, 
# #                         T=1e-23, stepsize=100, disp=False, niter=400,interval=10,)
#         final_partition_function = np.sum(xopt.x)
#         final_optimized_ps = np.abs(xopt.x) / final_partition_function
# #         ipdb.set_trace()
#         final_energies = -1 * self.kB * self.Temp * np.log(np.abs(xopt.x))
#         verbose = True
#         if verbose == True:
#             print("solved partition function: ", final_partition_function)
#             print("output equilibrium probabilities: ", final_optimized_ps)
#             print("input equilibrium probabilities: ", equilibrium_p_dist)
#             print("solved energies: ", final_energies)  # in units of J / particle
#         return [final_partition_function, np.abs(final_energies)]

    def compare_equilibrium_ps(self, exp_energies):
        exp_energies = exp_energies        
        partition_function = np.sum(exp_energies*np.exp(-1*self.chem_pot/(self.kB * self.Temp))*(1))        
        calcd_ps = (exp_energies*np.exp(-1*self.chem_pot/(self.kB * self.Temp))*(1)) / partition_function
        errors = np.abs(calcd_ps - self.equilibrium_p_dist)
        sse = np.sum(errors)
        return sse

    def solve_partition_function(self, equilibrium_p_dist,verbose=True):
        """
        Numerically solve for the partition function given an input stationary probability distribution.
        
        Parameters:
        -----------
        equilibrium_p_dist
            stationary distribution for a particular set of system conditions
                
        Return
        -----------
        final_partition_function
            thermodynamic normalization factor for the stationary distribution
        final_energies
            corresponding energies of each state for the stationary distribution
        """            
        self.equilibrium_p_dist = equilibrium_p_dist
        self.chem_pot = self.particle_count_key.T[0]*self.mu_mono*self.equilibrium_p_dist + self.particle_count_key.T[1]*self.mu_biv*self.equilibrium_p_dist
        guess_energies = equilibrium_p_dist * 10 ** -18
        exp_energies = np.exp(-1 * guess_energies / (self.kB * self.Temp))
        partition_function = np.sum(exp_energies)
        calcd_ps = exp_energies / partition_function
        errors = (calcd_ps - equilibrium_p_dist) ** 2
        sse = np.sum(errors)
        cons = [{'type':'ineq', 'fun': self.no_negatives}]
#                {'type':'ineq', 'fun': self.no_neg_energies}]
        
        xopt = opt.minimize(self.compare_equilibrium_ps, exp_energies, method="SLSQP", jac=None, hess=None,
                            hessp=None, bounds=None, constraints=cons, tol=None, callback=None, options=None)
#         xopt = opt.basinhopping(self.compare_equilibrium_ps, exp_energies, 
#                         T=1e-23, stepsize=100, disp=False, niter=400,interval=10,)
        final_partition_function = np.sum(xopt.x*np.exp(-1*self.chem_pot/(self.kB * self.Temp))*(1))
        final_optimized_ps = (xopt.x*np.exp(-1*self.chem_pot/(self.kB * self.Temp))*(1)) / final_partition_function
#         ipdb.set_trace()
        final_energies = -1 * self.kB * self.Temp * np.log(xopt.x-np.min(xopt.x))#*np.exp(-1*self.chem_pot/(self.kB * self.Temp))*(1))
#         verbose = True
        if verbose == True:
            print("solved partition function: ", final_partition_function)
            print("output equilibrium probabilities: ", final_optimized_ps)
            print("input equilibrium probabilities: ", equilibrium_p_dist)
            print("solved energies: ", final_energies)  # in units of J / particle
#         ipdb.set_trace()
        return [final_partition_function, final_energies]

    def get_state_entropies(self, equilibrium_p_dist):
        entropies = np.zeros((len(self.occupancy_key)))
        for state in range(0, len(self.occupancy_key)):
            entropies[state] = -1 * self.kB * equilibrium_p_dist[state] * np.log(equilibrium_p_dist[state])
        return entropies

    
    ### TRANSIENT CALCULATIONS ###

    def SPR_run(self):
        """
        Main simulation function that starts from an initial condition probability distribution.
        Takes Markov steps in time using the method of uniformized transition matrix to compute transient evolution of probability distribution.
        Produces a stratified visualization of the different states, their associated probabilities, and how they change with time.
        Also produces a stratified visualization of occupancy, i.e. each state's contribution to the SPR signal.
        
        Results produced:
        ----------
        self.occupancy 
            occupancy information at each time point for each state
        self.probability_vectors 
            probability of each state at each time point
        
        """          
        # set the variable parameter where it belongs based on what is happening

        rate_on_0 = self.k_on
        rate_off_0 = self.k_off

        stateLabels = np.arange(len(self.transition_matrix))
        
        timeSamples = self.final_time
        # probability vector initialization - this occurs outside the multiconcentration experiment loop
        timeInterval = timeSamples + 1  # seconds
        probabilityVectors = np.zeros((timeInterval, len(stateLabels)))
        probabilityVectors[0][0] = 1.0000  # because everything is starting in state 0 at the beginning of the run

        # make sure to rewrite properly
        time_points = np.append(self.time_points, timeSamples)

        # initialize the occupancy vector as well
        occupancy = np.zeros(timeInterval)
        deltat = 1.0

        for sub_run in range(0, len(time_points) - 1):
            t_lower_bound = time_points[sub_run]
            t_upper_bound = time_points[sub_run + 1]
            sub_run_concentration = self.concentrations[sub_run]

            # initialization of the transition rate matrix according to current concentration and resultant rate
            transitionRateMatrix = self.trm(rate_on_0 * sub_run_concentration, rate_off_0)
            # self.transitionRateMatrix = transitionRateMatrix
            # import pdb; pdb.set_trace()

            ########        ########        ########        ########        ########        ########        ########
            #                                            UNIFORMIZATION
            ########        ########        ########        ########        ########        ########        ########
            
            uniformization_matrix = self.uniformize(transitionRateMatrix)
            # self.uniformization_matrix = uniformization_matrix
            for t_step in range(t_lower_bound, t_upper_bound):
                iteration_depth = 30
                # if t_step == 475:
                #    print '\n \n  begin!!!'
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
                    # if t_step == 475:
                    #    print 'step: ', poisson_step, '\n ////p vector t: \n', probabilityVectors[t_step], '\n ////p vector t+1: \n', probabilityVectors[t_step+1],'\n uniformization rate q: ',uniformization_rate,'\n poisson factor: ', poisson_factor_i, '\n ::::::::inf. gen. matrix Q: \n ', infinitessimal_generator_matrix,'\n ::::::::uniformaziation matrix: \n', uniformization_matrix, '\n ::::::::transition matrix: \n', transitionRateMatrix
                probabilityVectors[t_step + 1] = probabilityVectors[t_step + 1]/np.sum(probabilityVectors[t_step + 1]) # renormalization - should not be necessary if no leakage though
                for state in range(0, len(self.occupancy_key)):
                    occupancy[t_step + 1] = occupancy[t_step + 1] + probabilityVectors[t_step + 1, state] * self.occupancy_key[state]

            
        if self.print_graph:

            print(probabilityVectors)
            plt.rc('xtick', labelsize=25)
            plt.rc('ytick', labelsize=25)
            plt.rcParams.update({'font.size': 25})

            plt.figure(figsize=(10, 5))
            plt.plot(range(0, len(occupancy)), occupancy, c="#ff3399", linewidth=2)
            plt.ylabel("occupancy \n $\\Omega \ [Ab/struct]$", fontsize=25)
            plt.xlabel("time [s]", fontsize=25)
            plt.ylim(0, 4)
            plt.xlim(0, timeInterval)
            plt.savefig(self.output_dir + "/occupancy" + self.experiment_name + ".svg")
            plt.show()
            plt.close()

            vals = np.linspace(0, 1, len(probabilityVectors[0]))
            np.random.shuffle(vals)
            cmap_main = cm.get_cmap('coolwarm')
            cmap = cm.colors.ListedColormap(cmap_main(vals))
            self.statecolors = cmap.colors

            state_printout_limit = 3

            end_rankings = np.argsort(probabilityVectors[-1])[::-1]
            self.end_rankings = end_rankings
            labels = ["$\\sigma_{" + str(int(i)) + "}$: $\\omega_{" + str(int(i)) + "}$ = " + str(
                int(self.occupancy_key[i])) + ", $p^{t_{final}}_{" + str(int(i)) + "}$ = " + str(
                np.round(probabilityVectors[-1][i], decimals=3)) for i in end_rankings[0:state_printout_limit]]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stackplot(range(0, timeInterval), probabilityVectors.T[end_rankings], colors=cmap.colors,
                         labels=labels)
            ax.legend(loc='upper left', fontsize=15)
            plt.ylabel("state probability", fontsize=25)
            plt.ylim((0, 1))
            plt.xlabel("t (sec)", fontsize=25)
            plt.xlim((0, timeInterval))
            # plt.tight_layout()
            plt.savefig(self.output_dir + "/stack_probabilities" + self.experiment_name + ".svg")
            plt.show()
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 5))
            weighted_ps = np.zeros((len(self.occupancy_key)),dtype=np.ndarray)
            for state in range(0,len(self.occupancy_key)):
                # import pdb; pdb.set_trace()
                weighted_ps[state] = probabilityVectors.T[state]*self.occupancy_key[state]
            occupancies = np.array([np.array(i, dtype=float) for i in weighted_ps])
            # import pdb; pdb.set_trace()
            end_rankings_weighted = np.argsort(occupancies.T[-1])[::-1]
            weighted_labels = ["$\\sigma_{" + str(int(i)) +  "}$: $\\omega_{" + str(int(i)) + "}p^{t_{final}}_{" + str(int(i)) + "}$ = " + str( np.round(probabilityVectors[-1][i] * self.occupancy_key[i], decimals=3)) for i in end_rankings_weighted[0:state_printout_limit]]
            # import pdb; pdb.set_trace()
            ax.stackplot(range(len(weighted_ps[0])), occupancies[end_rankings_weighted], colors=cmap.colors,labels=weighted_labels)
            # import pdb; pdb.set_trace()
            ax.legend(loc='upper left',fontsize=15)
            plt.ylabel("occupancy \n $p_i \\omega_i \ [Ab/struct]$",fontsize=25)
            plt.ylim((0, 4.0))
            plt.xlabel("t (sec)",fontsize=25)
            plt.xlim((0, timeInterval))
            plt.tight_layout()
            plt.savefig(self.output_dir + "/occupancies" + self.experiment_name + ".svg")
            plt.show()
            plt.close()
        self.occupancy = occupancy
        self.probability_vectors = probabilityVectors
    
        return 0

    def SPR_run_advanced(self):

            rate_on_0 = self.k_on
            rate_off_0 = self.k_off

            stateLabels = np.arange(len(self.transition_matrix))

            timeSamples = self.final_time
            # probability vector initialization - this occurs outside the multiconcentration experiment loop
            timeInterval = timeSamples + 1  # seconds
            probabilityVectors = np.zeros((timeInterval, len(stateLabels)))
            probabilityVectors[0][0] = 1.0000  # because everything is starting in state 0 at the beginning of the run

            # make sure to rewrite properly
            time_points = np.append(self.time_points, timeSamples)

            # initialize the occupancy vector as well
            occupancy = np.zeros(timeInterval)
            deltat = 1.0
            entropies = []
            energies = []    
            free_energies = []    
            average_energy = []
            total_entropy = []
            energy_levs=[]
            free_energies_eq=[]
            chem_pots = []
            partition_test=[]
            boltzmann_factors = []
            mono_bonds = []
            biv_bonds = []
            
            for sub_run in range(0, len(time_points) - 1):
                t_lower_bound = time_points[sub_run]
                t_upper_bound = time_points[sub_run + 1]
                sub_run_concentration = self.concentrations[sub_run]
                self.subconc = sub_run_concentration

                # initialization of the transition rate matrix according to current concentration and resultant rate
                transitionRateMatrix = self.trm(rate_on_0 * sub_run_concentration, rate_off_0)
                # self.transitionRateMatrix = transitionRateMatrix
                # import pdb; pdb.set_trace()

                ########        ########        ########        ########        ########        ########        ########
                #                                            UNIFORMIZATION
                ########        ########        ########        ########        ########        ########        ########

                uniformization_matrix = self.uniformize(transitionRateMatrix)
                # self.uniformization_matrix = uniformization_matrix
                
                try: 
                    stationary_distribution = self.stationary_distribution_suggestion[sub_run]
#                     ipdb.set_trace()
                except:
                    stationary_distribution = self.stationary_distribution(concentration = sub_run_concentration,guess_init=np.ones((len(self.occupancy_key))) / len(self.occupancy_key)) #e-9 M 
                Z, e_vector = self.solve_partition_function(stationary_distribution,verbose=False)
                for t_step in range(t_lower_bound, t_upper_bound):
                    iteration_depth = 30
                    # if t_step == 475:
                    #    print '\n \n  begin!!!'
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
                        # if t_step == 475:
                        #    print 'step: ', poisson_step, '\n ////p vector t: \n', probabilityVectors[t_step], '\n ////p vector t+1: \n', probabilityVectors[t_step+1],'\n uniformization rate q: ',uniformization_rate,'\n poisson factor: ', poisson_factor_i, '\n ::::::::inf. gen. matrix Q: \n ', infinitessimal_generator_matrix,'\n ::::::::uniformaziation matrix: \n', uniformization_matrix, '\n ::::::::transition matrix: \n', transitionRateMatrix
                    probabilityVectors[t_step + 1] = probabilityVectors[t_step + 1]/np.sum(probabilityVectors[t_step + 1]) # renormalization - should not be necessary if no leakage though
                    for state in range(0, len(self.occupancy_key)):
                        occupancy[t_step + 1] = occupancy[t_step + 1] + probabilityVectors[t_step + 1, state] * self.occupancy_key[state]

                    
                    energy = e_vector*probabilityVectors[t_step]
                    entropy = self.kB*probabilityVectors[t_step]*np.log(probabilityVectors[t_step])
                    #-1*self.kB*(1/q)*np.exp(-1*np.array(energy)/(self.Temp*self.kB))*(np.log(1/q)-np.array(energy)/(self.Temp*self.kB))
                    #-1*self.get_state_entropies(probabilityVectors[t_step])
                    chem_pot = self.particle_count_key.T[0]*self.mu_mono*np.array(probabilityVectors[t_step]) + self.particle_count_key.T[1]*self.mu_biv*np.array(probabilityVectors[t_step])
                    entropies.append(entropy)
                    energies.append(energy)
                    chem_pots.append(chem_pot)
                    boltzmann_factor = np.exp(-1*e_vector/(self.kB * self.Temp))*np.exp(-1*self.chem_pot/(self.kB * self.Temp))
                    boltzmann_factors.append(boltzmann_factor)
#                     ipdb.set_trace()
                    grand_pot = self.Temp*self.kB*np.log(1)
                    energy_levs.append(np.array(e_vector))
#                     total_entropy.append(np.sum(entropy)*self.Temp*-1)
#                     average_energy.append(np.sum(energy))
#                     solution_potential = 
                    free_energies.append(energy - chem_pot + self.Temp*entropy)#- self.Temp*entropy
#                     ipdb.set_trace()
                    mono_bonds.append(self.particle_count_key.T[0]*np.array(probabilityVectors[t_step]))
                    biv_bonds.append(self.particle_count_key.T[1]*np.array(probabilityVectors[t_step]))
#                     ipdb.set_trace()
                    free_energy_eq =np.array(e_vector)*np.array(stationary_distribution) +(self.particle_count_key.T[0]*self.mu_mono*np.array(stationary_distribution) + self.particle_count_key.T[1]*self.mu_biv*np.array(stationary_distribution))+ self.Temp*self.kB*stationary_distribution*np.log(stationary_distribution)
                    free_energies_eq.append(free_energy_eq)#-1*self.Temp*self.kB*np.log(Z))#np.sum(e_vector*probabilityVectors[t_step]) - self.Temp*np.sum(entropy))
                    partition_test.append(-1*self.Temp*self.kB*np.log(Z))# - np.sum(self.Temp*self.kB*stationary_distribution*np.log(stationary_distribution)))
#                     free_energies_eq.append(np.sum(e_vector*probabilityVectors[t_step]) + self.Temp*np.sum(entropy))
#                     ipdb.set_trace()
#                 ipdb.set_trace()
            energy_levs = np.array(energy_levs)
            energies = np.array(energies)
            entropies = np.array(entropies)
            chem_pots = np.array(chem_pots)
            free_energies = np.array(free_energies)
            free_energies_eq = np.array(free_energies_eq)
            partition_test = np.array(partition_test)
            boltzmann_factors = np.array(boltzmann_factors)
            mono_bonds = np.array(mono_bonds)
            biv_bonds = np.array(biv_bonds)
            
            if self.print_graph:

                print(probabilityVectors)
                plt.rc('xtick', labelsize=25)
                plt.rc('ytick', labelsize=25)
                plt.rcParams.update({'font.size': 25})

#                 plt.figure(figsize=(10, 5))
#                 plt.plot(range(0, len(occupancy)), occupancy, c="#ff3399", linewidth=2)
#                 plt.ylabel("occupancy \n $\\Omega \ [Ab/struct]$", fontsize=25)
#                 plt.xlabel("time [s]", fontsize=25)
#                 plt.ylim(0, 4)
#                 plt.xlim(0, timeInterval)
#                 plt.savefig(self.output_dir + "/occupancy" + self.experiment_name + ".svg")
#                 plt.show()
#                 plt.close()

                vals = np.linspace(0, 1, len(probabilityVectors[0]))
                np.random.shuffle(vals)
                cmap_main = cm.get_cmap('coolwarm')
                cmap = cm.colors.ListedColormap(cmap_main(vals))
                self.statecolors = cmap.colors
                state_printout_limit = 5
                end_rankings = np.argsort(probabilityVectors[-1])[::-1]
                self.end_rankings = end_rankings                
                plabels = ["$\\sigma_{" + str(int(i)) + "}$: $\\omega_{" + str(int(i)) + "}$ = " + str(
                    int(self.occupancy_key[i])) + ", $p^{t_{final}}_{" + str(int(i)) + "}$ = " + str(
                    np.round(probabilityVectors[-1][i], decimals=3)) for i in end_rankings[0:state_printout_limit]]

                weighted_ps = np.zeros((len(self.occupancy_key)),dtype=np.ndarray)
                for state in range(0,len(self.occupancy_key)):
                    weighted_ps[state] = probabilityVectors.T[state]*self.occupancy_key[state]
                occupancies = np.array([np.array(i, dtype=float) for i in weighted_ps])
                end_rankings_weighted = np.argsort(occupancies.T[-1])[::-1]
                weighted_labels = ["$\\sigma_{" + str(int(i)) +  "}$: " + str( np.round(probabilityVectors[-1][i] * self.occupancy_key[i], decimals=3)) + "$Ab/struct$" for i in end_rankings_weighted[0:state_printout_limit]]
                
                end_rankings_s = np.argsort(entropies[-1])[::-1]
#                 ipdb.set_trace()
                s_labels = ["$\\sigma_{" + str(int(i)) +  "}$: " + str( np.round(10**21*entropies[-1][i]*self.Temp, decimals=3)) + " $\\cdot 10^{-21} J$" for i in end_rankings_s[0:state_printout_limit]]                
                end_rankings_e = np.argsort(entropies[-1])[::-1]
#                 ipdb.set_trace()
                e_labels = ["$\\sigma_{" + str(int(i)) +  "}$: " + str( np.round(10**20*energies[-1][i], decimals=3)) + " $\\cdot 10^{-20} J$" for i in end_rankings_e[0:state_printout_limit]]                
                elev_labels = ["$\\sigma_{" + str(int(i)) +  "}$: " + str( np.round(10**20*energy_levs[-1][i], decimals=3)) + " $\\cdot 10^{-20} J$" for i in end_rankings_e[0:state_printout_limit]]                
                                

                #### PROBABILITY ####
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.stackplot(range(0, timeInterval), probabilityVectors.T[end_rankings], colors=cmap.colors[end_rankings],
                             labels=plabels)
                ax.legend(loc='upper left', fontsize=15)
                plt.ylabel("state probability", fontsize=25)
                plt.ylim((0, 1))
                plt.xlabel("t (sec)", fontsize=25)
                plt.xlim((0, timeInterval))
                # plt.tight_layout()
                plt.savefig(self.output_dir + "/stack_probabilities" + self.experiment_name + ".svg")
                plt.show()
                plt.close()

                #### OCCUPANCY ####
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.stackplot(range(len(weighted_ps[0])), occupancies[end_rankings_weighted], colors=cmap.colors[end_rankings_weighted],labels=weighted_labels,edgecolor=None)
                ax.legend(loc='upper left',fontsize=15)
                plt.ylabel("occupancy \n $p_i \\omega_i \ [n_{Ab}/n_{struct}]$",fontsize=25)
                plt.ylim((0, 4.0))
                plt.xlabel("t (sec)",fontsize=25)
                plt.xlim((0, timeInterval))
                plt.tight_layout()
                plt.savefig(self.output_dir + "/occupancies" + self.experiment_name + ".svg")
                plt.show()
                plt.close()
                
                
                
                #### ENTROPIC  ####
                fig, ax = plt.subplots(figsize=(10, 5))
                
                plt.stackplot(range(len(occupancy)-1), -1*self.Temp*entropies.T[end_rankings_s], colors=cmap.colors[end_rankings_s],labels=s_labels,edgecolor=None)
                ax.legend(loc='upper left',fontsize=15)
                plt.ylabel("entropy \n $ s_i \ [J]$",fontsize=25)
                plt.xlabel("t (sec)",fontsize=25)
                plt.savefig(self.output_dir + "/entropies" + self.experiment_name + ".svg")
                plt.show()
                plt.close()
                
                
                #### AVERAGE ENERGY ####
                fig, ax = plt.subplots(figsize=(10, 5))
#                 plt.plot(range(len(occupancy)-1), free_energies, linewidth = 5,c="#B1323F")
                plt.stackplot(range(len(occupancy)-1), (np.array(energies)-np.min(energies)).T[end_rankings_e], colors=cmap.colors[end_rankings_e],labels=e_labels,edgecolor=None)
                ax.legend(loc='upper left',fontsize=15)
#                 plt.ylim((0, 4.0))
                plt.ylabel("average state energy \n $E_i p_i \ [J]$",fontsize=25)
                plt.xlabel("t (sec)",fontsize=25)
                plt.savefig(self.output_dir + "/energies" + self.experiment_name + ".svg")
                plt.show()
                plt.close()
                
                #### Energy allocation ####
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.stackplot(range(len(occupancy)-1), np.array(energy_levs).T[end_rankings_e], colors=cmap.colors[end_rankings_e],labels=elev_labels,edgecolor=None)
                ax.legend(loc='upper left',fontsize=15)
                plt.ylabel("state energy levels \n $E_i \ [J]$",fontsize=25)
                plt.xlabel("t (sec)",fontsize=25)
#                 plt.ylim((0, 1.0))
                plt.savefig(self.output_dir + "/energies" + self.experiment_name + ".svg")
                plt.show()
                plt.close()
                
                #### state accessibilities ####
                fig, ax = plt.subplots(figsize=(10, 5))
#                 plt.plot(range(len(occupancy)-1), free_energies, linewidth = 5,c="#B1323F")
                plt.stackplot(range(len(occupancy)-1), boltzmann_factors.T[end_rankings_e], colors=cmap.colors[end_rankings_e],labels=e_labels,edgecolor=None)
                ax.legend(loc='upper left',fontsize=15)
#                 plt.ylim((0, 4.0))
                plt.ylabel("state accessibility",fontsize=25)
                plt.xlabel("t (sec)",fontsize=25)
                plt.savefig(self.output_dir + "/accessibilities" + self.experiment_name + ".svg")
                plt.show()
                plt.close()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.plot(range(len(occupancy)-1), np.sum(energies,1), linewidth = 3,c="#F1B895",linestyle='dashed')
                plt.plot(range(len(occupancy)-1), np.sum(chem_pots,1), linewidth = 3,c="#242C9F",linestyle='dashed')
                plt.plot(range(len(occupancy)-1), np.sum(entropies*self.Temp*-1,1), linewidth = 4,c="#93B6CD",linestyle='dashed')
                plt.plot(range(len(occupancy)-1), np.sum(free_energies,1), linewidth = 3,c="#B1323F")
                plt.scatter(range(len(occupancy)-1), np.sum(free_energies_eq,1), s = 3,c="r")#,linestyle='dashed')
                plt.scatter(range(len(occupancy)-1), partition_test, s = 3,c="k")#,linestyle='dashed')
#                 ax.legend(["$\\langle E \\rangle$", "$\\mu_{mono} \\langle M_{mono} \\rangle + \\mu_{biv} \\langle M_{biv} \\rangle$", "$-TS$""$\\Phi$","$\\Phi^*$"])
#                 plt.ylim((0, 4.0))
#                 plt.plot(range(len(occupancy)-1), np.array(free_energies).T, colors=cmap.colors,labels=weighted_labels,edgecolor=None)
                plt.ylabel("grand potential free energy \n $\\mathbb{Z}= \\langle E \\rangle - TS  - (\\mu_{0,1} \\langle M_{1} \\rangle + \\mu_{0,2} \\langle M_{2} \\rangle) [J]$",fontsize=25)            
                plt.xlabel("t (sec)",fontsize=25)
                plt.savefig(self.output_dir + "/free_energies" + self.experiment_name + ".svg")
                plt.show()
                plt.close()
#                 ipdb.set_trace()

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.plot(range(len(occupancy)-1), np.sum(biv_bonds,1), linewidth = 3,c="#F1B895")
                plt.plot(range(len(occupancy)-1), np.sum(mono_bonds,1), linewidth = 3,c="#242C9F")
#                 ax.legend(["$\\langle E \\rangle$", "$\\mu_{mono} \\langle M_{mono} \\rangle + \\mu_{biv} \\langle M_{biv} \\rangle$", "$-TS$""$\\Phi$","$\\Phi^*$"])
#                 plt.ylim((0, 4.0))
#                 plt.plot(range(len(occupancy)-1), np.array(free_energies).T, colors=cmap.colors,labels=weighted_labels,edgecolor=None)
                plt.ylabel("particles per structure \n $\\langle M_{1} \\rangle, \\langle M_{2} \\rangle$",fontsize=25)            
                plt.xlabel("t (sec)",fontsize=25)
                plt.savefig(self.output_dir + "/bonds_per_struct" + self.experiment_name + ".svg")
                plt.show()
                plt.close()

            self.occupancy = occupancy
            self.probability_vectors = probabilityVectors

            return 0