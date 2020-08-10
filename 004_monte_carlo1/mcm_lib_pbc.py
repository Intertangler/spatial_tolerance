#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime
import time as time
from StringIO import StringIO
import scipy.optimize as opt
import ipdb
import os
from matplotlib.axes import Axes as ax
from numpy.random import choice
import re
import cmath
import csv
import bisect
from tqdm import tqdm


def get_save_directory(name):
    # print 'get_save_directory'
    global directory_name
    global today
    today = datetime.date.today()
    directory_name = str(today) + '__' + str(time.time()) + str(name)
    os.makedirs(directory_name)
    return directory_name


def intersect(p1,p2,p3,p4, box_size):

    # ipdb.set_trace()
    border_1 = int(max(abs(round((p1[0] - p2[0]) / box_size)), abs(round((p1[1] - p2[1]) / box_size))))
    border_2 = int(max(abs(round((p3[0] - p4[0]) / box_size)), abs(round((p3[1] - p4[1]) / box_size))))

    if border_1 == 0 and border_2 == 0:
        # if intersect_default(p1,p2,p3,p4) is True:
        #     if not np.array_equal(p1,p3) and not np.array_equal(p1,p4) and not np.array_equal(p2,p3) and not np.array_equal(p2,p4):
        #         ipdb.set_trace() #ohIdontwannahearyousay
        return intersect_default(p1,p2,p3,p4)


    elif border_1 == 1:
        p1_border, p2_border = intersect_border_points(p1, p2, box_size)

        if border_2 != 1:
            if intersect_default(p1, p2_border, p3, p4):
                return True
            if intersect_default(p1_border, p2, p3, p4):
                return True

            return False

        else:
            p3_border, p4_border = intersect_border_points(p3, p4, box_size)

            if intersect_default(p1, p2_border, p3, p4_border):
                return True
            if intersect_default(p1_border, p2, p3, p4_border):
                return True
            if intersect_default(p1, p2_border, p3_border, p4):
                return True
            if intersect_default(p1_border, p2, p3_border, p4):
                return True

            return False

    elif border_2 == 1:
        # ipdb.set_trace()  # semetoki
        p3_border, p4_border = intersect_border_points(p3, p4, box_size)

        if intersect_default(p1, p2, p3_border, p4):
            return True
        elif intersect_default(p1, p2, p3, p4_border):
            return True
        else:
            return False
    else:
        print("border or intersection error, evaluate boundaries or antigen topology")
        ipdb.set_trace()


def intersect_border_points(p1, p2, box_size):
    # move one of the particles "closer" by bringing it out of the box
    p1_temp = [0, 0]
    p1_border = [0, 0]
    p2_border = [0, 0]

    p1_temp[0] = p1[0] - round((p1[0] - p2[0]) / box_size) * box_size #p1 gets moved "outside" of the box
    p1_temp[1] = p1[1] - round((p1[1] - p2[1]) / box_size) * box_size

    # get the gradient
    x_border = round(p2[0] / box_size) * box_size #closest borders to p2
    y_border = round(p2[1] / box_size) * box_size

    if (p2[0] - p1_temp[0]) != 0 and (p2[1] - p1_temp[1]) != 0: #if p1 and p2 are not vertical or horizontal to each other
        gradient1 = (p2[0] - p1_temp[0]) / (p2[1] - p1_temp[1])  #get gradient between p2 and p1_temp

        len_to_x_border = (p2[0] - x_border) ** 2 + ((p2[0] - x_border) / gradient1) ** 2 # find length to border from p2
        len_to_y_border = (p2[1] - y_border) ** 2 + ((p2[1] - y_border) * gradient1) ** 2 # in both dimensions

        if len_to_x_border < len_to_y_border: #if x border is closer
            p1_border[0] = x_border
            p1_border[1] = p2[1] - (p2[0] - p1_border[0]) / gradient1
            p2_border[0] = box_size * (1 - x_border / box_size)
            p2_border[1] = p1[1] + (p2_border[0] - p1[0]) / gradient1

        else: # if y border is closer
            p1_border[1] = y_border # p2 y border is the closest border
            p1_border[0] = p2[0] - (p2[1] - p1_border[1]) * gradient1 # p2 x coord at y border found by gradient
            p2_border[1] = box_size * (1 - y_border / box_size) # p1 y border opposite side of the box
            p2_border[0] = p1[0] + (p2_border[1] - p1[1]) * gradient1 # p1 x coord at y border found by gradient

    elif p2[0] - p1_temp[0] == 0:
        p1_border[0] = p2[0]
        p1_border[1] = y_border
        p2_border[0] = p2[0]
        p2_border[1] = box_size * (1 - y_border / box_size)

    else:
        p1_border[0] = x_border
        p1_border[1] = p2[1]
        p2_border[0] = box_size * (1 - x_border / box_size)
        p2_border[1] = p2[1]

    return p1_border, p2_border



def intersect_default(p1, p2, p3, p4):  # p1 and p2 is one line and p3 p4 the other line
    X1, X2, X3, X4 = p1[0], p2[0], p3[0], p4[0]
    Y1, Y2, Y3, Y4 = p1[1], p2[1], p3[1], p4[1]

    if max(X1, X2) < min(X3, X4) or max(Y1, Y2) < min(Y3, Y4) or max(X3, X4) < min(X1, X2) or max(Y3, Y4) < min(Y1, Y2):
        return False  # There is no mutual abcisse

    elif (p1[0] == p2[0] and p1[1] == p2[1]) or (p3[0] == p4[0] and p3[1] == p4[1]):
        return False # periodic point was on the border and two points of one "line" are actually a dot -> crossover impossible or excluded by other half of the line

    elif X1 != X2 and X3 != X4:
        A1 = (Y1 - Y2) / (X1 - X2)
        A2 = (Y3 - Y4) / (X3 - X4)
        b1 = Y1 - A1 * X1  # = Y2 - A1 * X2
        b2 = Y3 - A2 * X3  # = Y4 - A2 * X4
        if A1 == A2:
            return False
        else:
            Xa = (b2 - b1) / (A1 - A2)
            if Xa < max(min(X1, X2), min(X3, X4)) or Xa > min(max(X1, X2), max(X3, X4)):
                return False  # intersection is out of bound
            else:
                return True
    else:
        if Y1 != Y2 and Y3 != Y4:
            A1 = (X1 - X2) / (Y1 - Y2)
            A2 = (X3 - X4) / (Y3 - Y4)
            b1 = X1 - A1 * Y1  # = Y2 - A1 * X2
            b2 = X3 - A2 * Y3  # = Y4 - A2 * X4
            if A1 == A2:
                return False
            else:
                Xa = (b2 - b1) / (A1 - A2)
                if Xa < max(min(Y1, Y2), min(Y3, Y4)) or Xa > min(max(Y1, Y2), max(Y3, Y4)):
                    return False  # intersection is out of bound
                else:
                    return True

        else:
            return True


class Pattern:


    # attributes of the pattern are its name and coordinates
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        self.n_sites = len(self.coordinates)
        self.pointer_vector = (self.n_sites + 1) * np.ones(self.n_sites)
        self.site_status_vector = np.zeros((self.n_sites))
        self.graph_step = 0

        # done

    def configure_ab_ag_system_pbc(self, cutoff_distance, box_size):
        self.cutoff_distance = cutoff_distance
        self.distance_matrix = np.zeros((self.n_sites, self.n_sites))
        self.box_size = box_size
        # create the distance matrix, every pair of coordinates (antigens) is checked once
        self.neighbor_matrix = np.zeros((self.n_sites, self.n_sites))
        self.neighbor_directory = np.empty((self.n_sites), dtype=np.ndarray)
        for pair in range(0, self.n_sites):
            for compare_pair in range(pair, self.n_sites):
                distance_btw_pairs = np.sqrt(((self.coordinates[compare_pair][0]) - (self.coordinates[pair][0])
                                             - round(((self.coordinates[compare_pair][0]) - (self.coordinates[pair][0])) / self.box_size) * self.box_size) ** 2
                                             + ((self.coordinates[compare_pair][1]) - (self.coordinates[pair][1])
                                             - round(((self.coordinates[compare_pair][1]) - (self.coordinates[pair][1])) / self.box_size) * self.box_size) ** 2)  # shortest distance in case of pbc
                self.distance_matrix[pair, compare_pair] = distance_btw_pairs
                self.distance_matrix[compare_pair, pair] = distance_btw_pairs
                if distance_btw_pairs < self.cutoff_distance and distance_btw_pairs != 0:

                    self.neighbor_matrix[pair, compare_pair] = 1
                    self.neighbor_matrix[compare_pair, pair] = 1
                    self.neighbor_directory[pair] = np.append(self.neighbor_directory[pair], np.array([compare_pair]))
                    self.neighbor_directory[compare_pair] = np.append(self.neighbor_directory[compare_pair],
                                                                      np.array([pair]))
                else:
                    pass
        # ipdb.set_trace()
        return self.distance_matrix
        # done


    ## used in the main file
    def output_site_configuration(self):
        # print neighbor_directory, ' \n'
        plt.figure(figsize=(10, 10))

        for i in range(0, len(self.coordinates)):
            try:
                self.neighbor_list = self.neighbor_directory[i][1:]
            except:
                self.neighbor_list = np.array([])
            for neighbor in range(0, len(self.neighbor_list)):
                plt.plot([self.coordinates[i][1], self.coordinates[self.neighbor_list[neighbor]][1]],
                         [self.coordinates[i][0], self.coordinates[self.neighbor_list[neighbor]][0]], 'k-', linewidth=5,
                         alpha=1)
        if not os.path.exists(str(
                os.getcwd()) + '/' + 'State_Space'):  # test to se if a folder for storing the data exists
            os.makedirs(str(os.getcwd()) + '/' + 'State_Space')  # if no , create the folder
        if not os.path.exists(str(
                os.getcwd()) + '/' + 'State_Space/' +
                              self.name):
            os.makedirs(str(os.getcwd()) + '/' +
                        'State_Space/' + self.name)  # if no , create the folder
        plt.title('antigen configuration')
        plt.scatter(self.coordinates[:, 1], self.coordinates[:, 0], c='w', s=2000)
        plt.ylim(np.min([self.coordinates[:, 0], self.coordinates[:, 1]]) - 3, np.max([self.coordinates[:, 0],
                                                                                       self.coordinates[:, 1]]) + 3)
        plt.xlim(np.min([self.coordinates[:, 0], self.coordinates[:, 1]]) - 3, np.max([self.coordinates[:, 0],
                                                                                       self.coordinates[:, 1]]) + 3)
        plt.savefig(str(os.getcwd()) + '/' + 'State_Space/' +
                    self.name + '/' + "lattice_config" + ".png")
        plt.savefig(str(os.getcwd()) + '/' + 'State_Space/' +
                    self.name + '/' + "lattice_config" + ".svg")
        plt.close()
        # done


    def initialize_antibody_environment(self, rate_constants, directory_path):
        self.rate_constants = rate_constants
        self.directory_path = directory_path
        self.k_on = rate_constants[0]
        self.k_off = rate_constants[1]
        self.k_mono_bi = rate_constants[2]
        self.k_bi_mono = rate_constants[3]
        self.k1 = 1
        self.k_1 = -1
        self.k2 = 2
        self.k_2 = -2

        ## MOVE TO MULTIRUN FUNCTION
        self.cumulative_walking_time = 0
        self.walk_record = np.zeros(
            (10000, 4))  # 3 things to keep track of: state ID, holding time, cumulative walking time, and occupancy record
        self.id_record = []
        self.occupancy_record = np.zeros((self.time_samples))
        self.step = 0
        self.current_state = 0
        # done


    ################################################################################
    # Initialize a program, create empty vectors, input parameters for experiment
    def program_kinetics(self, time_points, concentration):
        self.time_samples = time_points[-1]
        self.time_points = time_points
        self.molar_concentrations = concentration
        self.program = 'universal'


    ############################################################################

    def multirun(self, nr_of_runs):  # timepoint is an array 1 element longer than conce. e.g. last element is total time.
        print(self.name)
        total_time = self.time_points[-1]
        self.dt = 10
        self.cumul_length = total_time / self.dt
        self.final_average_times = np.zeros(self.cumul_length)  # array of final averaged vals per uniform timestep
        self.total_writing_option = 'w'
        for run_nr in tqdm(range(1, nr_of_runs + 1)):
            # SETUP FOR EACH REC
            self.walk_record = np.zeros((self.time_samples + 1, 4))
            self.list_of_state_ids = []
            self.state_discoveries = 0
            self.cumulative_walking_time = 0
            self.walk_record = np.zeros((10000,
                                         4))  # 3 things to keep track of: state ID, holding time, cumulative walking time, and occupancy record
            self.id_record = []
            self.occupancy_record = np.zeros((self.time_samples))
            self.step = 0
            self.current_state = 0
            self.graph_step = 0
            self.site_status_vector = np.zeros((self.n_sites))
            self.pointer_vector = (self.n_sites + 1) * np.ones(self.n_sites)

#             print(run_nr)
            for conc_step in range(len(self.molar_concentrations)):
                # print(concentration[conc_step])
                self.random_walk(conc_step, run_nr)

            self.save_states_and_times(run_nr)
            self.total_writing_option = 'a+'
            # self.avg_out()
            # ipdb.set_trace()

            # if run_nr % 10 == 0:
            #     self.generate_average_occupancy_plot(run_nr, total_time)

        for i in range(0, self.cumul_length):
            self.final_average_times[i] = self.final_average_times[i] / nr_of_runs


    def avg_out(self):
        curr_run_index = 0
        tot_run_index = 1
        cumulative_time_run = np.array(self.walk_record)[:, 2]
        occ_run = np.array(self.walk_record)[:, 3]
        len_occ_run = len(occ_run)
        while tot_run_index in range(1, self.cumul_length) and curr_run_index < len_occ_run - 1:
            if cumulative_time_run[curr_run_index] < tot_run_index * self.dt < cumulative_time_run[curr_run_index + 1]:
                self.final_average_times[tot_run_index] = self.final_average_times[tot_run_index] + occ_run[curr_run_index]
                tot_run_index = tot_run_index + 1
            elif cumulative_time_run[curr_run_index + 1] < tot_run_index * self.dt:
                curr_run_index = curr_run_index + 1
            else:
                tot_run_index = tot_run_index + 1
        if tot_run_index < self.cumul_length:
            self.final_average_times[tot_run_index] = self.final_average_times[tot_run_index] + occ_run[curr_run_index]


    def random_walk(self, conc_step, recursion_nr):
        while self.cumulative_walking_time < self.time_points[conc_step + 1]:

            self.list_of_state_ids = np.ones((2, self.n_sites * 2))
            self.get_local_exit_array()
            self.get_transition_rate_array(self.molar_concentrations[conc_step])
#             ipdb.set_trace() # oslo
            exit_rate = np.sum(self.transition_rate_array)
            
            if self.step >= len(self.walk_record):
                break
            else:
                pass
            random_exit_probability = random.uniform(0, 1)
#             ipdb.set_trace()
            if exit_rate == 0:
                self.id_record.append(self.state_id)
                holding_time = self.time_points[conc_step + 1] - self.cumulative_walking_time
                self.walk_record[self.step][1] = holding_time
                self.cumulative_walking_time = self.time_points[conc_step + 1]
                self.walk_record[self.step][2] = self.cumulative_walking_time
                self.step += 1
                break
            else:
                pass
            holding_time = np.log(1 / (random_exit_probability)) / exit_rate
            self.id_record.append(self.state_id)
            self.walk_record[self.step][1] = holding_time
            self.cumulative_walking_time += holding_time
            if self.cumulative_walking_time > self.time_points[conc_step + 1]:
                self.walk_record[self.step][2] = self.time_points[conc_step + 1]
                self.walk_record[self.step][1] = self.time_points[conc_step + 1] - self.walk_record[self.step - 1][2]
                self.cumulative_walking_time = self.time_points[conc_step + 1]
                self.step += 1
                break
            else:
                self.walk_record[self.step][2] = self.cumulative_walking_time
            self.walk_record[self.step][3] = self.get_state_occupancy(self.state_id)
            # ipdb.set_trace()
            self.update_state_from_id_choice(
                choice(list(range(2, len(self.list_of_state_ids))), 1, p=self.transition_rate_array / exit_rate)[0])
            self.update_state_id()
            self.step += 1
            # print(self.step)
            ## PLOTTING STATES
            # if recursion_nr == 1 or recursion_nr == 10 or recursion_nr == 50:
            # self.generate_state_figure(recursion_nr)
            # if np.array_equal((self.state_id), np.array(
            #         [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 11.0, 7.0, 3.0,
            #          2.0, 15.0, 12.0, 9.0, 1.0, 14.0, 6.0, 13.0, 0.0, 5.0, 10.0, 8.0, 4.0])):
            #     ipdb.set_trace()  # aintnothingbutaheartbreak


    def save_states_and_times(self, recursions):
        if not os.path.exists(str(
                os.getcwd()) + '/' + 'data_logs'):  # test to se if a folder for storing the data exists
            os.makedirs(str(os.getcwd()) + '/' + 'data_logs')  # if no , create the folder
        if not os.path.exists(str(
                os.getcwd()) + '/' + 'data_logs/' +
                              self.name):
            os.makedirs(str(os.getcwd()) + '/' +
                        'data_logs/' + self.name)  # if no , create the folder
        f = open((str(os.getcwd()) + '/' +
                  'data_logs/' + self.name + '/' + self.name + '_' + str(recursions) + '.txt'), 'w')
        for i in range(0, len(self.id_record)):
            f.write(str(self.walk_record[i][1]))
            for j in range(0, len(self.id_record[i])):
                f.write("\t" + str(self.id_record[i][j]))
            f.write("\n")
        f.close()

        f = open((str(os.getcwd()) + '/' +
                  'data_logs/' + self.name + '/' + self.name + '.txt'), self.total_writing_option)
        for i in range(0, len(self.id_record)):
            f.write(str(recursions) + "\t" + str(self.walk_record[i][1]))
            for j in range(0, len(self.id_record[i])):
                f.write("\t" + str(self.id_record[i][j]))
            f.write("\n")


    # %%%%%%%%%%%%%%%%%%%% MAIN STATE EXPLORING DECISION TREE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    def get_local_exit_array(self):
        self.update_state_id()
        self.exit_array = np.zeros((0))
        self.distance_exit_array = np.zeros((0))
        self.occupancy_key = np.zeros((0))
        self.update_states_local()
        self.revert_state_id = list(self.state_id)
        for current_site in range(0, len(self.coordinates)):
            current_site_state = self.site_status_vector[current_site]
            if current_site_state == 1:

                self.check_mono_to_bi(current_site)

                self.check_mono_to_empty(current_site)
            elif current_site_state == 2:
                self.check_bi_to_mono(current_site)
            else:
                self.check_empty_to_mono(current_site)
        # done
        


    # %%%%%%%%%%%%%%%%%%%% MAIN STATE EXPLORING DECISION TREE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    def update_state_id(self):
        self.state_id = np.concatenate([self.site_status_vector, self.pointer_vector])
        # print(self.state_id)
        # done


    def update_states_local(self):
        if any((self.state_id == x).all() for x in self.list_of_state_ids):  # xx
            # print("I got here")
            pass
        else:
            # self.generate_state_figure()
            self.state_discoveries += 1
            self.list_of_state_ids = np.append(self.list_of_state_ids, [self.state_id], axis=0)

            self.exit_array = np.append(self.exit_array, [0])
            self.distance_exit_array = np.append(self.distance_exit_array, [0])
            self.occupancy_key = np.append(self.occupancy_key, [0])
            self.update_occupancy_key()
            # ipdb.set_trace() #tellmewhy
        # done


    def generate_average_occupancy_plot(self, recursion_nr, total_time):
        plt.figure(figsize=(10, 10))

        if not os.path.exists(str(
                os.getcwd()) + '/' + 'average_occupancy'):  # test to see if a folder for storing the data exists
            os.makedirs(str(os.getcwd()) + '/' + 'average_occupancy')  # if no , create the folder
        if not os.path.exists(str(
                os.getcwd()) + '/' + 'average_occupancy/' +
                              self.name):
            os.makedirs(str(os.getcwd()) + '/' +
                        'average_occupancy/' + self.name)  # if no , create the folder
        plt.title('Average occupancy as a function of time')
        plt.plot([i for i in range(0, total_time, self.dt)], self.final_average_times / recursion_nr)
        plt.ylim(0, self.n_sites)
        plt.xlim(0, total_time)
        plt.savefig(str(os.getcwd()) + '/' + 'average_occupancy/' +
                    self.name + '/' + "avg_occ" + str(recursion_nr) + ".png")
        plt.savefig(str(os.getcwd()) + '/' + 'average_occupancy/' +
                    self.name + '/' + "avg_occ" + str(recursion_nr) + ".svg")
        plt.close()
        # done


    def generate_state_figure(self, recursion_nr):
        plt.figure(figsize=(5, 5))
        self.my_plot_format(20, 20, 30)

        maxx = np.max(self.coordinates[:, 1])
        minx = np.min(self.coordinates[:, 1])
        maxy = np.max(self.coordinates[:, 0])
        miny = np.min(self.coordinates[:, 0])
        centerx = (maxx + minx) / 2
        centery = (maxy + miny) / 2
        plt.scatter(self.coordinates[:, 1], self.coordinates[:, 0], marker='o', s=2000, facecolors='none',
                    edgecolors='#11B5E5', linewidth=10)
        plt.ylim(centery - (maxy - miny) / 2 - 10, centery + (maxy - miny) / 2 + 10)
        plt.xlim(centerx - (maxx - miny) / 2 - 10, centerx + (maxx - minx) / 2 + 10)
        plt.axis('off')
        for site in range(0, len(self.site_status_vector)):
            if int(self.site_status_vector[site] == 1):
                plt.scatter(self.coordinates[site, 1], self.coordinates[site, 0], c='#FF00A6', s=2000, alpha=1,
                            edgecolors='#FF00A6', linewidth='12')
            elif int(self.site_status_vector[site] == 2):
                plt.scatter(self.coordinates[site, 1], self.coordinates[site, 0], c='#FF00A6', s=2000, alpha=1,
                            edgecolors='#FF00A6', linewidth='12')
                plt.plot([(self.coordinates[site, 1]), self.coordinates[int(self.pointer_vector[site]), 1]],
                         [(self.coordinates[site, 0]), self.coordinates[int(self.pointer_vector[site]), 0]],
                         '#FF00A6', linewidth=30, alpha=1)
            else:
                pass
        if not os.path.exists(str(
                os.getcwd()) + '/' + 'state_configurations'):
            os.makedirs(str(os.getcwd()) + '/' + 'state_configurations')
        if not os.path.exists(str(
                os.getcwd()) + '/' + 'state_configurations/' +
                              self.name + str(recursion_nr)):
            os.makedirs(str(os.getcwd()) + '/' +
                        'state_configurations/' + self.name + str(recursion_nr))

        plt.savefig(str(os.getcwd()) + '/' + 'state_configurations/' + self.name + str(recursion_nr) + '/' +
                    self.name + str(self.graph_step) + ".png")
        plt.savefig(str(os.getcwd()) + '/' + 'state_configurations/' + self.name + str(recursion_nr) + '/' +
                    self.name + str(self.graph_step) + ".svg")
        self.graph_step += 1
        plt.close('all')


    def my_plot_format(self, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rcParams.update(
            {'font.size': 15, 'figure.figsize': (6, 6), 'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
             'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'})


    def update_occupancy_key(self):
        count = 0
        for i in range(0, len(self.site_status_vector)):
            if self.site_status_vector[i] == 1:
                count += 1
            elif self.site_status_vector[i] == 2:
                count += 0.5
            else:
                pass
        self.occupancy_key[len(self.occupancy_key) - 1] = count
        # done


    def get_state_occupancy(self, state_id):
        site_status_vector = self.get_state_from_id(state_id)
        occupancy = 0
        for occ in range(0, len(site_status_vector)):
            if site_status_vector[occ] == 2:
                occupancy += 0.5
            else:
                occupancy += site_status_vector[occ]
        if occupancy * 2 % 2 == 1:
            ipdb.set_trace()
        return occupancy


    def check_mono_to_bi(self, current_site):
        # ipdb.set_trace()  # AOMAME
        try:
            neighbor_options = self.neighbor_directory[current_site][1:]
        except:
            neighbor_options = np.array([])
        bivalent_conversion = 0
        for i in range(0, len(neighbor_options)):
            obstruction = False
            for other_neighbor in range(0, len(neighbor_options)):
                if self.site_status_vector[neighbor_options[other_neighbor]] == 2 and obstruction is False:
                    obstruction = intersect(self.coordinates[current_site],
                                            self.coordinates[neighbor_options[i]],
                                            self.coordinates[neighbor_options[other_neighbor]],
                                            self.coordinates[int(self.pointer_vector[neighbor_options[other_neighbor]])], self.box_size)
                else:
                    pass
            if self.site_status_vector[neighbor_options[i]] == 0 and \
                    self.site_status_vector[current_site] == 1 and \
                    obstruction == False:
                # ipdb.set_trace()
                self.site_status_vector[neighbor_options[i]] = 2
                self.site_status_vector[current_site] = 2
                self.pointer_vector[neighbor_options[i]] = current_site
                self.pointer_vector[current_site] = neighbor_options[i]
                i = len(neighbor_options)
                bivalent_conversion = 1
                self.update_state_id()
                self.update_states_local()
                from_state_index = self.list_of_state_ids.tolist().index(self.revert_state_id)
                to_state_index = self.list_of_state_ids.tolist().index(self.state_id.tolist())

                self.distance_exit_array[to_state_index - 2] = \
                    self.distance_matrix[current_site, int(self.pointer_vector[current_site])]
                self.exit_array[to_state_index - 2] = self.k2
                self.update_state_from_id(self.revert_state_id)
                self.update_state_id()
        if bivalent_conversion == 0:
            self.site_status_vector[current_site] = 0
            self.update_state_id()
            self.update_states_local()
            to_state_index = self.list_of_state_ids.tolist().index(self.state_id.tolist())
            self.exit_array[-1] = self.k_1
            self.update_state_from_id(self.revert_state_id)
            self.update_state_id()
        else:
            pass


    def check_mono_to_empty(self, current_site):
        self.site_status_vector[current_site] = 0
        self.update_state_id()
        self.update_states_local()
        to_state_index = self.list_of_state_ids.tolist().index(self.state_id.tolist())
        self.exit_array[-1] = self.k_1
        self.update_state_from_id(self.revert_state_id)
        self.update_state_id()


    def check_bi_to_mono(self, current_site):
        for decision in [0, 1]:
            self.site_status_vector[current_site] = decision
            self.site_status_vector[int(self.pointer_vector[current_site])] = 1 - decision
            pointer_place_holder = int(self.pointer_vector[current_site])
            self.pointer_vector[pointer_place_holder] = self.n_sites + 1
            self.pointer_vector[current_site] = self.n_sites + 1
            self.update_state_id()
            self.update_states_local()
            to_state_index = self.list_of_state_ids.tolist().index(self.state_id.tolist())
            self.distance_exit_array[to_state_index - 2] = self.distance_matrix[int(current_site), pointer_place_holder]
            self.exit_array[-1] = self.k_2
            self.update_state_from_id(self.revert_state_id)
            self.update_state_id()


    def check_empty_to_mono(self, current_site):
        self.site_status_vector[int(current_site)] = 1
        self.update_state_id()
        self.update_states_local()
        to_state_index = self.list_of_state_ids.tolist().index(self.state_id.tolist())
        self.exit_array[-1] = self.k1
        self.update_state_from_id(self.revert_state_id)
        self.update_state_id()


    def update_state_from_id(self, input_state_ID):
        self.site_status_vector = input_state_ID[0:self.n_sites]
        self.pointer_vector = input_state_ID[(self.n_sites)::]


    def update_state_from_id_choice(self, input_state_ID_index):
        try:
            input_state_ID = self.list_of_state_ids[int(input_state_ID_index)]
        except:
            ipdb.set_trace()
        self.site_status_vector = input_state_ID[0:self.n_sites]
        self.pointer_vector = input_state_ID[(self.n_sites)::]


    def get_state_from_id(self, input_state_ID):
        site_status_vector = input_state_ID[0:self.n_sites]
        return site_status_vector


    def get_transition_rate_array(self, concentration):
        exit_length = len(self.exit_array)
        self.transition_rate_array = np.zeros((exit_length))
        for i in range(1, exit_length):
            if self.exit_array[i] == 0:
                pass
            elif self.exit_array[i] == self.k1:
                self.transition_rate_array[i] = self.k_on * concentration
            elif self.exit_array[i] == self.k_1:
                self.transition_rate_array[i] = self.k_off
            elif self.exit_array[i] == self.k2:
#                 self.transition_rate_array[i] = self.k_mono_bi * alpha(self.distance_exit_array[i])
                # print 'distance dependence modified rate: ', self.transition_rate_array[i]
                # print 'rate_mono_to_double: ', self.k_mono_bi
                self.transition_rate_array[i] = self.k_bi_mono / spatial_tolerance(self.distance_exit_array[i])
                
            elif self.exit_array[i] == self.k_2:
                self.transition_rate_array[i] = self.k_bi_mono
            else:
                pass
#             ipdb.set_trace()

def spatial_tolerance(x):
    l_t,alpha_t,l_c,alpha_c,KD2max=[25.4726412, 0.27, 1.1, 0.33, 0.99 ]
    compression = KD2max*np.exp(-1*alpha_c*x-l_c)
    tension = KD2max/(1+np.exp(-1*alpha_t*(x-l_t)))
    KD2 = (compression+tension)
    
    return KD2



#####        #####        #####        #####        #####        #####        #####        
## POST SIMULATION DATA ANALYSIS
#####        #####        #####        #####        #####        #####        #####        






def dwell_times_through_all_runs(name):
    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '.txt'), delimiter="\t")

    repeats = full_data[:, 0]
    ttimes = full_data[:, 1]
    c_times = np.cumsum(ttimes)
    states_arr = full_data[:, 2::]
    nsites = len(states_arr[0]) / 2

    del full_data
    track_changes = np.zeros((nsites, 2),
                             dtype=int)  # nr of the current antigen in the spot and how many timesteps it has stayed already, and pointer if bivalent binding
    for step in range(nsites):
        track_changes[step, 1] = nsites + 1

    dwell_log = np.zeros((1, 2))
    previous_state = np.zeros(nsites)
    ab_count = 1

    for step in range(1, len(ttimes)):
        if repeats[step] == repeats[step - 1]:
            for i in range(0, nsites):

                if states_arr[step, i] == 1:
                    if previous_state[i] == 0:
                        track_changes[i, 0] = ab_count
                        dwell_log = np.append(dwell_log, [[c_times[step - 1], 0]], axis=0)
                        # update total number of antibodies discovered
                        ab_count += 1
                    else:
                        if track_changes[i, 1] != nsites + 1:
                            track_changes[i, 0] = track_changes[track_changes[i, 1], 0]
                            track_changes[i, 1] = nsites + 1

                if states_arr[step, i] == 2:
                    if previous_state[i] == 0:
                        track_changes[i, 1] = states_arr[step, nsites + i]

                else:
                    if track_changes[i, 1] != nsites + 1:
                        track_changes[i, 1] = nsites + 1

                    elif previous_state[i] == 1:
                        dwell_log[track_changes[i, 0], 1] = c_times[step - 1]

        else:
            ids_of_rows_to_delete = []
            count_delete = 0
            for i in range(0, nsites):
                if dwell_log[track_changes[i, 0], 1] == 0:
                    track_changes[i, 1] = nsites + 1
                    ids_of_rows_to_delete.append(track_changes[i, 0])
                    count_delete += 1
            dwell_log = np.delete(dwell_log, ids_of_rows_to_delete, 0)
            ab_count -= count_delete

        previous_state = states_arr[step, 0:nsites]

    # ipdb.set_trace()  # tengo
    ids_of_rows_to_delete = []
    count_delete = 0
    for i in range(0, nsites):
        if dwell_log[track_changes[i, 0], 1] == 0:
            track_changes[i, 1] = nsites + 1
            ids_of_rows_to_delete.append(track_changes[i, 0])
    dwell_log = np.delete(dwell_log, ids_of_rows_to_delete, 0)

    dwell_times = np.diff(dwell_log, axis=1)

    # plot

    plt.hist(dwell_times, color='blue', edgecolor='black',
             bins=(1000))

    plt.show()


def analyze_separate_ab_walking(name, repetition):
    ttimes = []
    state_config = []

    # with open((str(os.getcwd()) + '/' +
    #            'data_logs/' + name + '/' + name + '.txt')) as csvfile:
    #     plots = csv.reader(csvfile, delimiter='\t')
    #     plots_list = list(plots)
    #     for row in plots:
    #         recurs.append(int(row[0]))
    #         ttimes.append(float(row[1]))
    #         state_config.append(str(row[2::]))

    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '_' + str(repetition) + '.txt'),
                           delimiter="\t")

    # recurs = full_data[:,0]
    ttimes = full_data[:, 0]
    c_times = np.cumsum(ttimes)
    states_arr = full_data[:, 1::]
    nsites = len(states_arr[0]) / 2
    starting_time = np.zeros(len(ttimes))

    track_changes = np.zeros((nsites, 3),
                             dtype=int)  # nr of the current antigen in the spot and how many timesteps it has stayed already, and pointer if bivalent binding
    for step in range(nsites):
        track_changes[step, 2] = nsites + 1

    ab_walking_log = np.zeros((len(ttimes), len(ttimes),
                               4))  # ab nr - step nr -  phase (0) & length (1) of step & cumulativetime (2), ag nr (3)
    previous_state = np.zeros(nsites)

    ab_count = 0

    for step in range(0, len(ttimes)):
        for i in range(0, nsites):
            if states_arr[step, i] == 1:
                if previous_state[i] == 0:
                    # update tracking matrix
                    track_changes[i, 0] = ab_count  # number of the latest antibody at site i
                    track_changes[i, 1] = 0  # pointer of number of timesteps given antibody has stayed at site i
                    starting_time[ab_count] = c_times[step - 1]  # when given antibody first bound to antigen
                    # update total number of antibodies discovered
                    ab_count += 1

                    # update ab walking log
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 0] = 1  # monovalent binding
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 1] = ttimes[step]
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 2] = ttimes[step]
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 2] = i + 1

                elif previous_state[i] == 1:

                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 1] += ttimes[step]
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 2] += ttimes[step]

                else:
                    if track_changes[i, 2] != nsites + 1:
                        track_changes[i, 0] = track_changes[track_changes[i, 2], 0]
                        track_changes[i, 1] = track_changes[track_changes[i, 2], 1]
                        track_changes[i, 2] = nsites + 1

                    track_changes[i, 1] += 1

                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 0] = 1
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 1] = ttimes[step]
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 2] = ab_walking_log[track_changes[i, 0],
                                                                                                 track_changes[
                                                                                                     i, 1] - 1, 2] + \
                                                                                  ttimes[step]
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 3] = i + 1

            if states_arr[step, i] == 2:
                if previous_state[i] == 2:
                    if track_changes[i, 2] == nsites + 1:
                        ab_walking_log[track_changes[i, 0], track_changes[i, 1], 1] += ttimes[step]
                        ab_walking_log[track_changes[i, 0], track_changes[i, 1], 2] += ttimes[step]

                elif previous_state[i] == 1:
                    track_changes[i, 1] += 1
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 0] = 2
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 1] = ttimes[step]
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 2] = ab_walking_log[track_changes[i, 0],
                                                                                                 track_changes[
                                                                                                     i, 1] - 1, 2] + \
                                                                                  ttimes[step]
                    ab_walking_log[track_changes[i, 0], track_changes[i, 1], 3] = (i + states_arr[
                        step, nsites + 1]) / 2  # this system doesn't guarantee distinguishable states (e.g. 4 and 5 and 3 and 6 will both return 4.5

                else:
                    track_changes[i, 2] = states_arr[step, nsites + i]

            else:
                if track_changes[i, 2] != nsites + 1:
                    track_changes[i, 2] = nsites + 1

        previous_state = states_arr[step, 0:nsites]

    # total dwell times
    dwell_times = np.zeros(ab_count)
    for i in range(0, ab_count):
        dwell_times[i] = np.sum(ab_walking_log[i, :, 1])

    plt.hist(dwell_times, color='blue', edgecolor='black',
             bins=(100))

    ab_walking_log[ab_walking_log == 0] = np.nan

    ## PLOTTING THE MONO/BIVALENT BINDING OF SEPARATE ABS IN ONE PLOT
    # for ab in range(0, ab_count):
    #     plt.step(np.append(starting_time[ab], ab_walking_log[ab, :, 2] + starting_time[ab]), np.append(1, ab_walking_log[ab, :, 0]))

    plt.show()


def analyze_ab_step_size(name, distance_matrix, repeat):
    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '_' + str(repeat) + '.txt'),
                           delimiter="\t")

    states_arr = full_data[:, 1::]
    nsites = len(states_arr[0]) / 2

    distance_matrix /= 2

    track_changes = np.zeros((nsites, 3),
                             dtype=int)  # nr of the current antigen in the spot and how many timesteps it has stayed already, and pointer if bivalent binding
    for step in range(nsites):
        track_changes[step, 2] = nsites + 1

    ab_dist_log = np.zeros(
        (1, len(states_arr)))  # ab nr - step nr -  phase (0) & length (1) of step & cumulativetime (2), ag nr (3)
    previous_state = np.zeros(nsites)

    ab_count = 1

    for step in range(1, len(states_arr)):
        for i in range(0, nsites):
            if states_arr[step, i] == 1:
                if previous_state[i] == 0:
                    # update tracking matrix
                    track_changes[i, 0] = ab_count  # number of the latest antibody at site i
                    track_changes[i, 1] = 0  # pointer of number of timesteps given antibody has stayed at site i
                    ab_dist_log = np.append(ab_dist_log, np.zeros((1, len(states_arr))), axis=0)

                    # update total number of antibodies discovered
                    ab_count += 1

                elif previous_state[i] == 2:
                    if track_changes[i, 2] != nsites + 1:
                        track_changes[i, 0] = track_changes[track_changes[i, 2], 0]
                        track_changes[i, 1] = track_changes[track_changes[i, 2], 1]
                        track_changes[i, 2] = nsites + 1
                    ab_dist_log[track_changes[i, 0], track_changes[i, 1]] = distance_matrix[
                        i, int(states_arr[step - 1, nsites + i])]

                    track_changes[i, 1] += 1

            if states_arr[step, i] == 2:
                if previous_state[i] == 1:
                    ab_dist_log[track_changes[i, 0], track_changes[i, 1]] = distance_matrix[
                        i, int(states_arr[step, nsites + i])]
                    track_changes[i, 1] += 1

                elif previous_state[i] == 0:
                    track_changes[i, 2] = states_arr[step, nsites + i]

            else:
                if track_changes[i, 2] != nsites + 1:
                    track_changes[i, 2] = nsites + 1

        previous_state = states_arr[step, 0:nsites]

    delete_abs = []
    for i in range(0, nsites):
        delete_abs.append(track_changes[i, 0])
        track_changes[i, 2] = nsites + 1
    ab_dist_log = np.delete(ab_dist_log, delete_abs, 0)
    ipdb.set_trace()

    ab_dist_log_concat = np.concatenate(ab_dist_log)
    ab_dist_log_concat[ab_dist_log_concat == 0] = np.nan
    ab_dist_log_concat = ab_dist_log_concat[~np.isnan(ab_dist_log_concat)]

    plt.hist(ab_dist_log_concat, color='blue', edgecolor='black', bins=(2))

    plt.show()
    
def analyze_ab_residence_times(name, repeats, ag_coordinates,box_size):
    
    new_rc_params = {'text.usetex': False,
    "svg.fonttype": 'none'
    }
    plt.rcParams.update(new_rc_params)

    plt.rc('xtick', labelsize=25) 
    plt.rc('ytick', labelsize=25) 
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20,5))

    antibody_residence_times = []
    
    for rep in repeats:
        full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '_' + str(rep) + '.txt'), delimiter="\t")
        states_arr = full_data[:, 1::]
        timestamps = full_data[:,0]
        nsites = len(states_arr[0]) / 2
        track_changes = np.zeros((nsites, 3), dtype=int)  # nr of the current antigen in the spot and how many timesteps it has stayed already, and pointer if bivalent binding
        for step in range(nsites):
            track_changes[step, 2] = nsites + 1

        ab_traj_log = np.zeros((1, len(states_arr)))
        ag_coord_traj_log_x = np.zeros((1, len(states_arr)))
        ag_coord_traj_log_y = np.zeros((1, len(states_arr)))
        previous_state = np.zeros(nsites)
        ab_log =[]
        ab_number = 0
        active_abs = np.empty(nsites)
        active_abs[:] = np.nan

        for step in range(1, len(states_arr)):
        #     # if step == 3: ipdb.set_trace()
            for i in range(0, nsites):
        #         print 'site: ' , i
                if states_arr[step, i] == 1 and previous_state[i] == 0:
                    # monovalent arrival event - new AB trajectory create
                    ab_log.append([])
                    active_abs[i] = ab_number
                    ab_log[ab_number].append([ag_coordinates[i][0]+(random.random()-0.5)*0, ag_coordinates[i][1]+(random.random()-0.5)*0, timestamps[step]])
                    ab_number += 1
                elif states_arr[step, i] == 1 and previous_state[i] == 2:
                    # STEP IS COMPLETED - ab number lookup required
                    # ipdb.set_trace()
                    active_abs[int(previous_state[i+nsites])] = np.nan
                    ab_log[int(active_abs[i])].append([ag_coordinates[i][0]+(random.random()-0.5)*0, ag_coordinates[i][1]+(random.random()-0.5)*0, timestamps[step]])
                elif states_arr[step, i] == 0 and previous_state[i] == 1:
                    # ANTIBODY FALLS OFF - still record coordinate, because it was here until the moment it falls off
                    # ab_log[int(active_abs[i])].append([ag_coordinates[i][0]+random.random()-0.5, ag_coordinates[i][1]+random.random()-0.5, timestamps[step]])
                    active_abs[i] = np.nan
                elif states_arr[step, i] == 2 and previous_state[i] == 1:
                    # begin a step - make a double record of the antibody ID - one for each site
                    # also need to record a centroid position for the coordinates
                    # ipdb.set_trace()
                    active_abs[int(states_arr[step, i + nsites])] = active_abs[i]
                    centroidx = np.average([ag_coordinates[i][0] , ag_coordinates[int(states_arr[step, i + nsites])][0]])
                    centroidy = np.average([ag_coordinates[i][1], ag_coordinates[int(states_arr[step, i + nsites])][1]])
                    if np.abs(centroidx-ag_coordinates[i][0]) >=box_size/3.0:
                        centroidx = centroidx - box_size/2.0
                    if np.abs(centroidy-ag_coordinates[i][1]) >=box_size/3.0:
                        centroidy = centroidy - box_size/2.0
                        # ipdb.set_trace()
                    ab_log[int(active_abs[i])].append([centroidx+(random.random()-0.5)*0, centroidy+(random.random()-0.5)*0, timestamps[step]])
            previous_state = states_arr[step, :]


        for antibody in range(0,len(ab_log)):#len(ab_log)):
            trans = np.transpose(ab_log[antibody])    
            

        
            antibody_residence_times.append(np.sum(trans[2]))
        

    
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.hist(antibody_residence_times,bins=100,color='k',log=False)
    

#     plt.xlim(-2.5,2.5)
#     plt.ylim(0,500)
    plt.xlabel("antibody residence time $[nm/s]$")
    plt.ylabel("cumulative time [s]")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  + 'residence_times.svg')
    plt.show()
    plt.close()

    
def get_cmap(n, name='RdBu'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
def analyze_ab_position(name, repeats, ag_coordinates,box_size):
    horizontalnetdisplacement = []
    horizontalnetdisp_weights = []
    horizontalvelocities = []
    horizontalweights = []    
    xmax = np.max(np.array([ag_coordinates]).T[1])
    xmin = np.min(np.array([ag_coordinates]).T[1])
    xrange = xmax-xmin
    ymax =np.max(np.array([ag_coordinates]).T[0])
    ymin = np.min(np.array([ag_coordinates]).T[0])
    yrange = ymax-ymin
#     horizontalbinno = len(ag_coordinates)
#     horizontalbinwidth = xrange/(horizontalbinno*0.5)
#     horizontalhistbins = np.arange(xmin-horizontalbinwidth/2,xmax+horizontalbinwidth,horizontalbinwidth)
#     horizontalhistcounts = np.zeros((len(horizontalhistbins)))
#     fig = plt.figure(figsize=(20,5))
    new_rc_params = {'text.usetex': False,
    "svg.fonttype": 'none'
    }
    plt.rcParams.update(new_rc_params)

    plt.rc('xtick', labelsize=25) 
    plt.rc('ytick', labelsize=25) 
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20,5))

    antibodylocs = {}
    
    for rep in repeats:
        full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '_' + str(rep) + '.txt'), delimiter="\t")
        states_arr = full_data[:, 1::]
        timestamps = full_data[:,0]
        nsites = len(states_arr[0]) / 2
        track_changes = np.zeros((nsites, 3), dtype=int)  # nr of the current antigen in the spot and how many timesteps it has stayed already, and pointer if bivalent binding
        for step in range(nsites):
            track_changes[step, 2] = nsites + 1

        ab_traj_log = np.zeros((1, len(states_arr)))
        ag_coord_traj_log_x = np.zeros((1, len(states_arr)))
        ag_coord_traj_log_y = np.zeros((1, len(states_arr)))
        previous_state = np.zeros(nsites)
        ab_log =[]
        ab_number = 0
        active_abs = np.empty(nsites)
        active_abs[:] = np.nan

        for step in range(1, len(states_arr)):
        #     # if step == 3: ipdb.set_trace()
            for i in range(0, nsites):
        #         print 'site: ' , i
                if states_arr[step, i] == 1 and previous_state[i] == 0:
                    # monovalent arrival event - new AB trajectory create
                    ab_log.append([])
                    active_abs[i] = ab_number
                    ab_log[ab_number].append([ag_coordinates[i][0]+(random.random()-0.5)*0, ag_coordinates[i][1]+(random.random()-0.5)*0, timestamps[step]])
                    ab_number += 1
                elif states_arr[step, i] == 1 and previous_state[i] == 2:
                    # STEP IS COMPLETED - ab number lookup required
                    # ipdb.set_trace()
                    active_abs[int(previous_state[i+nsites])] = np.nan
                    ab_log[int(active_abs[i])].append([ag_coordinates[i][0]+(random.random()-0.5)*0, ag_coordinates[i][1]+(random.random()-0.5)*0, timestamps[step]])
                elif states_arr[step, i] == 0 and previous_state[i] == 1:
                    # ANTIBODY FALLS OFF - still record coordinate, because it was here until the moment it falls off
                    # ab_log[int(active_abs[i])].append([ag_coordinates[i][0]+random.random()-0.5, ag_coordinates[i][1]+random.random()-0.5, timestamps[step]])
                    active_abs[i] = np.nan
                elif states_arr[step, i] == 2 and previous_state[i] == 1:
                    # begin a step - make a double record of the antibody ID - one for each site
                    # also need to record a centroid position for the coordinates
                    # ipdb.set_trace()
                    active_abs[int(states_arr[step, i + nsites])] = active_abs[i]
                    centroidx = np.average([ag_coordinates[i][0] , ag_coordinates[int(states_arr[step, i + nsites])][0]])
                    centroidy = np.average([ag_coordinates[i][1], ag_coordinates[int(states_arr[step, i + nsites])][1]])
                    if np.abs(centroidx-ag_coordinates[i][0]) >=box_size/3.0:
                        centroidx = centroidx - box_size/2.0
                    if np.abs(centroidy-ag_coordinates[i][1]) >=box_size/3.0:
                        centroidy = centroidy - box_size/2.0
                        # ipdb.set_trace()
                    ab_log[int(active_abs[i])].append([centroidx+(random.random()-0.5)*0, centroidy+(random.random()-0.5)*0, timestamps[step]])
            previous_state = states_arr[step, :]


        velocityinterval = int(len(ag_coordinates)*10)

        for antibody in range(0,len(ab_log)):#len(ab_log)):
            trans = np.transpose(ab_log[antibody])    
            
            horizontalnetdisplacement.append(np.average(trans[1][int(0*len(trans[1])):])- np.average(trans[1][0:1]) )
            horizontalnetdisp_weights.append(len(trans[1]))
            try:
                for step in range(0,len(ag_coordinates)*10)[::velocityinterval]: #limit chosen to capture traversal distance
                    residence_time = np.sum([trans[2][x] for x in range(step,step+velocityinterval)])
                    velocity = (trans[1][step+velocityinterval] - trans[1][step])/residence_time
                    if velocity != 0:
                        horizontalvelocities.append(velocity)
                        horizontalweights.append(residence_time)
            except: pass
        cmap = get_cmap(len(ab_log))
        for antibody in range(0,len(ab_log)):#len(ab_log)):
            trans = np.transpose(ab_log[antibody])
            cumulative_times = np.cumsum(trans[2])-np.min(np.cumsum(trans[2]))
            if trans[1][0] >= 0.0*xrange and trans[1][0] <= 0.99*xrange:
                plt.plot(cumulative_times, trans[1]-trans[1][0],alpha=0.4,c='#80002a')#cmap(np.random.randint(len(ab_log))))
            for step in range(0,len(trans[2])):
                residence_time = trans[2][step]
                xpos = trans[1][step]
                try: 
                    antibodylocs[np.round(xpos, 1)] += 1
                except:
                    antibodylocs[np.round(xpos, 1)] = 1

    plt.ylabel("antibody location [nm]")
    plt.xlabel("time [s]")
    plt.ylim(-1*box_size,box_size)
    plt.xlim(0,250)
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'   + 'random_walk_chart.svg')
    plt.show()
    plt.close()
#     ipdb.set_trace()

    locationbins = [key for key, value in antibodylocs.iteritems()]
    locationcounts = np.array([value for key, value in antibodylocs.iteritems()],dtype=np.float)
    locationcounts = locationcounts/(np.float(len(repeats)*len(ab_log)))
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.bar(locationbins,locationcounts,color='k')
    plt.ylabel("cumulative residence time $[s]$")
    plt.xlabel("antibody location [nm]")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  +  'antibodywalker_positiondistribution.svg')
    plt.show()
    plt.close()     
#     ipdb.set_trace()
    
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.hist(horizontalvelocities,bins=100,weights=np.array(horizontalweights,dtype=np.float)/(np.float(len(repeats)*len(ab_log))),color='k',log=False)
    averagevelocity = np.sum(horizontalweights/np.sum(horizontalweights)*horizontalvelocities)
    print("average velocity: ", averagevelocity)
    plt.xlim(-2.5,2.5)
#     plt.ylim(0,500)
    plt.xlabel("antibody velocity $[nm/s]$")
    plt.ylabel("cumulative time [s]")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  + 'velocity_distribution.svg')
    plt.show()
    plt.close()
    
    ### net movement 
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.hist(horizontalnetdisplacement,bins=len(ag_coordinates)*10,weights=np.array(horizontalnetdisp_weights,dtype=np.float)/(np.sum(horizontalnetdisp_weights)),color='k',log=False)
    plt.xlim(-200,200)
#     plt.ylim(0,500)
    plt.xlabel("antibody net displacement $[nm]$")
    plt.ylabel("frequency")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  + 'netdisplacment.svg')
    plt.show()
    plt.close()
    averagenetdisplacement = np.sum(np.array(horizontalnetdisp_weights,dtype=np.float)/np.sum(horizontalnetdisp_weights)*horizontalnetdisplacement)
    print("average net displacement: ", averagenetdisplacement)
#     horizontalvelocities = []
#     for antibody in range(0,1):#len(ab_log)):
#         trans = np.transpose(ab_log[antibody])    
#         for step in range(int(len(trans[2])*0.5),len(trans[2])-10):
#             residence_time = np.sum([trans[2][x] for x in range(step,step+velocityinterval)])
#             velocity = (trans[1][step+velocityinterval] - trans[1][step])/residence_time
#             horizontalvelocities.append(velocity)
# #     ipdb.set_trace()
#     plt.hist(horizontalvelocities,bins=100)
#     plt.show()
#     plt.close()    
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.hist(np.abs(horizontalvelocities),bins=100,weights=np.array(horizontalweights,dtype=np.float)/np.sum(horizontalweights),color='k',log=False)
    averagevelocity = np.sum(horizontalweights/np.sum(horizontalweights)*horizontalvelocities)
    plt.xlim(0,2.5)
#     plt.ylim(0,500)
    plt.xlabel("antibody speed $[nm/s]$")
    plt.ylabel("frequency")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  + 'absvelocity_distribution.svg')
    plt.show()
    plt.close()
#     ipdb.set_trace()

def analyze_ab_position2d(name, repeats, ag_coordinates,box_size):
    horizontalnetdisplacement = []
    horizontalnetdisp_weights = []
    horizontalvelocities = []
    horizontalweights = []    
    xmax = np.max(np.array([ag_coordinates]).T[1])
    xmin = np.min(np.array([ag_coordinates]).T[1])
    xrange = xmax-xmin
    ymax =np.max(np.array([ag_coordinates]).T[0])
    ymin = np.min(np.array([ag_coordinates]).T[0])
    yrange = ymax-ymin
#     horizontalbinno = len(ag_coordinates)
#     horizontalbinwidth = xrange/(horizontalbinno*0.5)
#     horizontalhistbins = np.arange(xmin-horizontalbinwidth/2,xmax+horizontalbinwidth,horizontalbinwidth)
#     horizontalhistcounts = np.zeros((len(horizontalhistbins)))
#     fig = plt.figure(figsize=(20,5))
    new_rc_params = {'text.usetex': False,
    "svg.fonttype": 'none'
    }
    plt.rcParams.update(new_rc_params)

    plt.rc('xtick', labelsize=25) 
    plt.rc('ytick', labelsize=25) 
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20,5))

    antibodylocs = {}
    
    for rep in repeats:
        full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '_' + str(rep) + '.txt'), delimiter="\t")
        states_arr = full_data[:, 1::]
        timestamps = full_data[:,0]
        nsites = len(states_arr[0]) / 2
        track_changes = np.zeros((nsites, 3), dtype=int)  # nr of the current antigen in the spot and how many timesteps it has stayed already, and pointer if bivalent binding
        for step in range(nsites):
            track_changes[step, 2] = nsites + 1

        ab_traj_log = np.zeros((1, len(states_arr)))
        ag_coord_traj_log_x = np.zeros((1, len(states_arr)))
        ag_coord_traj_log_y = np.zeros((1, len(states_arr)))
        previous_state = np.zeros(nsites)
        ab_log =[]
        ab_number = 0
        active_abs = np.empty(nsites)
        active_abs[:] = np.nan

        for step in range(1, len(states_arr)):
        #     # if step == 3: ipdb.set_trace()
            for i in range(0, nsites):
        #         print 'site: ' , i
                if states_arr[step, i] == 1 and previous_state[i] == 0:
                    # monovalent arrival event - new AB trajectory create
                    ab_log.append([])
                    active_abs[i] = ab_number
                    ab_log[ab_number].append([ag_coordinates[i][0]+(random.random()-0.5)*0, ag_coordinates[i][1]+(random.random()-0.5)*0, timestamps[step]])
                    ab_number += 1
                elif states_arr[step, i] == 1 and previous_state[i] == 2:
                    # STEP IS COMPLETED - ab number lookup required
                    # ipdb.set_trace()
                    active_abs[int(previous_state[i+nsites])] = np.nan
                    ab_log[int(active_abs[i])].append([ag_coordinates[i][0]+(random.random()-0.5)*0, ag_coordinates[i][1]+(random.random()-0.5)*0, timestamps[step]])
                elif states_arr[step, i] == 0 and previous_state[i] == 1:
                    # ANTIBODY FALLS OFF - still record coordinate, because it was here until the moment it falls off
                    # ab_log[int(active_abs[i])].append([ag_coordinates[i][0]+random.random()-0.5, ag_coordinates[i][1]+random.random()-0.5, timestamps[step]])
                    active_abs[i] = np.nan
                elif states_arr[step, i] == 2 and previous_state[i] == 1:
                    # begin a step - make a double record of the antibody ID - one for each site
                    # also need to record a centroid position for the coordinates
                    # ipdb.set_trace()
                    active_abs[int(states_arr[step, i + nsites])] = active_abs[i]
                    centroidx = np.average([ag_coordinates[i][0] , ag_coordinates[int(states_arr[step, i + nsites])][0]])
                    centroidy = np.average([ag_coordinates[i][1], ag_coordinates[int(states_arr[step, i + nsites])][1]])
                    if np.abs(centroidx-ag_coordinates[i][0]) >=box_size/3.0:
                        centroidx = centroidx - box_size/2.0
                    if np.abs(centroidy-ag_coordinates[i][1]) >=box_size/3.0:
                        centroidy = centroidy - box_size/2.0
                        # ipdb.set_trace()
                    ab_log[int(active_abs[i])].append([centroidx+(random.random()-0.5)*0, centroidy+(random.random()-0.5)*0, timestamps[step]])
            previous_state = states_arr[step, :]


        velocityinterval = int(len(ag_coordinates)*10)

        for antibody in range(0,len(ab_log)):#len(ab_log)):
            trans = np.transpose(ab_log[antibody])    
            
            horizontalnetdisplacement.append(np.average(trans[1][int(0*len(trans[1])):])- np.average(trans[1][0:1]) )
            horizontalnetdisp_weights.append(len(trans[1]))
            try:
                for step in range(0,len(ag_coordinates)*10)[::velocityinterval]: #limit chosen to capture traversal distance
                    residence_time = np.sum([trans[2][x] for x in range(step,step+velocityinterval)])
                    velocity = (trans[1][step+velocityinterval] - trans[1][step])/residence_time
                    if velocity != 0:
                        horizontalvelocities.append(velocity)
                        horizontalweights.append(residence_time)
            except: pass
        cmap = get_cmap(len(ab_log))
        for antibody in range(0,len(ab_log)):#len(ab_log)):
            trans = np.transpose(ab_log[antibody])
            cumulative_times = np.cumsum(trans[2])-np.min(np.cumsum(trans[2]))
            if trans[1][0] >= 0.35*xrange and trans[1][0] <= 0.65*xrange:
                plt.plot(cumulative_times, trans[1]-trans[1][0],alpha=0.4,c='#80002a')#cmap(np.random.randint(len(ab_log))))
            for step in range(0,len(trans[2])):
                residence_time = trans[2][step]
                xpos = trans[1][step]
                try: 
                    antibodylocs[np.round(xpos, 1)] += 1
                except:
                    antibodylocs[np.round(xpos, 1)] = 1

    plt.ylabel("antibody location [nm]")
    plt.xlabel("time [s]")
    plt.ylim(-1*box_size,box_size)
    plt.xlim(0,500)
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'   + 'random_walk_chart.svg')
    plt.show()
    plt.close()
#     ipdb.set_trace()

    locationbins = [key for key, value in antibodylocs.iteritems()]
    locationcounts = np.array([value for key, value in antibodylocs.iteritems()],dtype=np.float)
    locationcounts = locationcounts/(np.float(len(repeats)*len(ab_log)))
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.bar(locationbins,locationcounts,color='k')
    plt.ylabel("cumulative residence time $[s]$")
    plt.xlabel("antibody location [nm]")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  +  'antibodywalker_positiondistribution.svg')
    plt.show()
    plt.close()     
#     ipdb.set_trace()
    
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.hist(horizontalvelocities,bins=100,weights=np.array(horizontalweights,dtype=np.float)/(np.float(len(repeats)*len(ab_log))),color='k',log=False)
    averagevelocity = np.sum(horizontalweights/np.sum(horizontalweights)*horizontalvelocities)
    print("average velocity: ", averagevelocity)
    plt.xlim(-2.5,2.5)
#     plt.ylim(0,500)
    plt.xlabel("antibody velocity $[nm/s]$")
    plt.ylabel("cumulative time [s]")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  + 'velocity_distribution.svg')
    plt.show()
    plt.close()
    
    ### net movement 
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.hist(horizontalnetdisplacement,bins=len(ag_coordinates)*10,weights=np.array(horizontalnetdisp_weights,dtype=np.float)/(np.sum(horizontalnetdisp_weights)),color='k',log=False)
    plt.xlim(-200,200)
#     plt.ylim(0,500)
    plt.xlabel("antibody net displacement $[nm]$")
    plt.ylabel("frequency")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  + 'netdisplacment.svg')
    plt.show()
    plt.close()
    averagenetdisplacement = np.sum(np.array(horizontalnetdisp_weights,dtype=np.float)/np.sum(horizontalnetdisp_weights)*horizontalnetdisplacement)
    print("average net displacement: ", averagenetdisplacement)
#     horizontalvelocities = []
#     for antibody in range(0,1):#len(ab_log)):
#         trans = np.transpose(ab_log[antibody])    
#         for step in range(int(len(trans[2])*0.5),len(trans[2])-10):
#             residence_time = np.sum([trans[2][x] for x in range(step,step+velocityinterval)])
#             velocity = (trans[1][step+velocityinterval] - trans[1][step])/residence_time
#             horizontalvelocities.append(velocity)
# #     ipdb.set_trace()
#     plt.hist(horizontalvelocities,bins=100)
#     plt.show()
#     plt.close()    
    plt.figure(figsize=(10,5))
    plt.tight_layout()
    plt.hist(np.abs(horizontalvelocities),bins=100,weights=np.array(horizontalweights,dtype=np.float)/np.sum(horizontalweights),color='k',log=False)
    averagevelocity = np.sum(horizontalweights/np.sum(horizontalweights)*horizontalvelocities)
    plt.xlim(0,2.5)
#     plt.ylim(0,500)
    plt.xlabel("antibody speed $[nm/s]$")
    plt.ylabel("frequency")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/' + name + '/'  + 'absvelocity_distribution.svg')
    plt.show()
    plt.close()
#     ipdb.set_trace()

def analyze_ab_trajectory(name, repeat):
    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '_' + str(repeat) + '.txt'),
                           delimiter="\t")

    states_arr = full_data[:, 1::]
    nsites = len(states_arr[0]) / 2

    track_changes = np.zeros((nsites, 3),
                             dtype=int)  # nr of the current antigen in the spot and how many timesteps it has stayed already, and pointer if bivalent binding
    for step in range(nsites):
        track_changes[step, 2] = nsites + 1

    ab_traj_log = np.zeros((1, len(states_arr)))
    previous_state = np.zeros(nsites)

    ab_count = 1

    for step in range(1, len(states_arr)):
        for i in range(0, nsites):
            if states_arr[step, i] == 1:
                if previous_state[i] == 0:
                    # update tracking matrix
                    track_changes[i, 0] = ab_count  # number of the latest antibody at site i
                    track_changes[i, 1] = 0  # pointer of number of timesteps given antibody has stayed at site i
                    ab_traj_log = np.append(ab_traj_log, np.zeros((1, len(states_arr))), axis=0)
                    # update total number of antibodies discovered
                    ab_count += 1
                    ab_traj_log[track_changes[i, 0], track_changes[i, 1]] = i

                elif previous_state[i] == 2:
                    if track_changes[i, 2] != nsites + 1:
                        track_changes[i, 0] = track_changes[track_changes[i, 2], 0]
                        track_changes[i, 1] = track_changes[track_changes[i, 2], 1]

                        track_changes[i, 1] += 1
                        ab_traj_log[track_changes[i, 0], track_changes[i, 1]] = i
                    track_changes[i, 2] = nsites + 1

            if states_arr[step, i] == 2:
                if previous_state[i] == 0:
                    track_changes[i, 2] = states_arr[step, nsites + i]

            else:
                if track_changes[i, 2] != nsites + 1:
                    track_changes[i, 2] = nsites + 1

        previous_state = states_arr[step, 0:nsites]

    ipdb.set_trace()
    ab_traj_log[ab_traj_log == 0] = np.nan
    for i in range(1, len(ab_traj_log)):
        plt.plot(ab_traj_log[i])

    plt.show()


def analyze_possible_configurations(name):
    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '.txt'),
                           delimiter="\t")

    state_configs = full_data[:, 2::]
    config_count = []
    unique_configs = []

    for row in range(0, len(state_configs)):
        try:
            index = unique_configs.index(state_configs[row].tolist())
            config_count[index] += 1
        except:
            unique_configs.append(state_configs[row].tolist())
            config_count.append(1)
    ipdb.set_trace()  #
    unique_configs = np.array(unique_configs)
    config_count = np.array(config_count)

    inds = config_count.argsort()
    sorted_configs = unique_configs[inds]

    plt.plot(config_count)
    plt.show()




def stratified_spr_plots(name, total_time):
    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '.txt'),
                           delimiter="\t")

    dt = 2
    repeat_nr = full_data[:, 0]
    runs_count = int(max(repeat_nr))
    dt_count = int(total_time / dt)

    stay_times = full_data[:, 1]

    # 0 if the run hasn't changed, -1 if it has
    change_nr = np.ediff1d(repeat_nr)
    change_nr = np.append(0, change_nr)
    # ipdb.set_trace()
    zeros = np.where(change_nr != 0)[0]

    state_configs = full_data[:, 2::]
    len_signature = len(state_configs[0])
    log_of_all_states = np.split(state_configs, zeros)
    log_of_all_times = np.split(stay_times, zeros)

    # A matrix that will have states of all runs with const timestep
    all_runs = np.zeros((runs_count, dt_count, len_signature))

    # ipdb.set_trace() # caesar
    for x in range(runs_count):
        log_of_all_times[x] = np.cumsum(log_of_all_times[x])

    for nr in range(runs_count):
        uneven_step = 0
        even_step = 0
        max_step = len(log_of_all_times[nr])
        while uneven_step < max_step and even_step in range(dt_count):
            if log_of_all_times[nr][uneven_step] >= dt * even_step:
                all_runs[nr, even_step][:] = log_of_all_states[nr][uneven_step][:]
                even_step += 1
            elif log_of_all_times[nr][uneven_step] < dt * even_step:
                uneven_step += 1
        if even_step < dt_count - 1:
            while even_step < dt_count:
                all_runs[nr, even_step][:] = log_of_all_states[nr][-1][:]
                print("think abt why")

    # ipdb.set_trace()

    unique_config_log = []
    unique_config_frequency = []

    for even_step in range(dt_count):
        for nr in range(runs_count):
            try:
                index = unique_config_log.index(all_runs[nr, even_step].tolist())
                unique_config_frequency[index][even_step] += 1
                # print("I get here")
            except:
                unique_config_log.append(all_runs[nr, even_step].tolist())
                unique_config_frequency.append(([0] * dt_count))
                unique_config_frequency[-1][even_step] += 1

    # ipdb.set_trace() #samatoki
    unique_config_frequency = [[float(i) / runs_count for i in a] for a in unique_config_frequency]
    plt.stackplot(np.linspace(0, total_time, dt_count), unique_config_frequency)
    # ipdb.set_trace()
    plt.show()
    plt.close()
    for state in range(len(unique_config_log)):
        nsites = 0
        for site in range(0, len_signature / 2):
            if unique_config_log[state][site] == 1:
                nsites += 1.
            if unique_config_log[state][site] == 2:
                nsites += 0.5
        for even_step in range(0, dt_count):
            unique_config_frequency[state][even_step] *= nsites

    plt.stackplot(np.linspace(0, total_time, dt_count), unique_config_frequency)
    plt.show()
    plt.close()


def stratified_spr_plots_tot(name, total_time, equil_time, concentrations):
    tot_probs = []
    all_states = []
    for conc in range(len(concentrations)):
        full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + str(
            concentrations[conc]) + '/' + name + str(concentrations[conc]) + '.txt'),
                               delimiter="\t")

        dt = 10
        repeat_nr = full_data[:, 0]
        runs_count = int(max(repeat_nr))
        dt_count = int(total_time / dt)

        stay_times = full_data[:, 1]

        change_nr = np.ediff1d(repeat_nr)
        change_nr = np.append(0, change_nr)
        zeros = np.where(change_nr != 0)[0]
        state_configs = full_data[:, 2::]
        len_signature = len(state_configs[0])
        log_of_all_states = np.split(state_configs, zeros)
        log_of_all_times = np.split(stay_times, zeros)
        all_runs = np.zeros((runs_count, dt_count, len_signature))

        for x in range(runs_count):
            log_of_all_times[x] = np.cumsum(log_of_all_times[x])

        for nr in range(runs_count):
            uneven_step = 0
            even_step = 0
            max_step = len(log_of_all_times[nr])
            while uneven_step < max_step and even_step in range(dt_count):
                if log_of_all_times[nr][uneven_step] >= dt * even_step:
                    all_runs[nr, even_step][:] = log_of_all_states[nr][uneven_step][:]
                    even_step += 1
                elif log_of_all_times[nr][uneven_step] < dt * even_step:
                    uneven_step += 1
            if even_step < dt_count - 1:
                while even_step < dt_count:
                    all_runs[nr, even_step][:] = log_of_all_states[nr][-1][:]
                    print("think abt why")

        # ipdb.set_trace()

        unique_config_log = []
        unique_config_frequency = []

        for even_step in range(dt_count):
            for nr in range(runs_count):
                try:
                    index = unique_config_log.index(all_runs[nr, even_step].tolist())
                    unique_config_frequency[index][even_step] += 1
                    # print("I get here")
                except:
                    unique_config_log.append(all_runs[nr, even_step].tolist())
                    unique_config_frequency.append(([0] * dt_count))
                    unique_config_frequency[-1][even_step] += 1

        # ipdb.set_trace() #anjing
        unique_config_frequency_avg = [0] * len(unique_config_frequency)
        for row in range(len(unique_config_frequency)):
            unique_config_frequency_avg[row] = sum(unique_config_frequency[row][equil_time / dt:]) / (
                        len(unique_config_frequency[0]) * 1. - equil_time / (1. * dt))

        for nr in range(len(unique_config_frequency_avg)):
            try:
                index = all_states.index(unique_config_log[nr])
                tot_probs[index][conc] = unique_config_frequency_avg[nr]

            except:
                all_states.append(unique_config_log[nr])
                tot_probs.append([0] * len(concentrations))
                tot_probs[-1][conc] = unique_config_frequency_avg[nr]

    tot_probs = [[float(i) / runs_count for i in a] for a in tot_probs]
    ipdb.set_trace()

    # as of right now I have:
    #   tot_probs with average probabilities of each state at each concentration
    #   all_states aka list of all states
    #   input concentrations tells us all concentrations

    # log all of this info into one file for further analysis:

    if not os.path.exists(str(
            os.getcwd()) + '/' + 'data_logs'):  # test to see if a folder for storing the data exists
        os.makedirs(str(os.getcwd()) + '/' + 'data_logs')  # if no , create the folder
    if not os.path.exists(str(
            os.getcwd()) + '/' + 'data_logs/state_prob/' +
                          name):
        os.makedirs(str(os.getcwd()) + '/' +
                    'data_logs/state_prob/' + name)  # if no , create the folder
    f = open((str(os.getcwd()) + '/' +
              'data_logs/state_prob/' + name + '/' + name + '_prob.txt'), 'w')

    for i in range(len(all_states)):
        for j in range(len(concentrations)):
            for n in range(len(all_states[0])):
                f.write(str(all_states[i][n]) + '\t')
            f.write(str(tot_probs[i][j]) + '\t' + str(concentrations[j]) + '\n')
    f.close()


def gibbs_free_energy(name, rate_constants, time_points, molar_concentrations, distance_matrix):
    ## SORT OUT THE STATE PROBABILITY VS TIME
    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '.txt'),
                           delimiter="\t")
    dt = 10
    repeat_nr = full_data[:, 0]
    runs_count = int(max(repeat_nr))
    dt_count = int(time_points[-1] / dt)

    stay_times = full_data[:, 1]

    # 0 if the run hasn't changed, -1 if it has
    change_nr = np.ediff1d(repeat_nr)
    change_nr = np.append(0, change_nr)
    # ipdb.set_trace()
    zeros = np.where(change_nr != 0)[0]

    state_configs = full_data[:, 2::]
    len_signature = len(state_configs[0])
    log_of_all_states = np.split(state_configs, zeros)
    log_of_all_times = np.split(stay_times, zeros)

    # A matrix that will have states of all runs with const timestep
    all_runs = np.zeros((runs_count, dt_count, len_signature))

    # ipdb.set_trace() # caesar
    for x in range(runs_count):
        log_of_all_times[x] = np.cumsum(log_of_all_times[x])

    for nr in range(runs_count):
        uneven_step = 0
        even_step = 0
        max_step = len(log_of_all_times[nr])
        while uneven_step < max_step and even_step in range(dt_count):
            if log_of_all_times[nr][uneven_step] >= dt * even_step:
                all_runs[nr, even_step][:] = log_of_all_states[nr][uneven_step][:]
                even_step += 1
            elif log_of_all_times[nr][uneven_step] < dt * even_step:
                uneven_step += 1
        if even_step < dt_count - 1:
            while even_step < dt_count:
                all_runs[nr, even_step][:] = log_of_all_states[nr][-1][:]
                print("think abt why")

    # ipdb.set_trace()

    unique_config_log = []
    unique_config_frequency = []

    for even_step in range(dt_count):
        for nr in range(runs_count):
            try:
                index = unique_config_log.index(all_runs[nr, even_step].tolist())
                unique_config_frequency[index][even_step] += 1
                # print("I get here")
            except:
                unique_config_log.append(all_runs[nr, even_step].tolist())
                unique_config_frequency.append(([0] * dt_count))
                unique_config_frequency[-1][even_step] += 1

    # ipdb.set_trace() #samatoki
    unique_config_frequency = [[float(i) / runs_count for i in a] for a in unique_config_frequency]

    ## RATE RATIOS

    deltaG_1 = rate_constants[0] / rate_constants[1]
    deltaG_2 = rate_constants[2] / rate_constants[3]

    ## COUNT NR OF ABS AND BIVALENTLY BOUND AB DELTAGIBBS:
    states_zero_gib = np.zeros((len(unique_config_log), 2))  # one row for gibbs_mono_bi and one for abs count

    for state_nr in range(len(unique_config_log)):
        for i in range(len_signature / 2):
            if unique_config_log[state_nr][i] == 1:
                states_zero_gib[state_nr, 0] += 1  # count all abs
            elif unique_config_log[state_nr][i] == 2:
                states_zero_gib[state_nr, 0] += 0.5  # count all abs
                states_zero_gib[state_nr, 1] += 0.5 * cmath.log((distance_matrix[i, int(
                    unique_config_log[state_nr][len_signature / 2 + i])] * deltaG_2))  # count deltaG from mono->biv

    ## CONVERT TIMES WHEN CONCENTRATION CHANGES TO TIMESTEPS
    time_points = (time_points / dt)

    ## FOR EACH DELTA T DO THE THING!!
    gib_energies = np.zeros(dt_count)
    conc_step = 0
    deltaG_1_conc = molar_concentrations[0] * deltaG_1

    ipdb.set_trace()  # basara
    for time_step in range(dt_count):
        if time_step > time_points[conc_step + 1]:
            conc_step += 1
            for state in range(len(unique_config_log)):
                deltaG_1_conc = deltaG_1 * molar_concentrations[conc_step]  # bind on off rate

        for i in range(len(unique_config_log)):
            gib_energies[time_step] += cmath.log(deltaG_1_conc) * states_zero_gib[i, 0] * unique_config_frequency[i][
                time_step]
            gib_energies[time_step] += states_zero_gib[i, 1] * unique_config_frequency[i][time_step]

    # ipdb.set_trace()
    plt.plot((-8.314 * 293 * gib_energies / runs_count))
    plt.show()
    # ipdb.set_trace()
    plt.close()



def analyze_possible_configs_equilibrium(name, run_count, equil_time):

    # Initialize unique configs and cumulative times in those states
    unique_config_log = []
    unique_config_cumul_t = []


    for run_nr in range(1, run_count + 1):
        # for run_nr in range(1, 50):
        print(run_nr)
        part_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/' + name + '/' + name + '_' + str(run_nr) + '.txt'),
                               delimiter="\t")
        try:
            useless_var = len(part_data[0])
        except:
            useless_var = 1

        if useless_var == 1:
            continue
        cum_times = np.cumsum(part_data[:, 0])
        start_from = bisect.bisect_left(cum_times, equil_time) # Find index at which the system should be equilibrated

        state_configs = part_data[start_from::, 1::] # State configs of the equilibrated part
        times_eq = part_data[start_from::, 0]       # and the stay times in those states

        for nr in range(len(state_configs)):
            try:
                index = unique_config_log.index(state_configs[nr].tolist()) # try finding the state
                unique_config_cumul_t[index] += times_eq[nr] # and adding to total time in that config
                # print("I get here")
            except:
                unique_config_log.append(state_configs[nr].tolist()) # else register the new unique state
                unique_config_cumul_t.append((times_eq[nr]))

    max_states = len(unique_config_log)
    unique_config_cumul_t_a = np.array(unique_config_cumul_t)
    inds = unique_config_cumul_t_a.argsort()[::-1]
    # ipdb.set_trace() # uraja
    sorted_configs = np.array(unique_config_log)[inds]
    sorted_times = unique_config_cumul_t_a[inds]

    if not os.path.exists(str(
            os.getcwd()) + '/' + 'data_logs'):  # test to see if a folder for storing the data exists
        os.makedirs(str(os.getcwd()) + '/' + 'data_logs')  # if no , create the folder
    if not os.path.exists(str(
            os.getcwd()) + '/' + 'data_logs/state_prob/' +
                          name):
        os.makedirs(str(os.getcwd()) + '/' +
                    'data_logs/state_prob/' + name)  # if no , create the folder
    f = open((str(os.getcwd()) + '/' +
              'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_states.txt'), 'w')

    for i in range(len(sorted_configs)):
        f.write('\n' + str(sorted_times[i]))
        for j in range(len(sorted_configs[0])):
            f.write('\t' + str(sorted_configs[i][j]))
    f.close()

    return max_states


def plot_already_calculated_times_of_states(name, max_states):
    time_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_states.txt'),
                           delimiter="\t")
    times = time_data[:,0]

    plt.plot([i for i in range(1, len(times) + 1)], times)
    plt.ylabel("cumulative time in the state")
    plt.xlabel("unique state")
    plt.grid(color='r', linestyle='-', linewidth=.2)
    # plt.show()
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(
        max_states) + '_times_in_separate_states.png')
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(
        max_states) + '_times_in_separate_states.svg')
    plt.close()


def equivalent_sorted_states(name, distance_matrix, max_states):
    sorted_states = np.loadtxt(
        (str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_states.txt'),
        delimiter="\t")

    states = sorted_states[:,1::]
    ag_count = int(len(states[0])/2)

    sigs = np.zeros((max_states, 2))

    for state in range(0, max_states):
        ab_count = 0
        distsum = 0
        for i in range(0, ag_count):
            if states[state, i] == 1:
                ab_count += 1
            elif states[state, i] == 2:
                distsum += 0.5 * distance_matrix[i, int(states[state, ag_count + i])]
                ab_count += 0.5
        sigs[state, 0] = distsum
        sigs[state, 1] = ab_count

    times = sorted_states[:, 0]



    # plt.bar([i for i in range(1, len(times) + 1)], times)
    # plt.scatter([i for i in range(1, len(times) + 1)], gibbs_ens)
    plt.scatter(sigs[:,0], times, c=sigs[:,1])
    plt.ylabel("cumulative time")
    plt.xlabel("distsum")
    plt.show()
    # plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_gibbs_of_sorted_states.png')
    # plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_gibbs_of_sorted_states.svg')
    plt.close()



def equivalent_states_total_times(name, distance_matrix, max_states, optimal_distance):
    sorted_states = np.loadtxt(
        (str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_states.txt'),
        delimiter="\t")

    times = sorted_states[:, 0]
    states = sorted_states[:,1::]
    ag_count = int(len(states[0])/2)


    cumul_times = []
    original_state_stdv = []
    original_state_ab_count = []
    original_state_ab_count_biv = []

    for state in range(0, max_states):
        ab_count = 0
        biv_ag_count = 0
        stdv = 0
        for i in range(0, ag_count):
            if states[state, i] == 1:
                ab_count += 1
            elif states[state, i] == 2:
                stdv += 0.5 * (distance_matrix[i, int(states[state, ag_count + i])] - optimal_distance)**2
                biv_ag_count += 0.5
                ab_count += 0.5

        # ipdb.set_trace() #ananas
        if stdv in original_state_stdv:
            index = np.where(np.array(original_state_stdv) == stdv)[0] #find all places where the condition is fulfilled
            found = 0
            # ipdb.set_trace() #chottodakebaka
            for place in index:
                if original_state_ab_count[place] == ab_count:
                    cumul_times[place] += times[state]
                    found = 1
                    break
            if found == 0:
                original_state_stdv.append(stdv)
                cumul_times.append(times[state])
                original_state_ab_count.append(ab_count)
                original_state_ab_count_biv.append(biv_ag_count)

        else:
            original_state_stdv.append(stdv)
            cumul_times.append(times[state])
            original_state_ab_count.append(ab_count)
            original_state_ab_count_biv.append(biv_ag_count)
    # ipdb.set_trace() # everybodyrockyourbody


    original_state_stdv = np.divide(original_state_stdv,original_state_ab_count_biv)
    labels = []
    for i in range(int(min(original_state_ab_count)), int(max(original_state_ab_count))+1):
        plt.bar(np.extract(np.array(original_state_ab_count) == i, np.array(original_state_stdv)), np.extract(np.array(original_state_ab_count) == i, np.array(cumul_times)))
        labels.append(i)
        # ipdb.set_trace() #xyzkanjigaii
    plt.legend(labels)
    # plt.show()
    # ipdb.set_trace() #cHIKUSHYOOO
    # plt.bar([i for i in range(1, len(times) + 1)], times)
    # plt.scatter([i for i in range(1, len(times) + 1)], gibbs_ens)
    # plt.scatter(original_state_dsum, cumul_times, c=original_state_ab_count)
    plt.ylabel("cumulative time")
    plt.xlabel("dsum")
    # plt.show()
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_total_gibbs_ens.png')
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_total_gibbs_ens.svg')
    # plt.show()

    plt.close()


def gibbs_of_sorted_states(name, rate_constants, molar_concentration, distance_matrix, max_states):
    sorted_states = np.loadtxt(
        (str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_states.txt'),
        delimiter="\t")

    states = sorted_states[:,1::]
    ag_count = int(len(states[0])/2)

    gibbs_ens = np.zeros(max_states)
    deltaG1 = cmath.log(rate_constants[0] / rate_constants[1] * molar_concentration[0])

    deltaG2 = np.zeros((ag_count, ag_count))
    for i in range (0, ag_count-1):
        for j in range(i+1, ag_count):
            # ipdb.set_trace() #brilliantovajaruka
            if alpha(distance_matrix[i,j]) != 0:
                deltaG2[i,j] = (cmath.log(rate_constants[2] / rate_constants[3] * alpha(distance_matrix[i,j]))).real
                deltaG2[j,i] = deltaG2[i,j]

    ipdb.set_trace()  # bilar
    for state in range(0, max_states):
        ab_count = 0
        bivalent_deltaG = 0
        for i in range(0, ag_count):
            if states[state, i] == 1:
                ab_count += 1
            elif states[state, i] == 2:
                bivalent_deltaG += 0.5* (deltaG2[i, ag_count + i])
                ab_count += 0.5
        gibbs_ens[state] = bivalent_deltaG.real + ab_count * deltaG1.real
        if bivalent_deltaG.imag != 0:
            print("IM GETTING IMAGINARY VALUES FOR SOME REASON")

    times = sorted_states[:, 0]


    # plt.bar([i for i in range(1, len(times) + 1)], times)
    # plt.scatter([i for i in range(1, len(times) + 1)], gibbs_ens)
    plt.scatter(gibbs_ens, times)
    plt.ylabel("cumulative time")
    plt.xlabel("delta G")
    # plt.show()
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_gibbs_of_sorted_states.png')
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_gibbs_of_sorted_states.svg')
    plt.close()


def gibbs_total_times(name, rate_constants, molar_concentration, distance_matrix, max_states):
    sorted_states = np.loadtxt(
        (str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_states.txt'),
        delimiter="\t")

    times = sorted_states[:, 0]
    states = sorted_states[:,1::]
    ag_count = int(len(states[0])/2)

    deltaG1 = cmath.log(rate_constants[0] / rate_constants[1] * molar_concentration[0])
    deltaG2 = np.zeros((ag_count, ag_count))
    for i in range (0, ag_count-1):
        for j in range(i+1, ag_count):
            # ipdb.set_trace() #brilliantovajaruka
            if alpha(distance_matrix[i,j]) != 0:
                deltaG2[i,j] = (cmath.log(rate_constants[2] / rate_constants[3] * alpha(distance_matrix[i,j]))).real
                deltaG2[j,i] = deltaG2[i,j]
            # print(i)
            # print(j)
            # ipdb.set_trace() # aintnothingbutaheartbreak
    # ipdb.set_trace() #bilar

    cumul_times = []
    original_gibbs_en = []

    for state in range(0, max_states):
        ab_count = 0
        bivalent_deltaG = 0
        for i in range(0, ag_count):
            if states[state, i] == 1:
                ab_count += 1
            elif states[state, i] == 2:
                bivalent_deltaG += 0.5 * deltaG2[i, int(states[state, ag_count + i])]
                ab_count += 0.5
        gibbs_en = bivalent_deltaG.real + ab_count * deltaG1.real
        if bivalent_deltaG.imag != 0:
            print("IM GETTING IMAGINARY VALUES FOR SOME REASON")

        if gibbs_en in original_gibbs_en:
            index = original_gibbs_en.index(gibbs_en)  # try finding the state
            cumul_times[index] += times[state]  # and adding to total time in that config
        else:
            original_gibbs_en.append(gibbs_en)
            cumul_times.append(times[state])
    # ipdb.set_trace() # everybodyrockyourbody




    # plt.bar([i for i in range(1, len(times) + 1)], times)
    # plt.scatter([i for i in range(1, len(times) + 1)], gibbs_ens)
    plt.hist(original_gibbs_en, bins=int(max(original_gibbs_en) - min(original_gibbs_en)), weights = cumul_times)
    plt.ylabel("cumulative time")
    plt.xlabel("discretized delta G")
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_total_gibbs_ens.png')
    plt.savefig(str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_total_gibbs_ens.svg')
    # plt.show()

    plt.close()

def generate_state_fig(name, coordinates, max_states):
    full_data = np.loadtxt((str(os.getcwd()) + '/' + 'data_logs/state_prob/' + name + '/' + name + '_ ' + str(max_states) + '_states.txt'),
                           delimiter="\t")

    ipdb.set_trace()
    configs = full_data[:, 1::]
    state_count = len(full_data)
    # ipdb.set_trace() #kinpuri
    nr_sites = int(len(configs[0])/2)

    maxx = np.max(coordinates[:, 1])
    minx = np.min(coordinates[:, 1])
    maxy = np.max(coordinates[:, 0])
    miny = np.min(coordinates[:, 0])
    centerx = (maxx + minx) / 2
    centery = (maxy + miny) / 2

    if not os.path.exists(str(os.getcwd()) + '/data_logs/state_prob/' + name + '/figs'):
        os.makedirs(str(os.getcwd()) + '/data_logs/state_prob/' + name + '/figs')

    for state in range(0, state_count):
        plt.figure(figsize=(5, 5))
        my_plt_format(2, 5, 30)

        plt.scatter(coordinates[:, 1], coordinates[:, 0], marker='o', s=2000, facecolors='none',
                edgecolors='#11B5E5', linewidth=1)
        plt.ylim(centery - (maxy - miny) / 2 - 10, centery + (maxy - miny) / 2 + 10)
        plt.xlim(centerx - (maxx - miny) / 2 - 10, centerx + (maxx - minx) / 2 + 10)
        plt.axis('off')
        for site in range(0, nr_sites):
            if int(configs[state, site]) == 1:
                plt.scatter(coordinates[site, 1], coordinates[site, 0], c='#FF00A6', s=20, alpha=1,
                        edgecolors='#FF00A6', linewidth='2')
            elif int(configs[state, site]) == 2:
                plt.scatter(coordinates[site, 1], coordinates[site, 0], c='#FF00A6', s=20, alpha=1,
                        edgecolors='#FF00A6', linewidth='2')
                plt.plot([(coordinates[site, 1]), coordinates[int(configs[state, nr_sites + site]), 1]],
                    [(coordinates[site, 0]), coordinates[int(configs[state, nr_sites + site]), 0]],
                    '#FF00A6', linewidth=3, alpha=1)
            else:
                pass

        plt.savefig(str(os.getcwd()) + '/data_logs/state_prob/' + name + '/figs' + '/' +
                name + str(state) + ".png")
        plt.savefig(str(os.getcwd()) + '/data_logs/state_prob/' + name + '/figs' + '/' +
                name + str(state) + ".svg")
        plt.close('all')


def my_plt_format(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams.update(
        {'font.size': 15, 'figure.figsize': (6, 6), 'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
            'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'})