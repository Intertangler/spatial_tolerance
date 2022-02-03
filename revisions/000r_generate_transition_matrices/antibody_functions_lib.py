import os as os
import numpy as np
#import tqdm
import ipdb
import re
import matplotlib.pyplot as plt
#import networkx as nx
from StringIO import StringIO
import matplotlib.cm as cm
from matplotlib.image import BboxImage,imread
#from PIL import Image
#import plotting_for_manuscript as plotms
import random
import time as time  # for creating dir with current time?
import datetime
import scipy.optimize as opt
# import math as math


########### CREATE A NEW DIRECTORY WITH CURRENT DATE TIME AND NAME TAKEN AS AN INPUT ##############
def create_directory(name):
    print('create_directory')
    global directory_name
    global today
    today = datetime.date.today()
    directory_name = str(today) + '__' + str(time.time()) + str(name)
    os.makedirs(directory_name)
    return directory_name


def configure_space(cutoff_dist, coord_v, transition_m_name):
    site_status_v = np.zeros((len(coord_v)))
    dist_m = np.zeros((len(coord_v), len(coord_v)))
    neigh_m = np.zeros((len(coord_v), len(coord_v)))
    neigh_dir = np.empty((len(coord_v)), dtype=np.ndarray)
    for pair1 in (range(0, len(coord_v))):
        for pair2 in (range(pair1, len(coord_v))):
            dist_pair = np.sqrt((coord_v[pair2][0]-coord_v[pair1][0])**2 + (coord_v[pair2][1]-coord_v[pair1][1])**2)
            dist_m[pair1, pair2] = dist_pair
            dist_m[pair2, pair1] = dist_pair
            if dist_pair < cutoff_dist and dist_pair != 0:
                neigh_m[pair1, pair2] = 1
                neigh_m[pair2, pair1] = 1
                neigh_dir[pair1] = np.append(neigh_dir[pair1], np.array([pair2]))
                neigh_dir[pair2] = np.append(neigh_dir[pair2], np.array([pair1]))
            else:
                pass

    # plotting part
    plot_configure_space(coord_v, neigh_dir, transition_m_name)

    # dist_m and neigh_m just initialized so maybe put them somewhere else?
    return site_status_v, dist_m, neigh_m, neigh_dir


def get_state_id(site_status_v, pointer_v):
    current_site_id = str()
    for i in range(0, len(site_status_v)):
        current_site_id = current_site_id + str(int(site_status_v[i])) + ','
    current_site_id += '\n'
    for i in range(0, len(site_status_v)):
        current_site_id = current_site_id + str(pointer_v[i]) + ','
    return current_site_id


def generate_state_from_id(input_state_id):
    new_site_status_v = np.fromstring(re.split('\n', input_state_id)[0], sep=",")
    new_pointer_v = np.fromstring(re.split('\n', input_state_id)[1], sep=",")
    return new_site_status_v, new_pointer_v


def update_queue(state_id, bfsq, visited_nodes_list):
    if state_id in bfsq or state_id in visited_nodes_list:
        pass
    else:
        bfsq.append(state_id)
    return bfsq


# def update_states(stateID,list_of_state_identifiers,transition_matrix,occupancy_key,distance_transition_matrix,
#                   discoveries, site_status_vector, coordinate_vector, pointer_vector, transition_matrix_name):
#     import ipdb
def update_states(state_id, list_of_state_ids, transition_m, occ_key, dist_transition_m, site_status_v, coord_v, pointer_v, transition_m_name,particle_count_key):
    if state_id in list_of_state_ids:
        pass
    else:
        plot_update_states(coord_v, site_status_v, pointer_v, transition_m_name, list_of_state_ids)

        list_of_state_ids.append(state_id)
        new_m = np.zeros((len(transition_m)+1, len(transition_m)+1))
        new_dist_transition_m = np.zeros((len(dist_transition_m) + 1, len(dist_transition_m) + 1))
        new_array = np.zeros((len(transition_m) + 1))
        new_array_2 = np.zeros((len(transition_m) + 1,2))

        for i in range(0, len(transition_m)):
            new_array[i] = occ_key[i]  # copy the old occupancy keys
            new_array_2[i] = particle_count_key[i]
            for j in range(0, len(transition_m)):
                new_m[i, j] = transition_m[i, j]  # copy old transitions
                new_dist_transition_m[i, j] = dist_transition_m[i, j]  # copy old distance transitions
        # rewrite old vals of transition matrix, dist transition matrix and occ key
        transition_m = new_m
        dist_transition_m = new_dist_transition_m
        occ_key = new_array
        particle_count_key = new_array_2
        
        # calculate new occupancy key
        count = 0
        mono_count = 0
        biv_count = 0
        for i in range(0, len(site_status_v)):
            if site_status_v[i] == 1:
                count = count + 1
                mono_count += 1
            elif site_status_v[i] == 2:
                count = count + 0.5
                biv_count += 0.5
            else:
                pass
        occ_key[len(occ_key) - 1] = count  # last new element of occ key
        particle_count_key[len(occ_key)-1][0] = mono_count
        particle_count_key[len(occ_key)-1][1] = biv_count
    return list_of_state_ids, transition_m, occ_key, dist_transition_m, particle_count_key


def plot_update_states(coord_v, site_status_v, pointer_v, transition_m_name, list_of_state_ids):

    plt.figure(figsize=(5, 5))
    my_plot_format(20, 20, 30)
#     ipdb.set_trace()
    span_of_y = np.max([coord_v[:, 0], coord_v[:, 1]]) - np.min([coord_v[:, 0], coord_v[:, 1]])
    span_of_x = np.max([coord_v[:, 0], coord_v[:, 1]]) - np.min([coord_v[:, 0], coord_v[:, 1]])
    center_x = span_of_x / 2
    center_y = span_of_y / 2
    plt.scatter(coord_v[:, 1], coord_v[:, 0], marker='o', s=2000, facecolors='#BF3A1A',
                edgecolors='k', linewidth=10)
    # plt.title('antibody binding state' + str(stateID))
    plt.ylim(center_y - span_of_y / 2 - 4, center_y + span_of_y / 2 + 4)
    plt.xlim(center_x - span_of_x / 2 - 4, center_x + span_of_x / 2 + 4)

    plt.axis('off')

    # next, a simple check for monovalently bound antibodies which will be highlighted
    # with scatter plot points in a new color
    for site in range(0, len(site_status_v)):
        if int(site_status_v[site]) == 1:
            plt.scatter(coord_v[site, 1], coord_v[site, 0], c='#5384C7', s=5000, alpha=1,
                        edgecolors='k', linewidth=12)
        elif int(site_status_v[site]) == 2:
            #now search for bivalent points in the site status vector and connect them with ...
            # ...pyplot using the pointer matrix
            plt.scatter(coord_v[site, 1], coord_v[site, 0], c='#5384C7', s=5000, alpha=1,
                        edgecolors='k', linewidth=12)
            edge_dist = np.sqrt(((coord_v[site, 1]) - coord_v[int(pointer_v[site]), 1]) ** 2 + ((coord_v[site, 0]) - coord_v[int(pointer_v[site]), 0]) ** 2)
            # plt.text(np.mean([int(coord_v[site, 1]),coord_v[int(pointer_v[site]), 1]]),np.mean([int(coord_v[site, 0]),coord_v[int(pointer_v[site]), 0]]),str(edge_dist),fontsize=20)
#             plt.plot([(coord_v[site, 1]), coord_v[int(pointer_v[site]), 1]],
#                      [(coord_v[site, 0]), coord_v[int(pointer_v[site]), 0]],
#                      'k', linewidth=35, alpha=1)
            plt.plot([(coord_v[site, 1]), coord_v[int(pointer_v[site]), 1]],
                     [(coord_v[site, 0]), coord_v[int(pointer_v[site]), 0]],
                     '#5384C7', linewidth=40, alpha=1)

        else:
            pass
    
    if not os.path.exists(str(
            os.getcwd()) + '/' + 'state_configurations'):  # test to se if a folder for storing the data exists
        os.makedirs(str(os.getcwd()) + '/' + 'state_configurations')  # if no , create the folder
    # ipdb.set_trace()
    if not os.path.exists(str(os.getcwd()) + '/' + 'state_configurations' + '/' + transition_m_name):  # test to se if a folder for storing the data exists
        os.makedirs(str(os.getcwd()) + '/' + 'state_configurations' + '/' + transition_m_name)  # if no , create the folder
    print(type(center_x))
    # ipdb.set_trace()
    # THE TWO LINES CAUSING ERRORS

    plt.savefig(str(os.getcwd()) + '/' + 'state_configurations' + '/' + transition_m_name + '/' + transition_m_name +
                str((len(list_of_state_ids) + 1)) + ".png")
    plt.savefig(str(str(os.getcwd()) + '/' + 'state_configurations' + '/' + transition_m_name + '/' +
                transition_m_name + str((len(list_of_state_ids) + 1)) + ".svg"))

    plt.show()
    # END OF TWO LINES CAUSING ERRORS
    plt.close()
#     ipdb.set_trace()

def my_plot_format(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    # SMALL_SIZE = 10
    # MEDIUM_SIZE = 20
    # BIGGER_SIZE = 30
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


def plot_configure_space(coord_v, neigh_dir, transition_m_name):
    plt.figure(figsize=(5, 5))

    for i in range(0, len(coord_v)):
        try:
            neighbor_list = neigh_dir[i][1:]
        except:
            neighbor_list = np.array([])
        for neighbor in range(0, len(neighbor_list)):
            
            plt.plot([coord_v[i][1], coord_v[neighbor_list[neighbor]][1]],
                     [coord_v[i][0], coord_v[neighbor_list[neighbor]][0]], 'k--', linewidth=5,
                     alpha=1)
            plt.scatter(coord_v[:, 1], coord_v[:, 0], marker='o', s=2000, facecolors='#BF3A1A',
                edgecolors='k', linewidth=10)
#             plt.scatter(coord_v[:, 1], coord_v[:, 0], marker='o',s=2000, c='#BF3A1A',
#                 linewidth=30)
            
    if not os.path.exists(str(
            os.getcwd()) + '/' + 'State_Space'):  # test to se if a folder for storing the data exists
        os.makedirs(str(os.getcwd()) + '/' + 'State_Space')  # if no , create the folder
    if not os.path.exists(str(
            os.getcwd()) + '/' + 'State_Space/' +
                          transition_m_name):
        os.makedirs(str(os.getcwd()) + '/' +
                    'State_Space/' + transition_m_name)  # if no , create the folder
    plt.xlabel('x position [nm]')
    plt.ylabel('y position [nm]')
#     plt.scatter(coord_v[:, 1], coord_v[:, 0], c='w', s=2000)
    plt.ylim(np.min([coord_v[:, 0], coord_v[:, 1]]) - 6,
             np.max([coord_v[:, 0], coord_v[:, 1]]) + 6)
    plt.xlim(np.min([coord_v[:, 0], coord_v[:, 1]]) - 6,
             np.max([coord_v[:, 0], coord_v[:, 1]]) + 6)
    plt.savefig(str(os.getcwd()) + '/' + 'State_Space/' +
                transition_m_name + '/' + "lattice_config" + ".png")
    plt.savefig(str(os.getcwd()) + '/' + 'State_Space/' +
                transition_m_name + '/' + "lattice_config" + ".svg")
    plt.close()


def intersect(p1,p2,p3,p4):
    x1,x2,x3,x4 = p1[0],p2[0],p3[0],p4[0]
    y1,y2,y3,y4 = p1[1],p2[1],p3[1],p4[1]
    if max(x1, x2) < min(x3, x4):
        return False  # There is no mutual abcisse
    else:
        a1 = (y1 - y2) / (x1 - x2)
        a2 = (y3 - y4) / (x3 - x4)
        b1 = y1 - a1 * x1 # = Y2 - A1 * X2
        b2 = y3 - a2 * x3 # = Y4 - A2 * X4
        if a1 == a2:
            return False
        else:
            xa = (b2 - b1) / (a1 - a2)
            if xa < max(min(x1, x2), min(x3, x4)) or xa > min(max(x1, x2), max(x3, x4)):
                return False # intersection is out of bound
            else:
                return True


def transition_m_continuous(coord_v, site_status_v, neigh_dir, dist_m, transition_m_name):
    # -0.42 if no bivalent coupling, pointer to a friend otherwise
    pointer_v = np.ones((len(coord_v))) * -0.42
    current_globalstate_id = get_state_id(site_status_v, pointer_v)
    list_of_state_ids = []
    k_on = 1
    k_off = 2
    k_mono_bi = 3
    k_bi_mono = 4

    transition_m = np.zeros((0, 0))
    dist_transition_m = np.zeros((0, 0))
    occ_key = np.zeros((0))
    particle_count_key = np.zeros((0))

    current_globalstate_id = get_state_id(site_status_v, pointer_v)

    list_of_state_ids, transition_m, occupancy_key, dist_transition_m,particle_count_key = \
    update_states(current_globalstate_id, list_of_state_ids, transition_m, occ_key, dist_transition_m,
                  site_status_v, coord_v, pointer_v, transition_m_name,particle_count_key)

    # initialize the queue
    bfsq = []
    visited_states = []
    bfsq.append(list_of_state_ids.index(current_globalstate_id))
    revert_state_id = ''
    revert_state_id = revert_state_id + current_globalstate_id

    while len(bfsq) > 0:
        print(len(bfsq))
        for current_site in (range(0, len(coord_v))):

            current_site_state = site_status_v[current_site]

            if current_site_state == 1:
                for decision in range(1,3):
                    if decision == 1:
#                         import pdb; pdb.set_trace()
                        try:
                            neigh_options = neigh_dir[current_site][1:]
                        except:
                            neigh_options = np.array([])

                        bivalent_conversion = 0

                        for i in range(0, len(neigh_options)):
                            obstruction = False

                            for other_neigh in range(0, len(neigh_options)):

                                if site_status_v[neigh_options[other_neigh]] == 2 and intersect(coord_v[current_site],
                                                                                               coord_v[neigh_options[i]],
                                                                                               coord_v[neigh_options[other_neigh]],
                                                                                               coord_v[int(pointer_v[neigh_options[other_neigh]])]):
                                    obstruction = True
                                    break

                            if site_status_v[neigh_options[i]] == 0 and site_status_v[current_site] == 1 and \
                                    obstruction == False:
                                site_status_v[neigh_options[i]] = 2  # empty neighbor site was found, convert
                                site_status_v[current_site] = 2
                                
                                pointer_v[neigh_options[i]] = current_site
                                # update the pointer lattices with new coordinates
                                pointer_v[current_site] = neigh_options[i]
                                bivalent_conversion = 1
                                current_globalstate_id = get_state_id(site_status_v, pointer_v)
                                list_of_state_ids, transition_m, occupancy_key, \
                                dist_transition_m, particle_count_key = \
                                    update_states(current_globalstate_id,
                                                  list_of_state_ids, transition_m, occupancy_key,
                                                  dist_transition_m, site_status_v,
                                                  coord_v, pointer_v, transition_m_name,particle_count_key)

                                # add a transition to the transition matrix
                                from_state_index = list_of_state_ids.index(revert_state_id)
                                to_state_index = list_of_state_ids.index(current_globalstate_id)
                                dist_transition_m[from_state_index, to_state_index] = \
                                    dist_m[current_site, int(pointer_v[current_site])]
                                dist_transition_m[to_state_index, from_state_index] = \
                                    dist_m[current_site, int(pointer_v[current_site])]
                                transition_m[from_state_index, to_state_index] = k_mono_bi
                                # here we dont include the distance dependence, because later the rates will be
                                # determined, now we just identify what class of process the transition
                                # matrix should reflect
                                transition_m[to_state_index, from_state_index] = k_bi_mono
                                # by symmetry, we can add the reverse transition each time this happens as well!

                                bfsq = update_queue(
                                    list_of_state_ids.index(current_globalstate_id),
                                    bfsq, visited_states)
                                site_status_v, pointer_v = generate_state_from_id(revert_state_id)  # REVERT!
                                current_globalstate_id = get_state_id(site_status_v, pointer_v)

                        if bivalent_conversion == 0:  # no viable neighbor was discovered, the block above this one
                            # was exhausted without finding a viable neighbor, and bivalent_conversion
                            # switch was never flipped
                            decision = 2
                            # print 'no neighbor found, new decision =', decision
                            # print 'monovalent removal - no bivalent candidate found' # since no candidate for a
                            # bivalent addition was found, the only remaining option to continue state exploration
                            # is by removing the monovalently bound antibody that we discovered in this site
                            # update status of converted mono to empty site
                            site_status_v[current_site] = 0
                            current_site_state = site_status_v[current_site]
                            current_globalstate_id = get_state_id(site_status_v, pointer_v)
                            # check if the newly formed global state is on the list and update if not
                            list_of_state_ids, transition_m, occupancy_key, dist_transition_m, particle_count_key, \
                                = update_states(current_globalstate_id, list_of_state_ids,
                                                        transition_m, occupancy_key,
                                                        dist_transition_m, site_status_v,
                                                        coord_v, pointer_v, transition_m_name,particle_count_key)
                            # add a transition to the transition matrix
                            from_state_index = list_of_state_ids.index(revert_state_id)
                            to_state_index = list_of_state_ids.index(current_globalstate_id)
                            transition_m[from_state_index, to_state_index] = k_off
                            transition_m[to_state_index, from_state_index] = k_on  # by symmetry, we can add the
                            # reverse transition each time this happens as well!

                            bfsq = update_queue(list_of_state_ids.index(current_globalstate_id), bfsq, visited_states)
                            site_status_v, pointer_v = generate_state_from_id(revert_state_id)  # REVERT!
                            current_globalstate_id = get_state_id(site_status_v, pointer_v)

                    else:
                        # monovalent removal
                        # update status of converted mono to empty site
                        site_status_v[current_site] = 0
                        current_site_state = site_status_v[current_site]
                        current_globalstate_id = get_state_id(site_status_v, pointer_v)
                        # check if the newly formed global state is on the list and update if not
                        list_of_state_ids, transition_m, occupancy_key, dist_transition_m, particle_count_key\
                            = update_states(current_globalstate_id, list_of_state_ids, transition_m, occupancy_key,
                                            dist_transition_m, site_status_v, coord_v, pointer_v, transition_m_name,particle_count_key)
                        # add a transition to the transition matrix
                        from_state_index = list_of_state_ids.index(revert_state_id)
                        to_state_index = list_of_state_ids.index(current_globalstate_id)
                        transition_m[from_state_index, to_state_index] = k_off
                        transition_m[to_state_index, from_state_index] = k_on  # by symmetry, we can add the
                        # reverse transition each time this happens as well!

                        bfsq = update_queue(list_of_state_ids.index(current_globalstate_id), bfsq, visited_states)
                        site_status_v, pointer_v = generate_state_from_id(revert_state_id)  # REVERT!
                        current_globalstate_id = get_state_id(site_status_v, pointer_v)

            elif current_site_state == 2:
                # convert to monovalent

                # first update the lattice - the current site is reset to 1, then use current coordinates in the
                # two pointer matrices to locate the current site's partner and convert it
                for decision2 in range(1, 3):
                    if decision2 == 1:
                        site_status_v[current_site] = 1
                        site_status_v[int(pointer_v[current_site])] = 0
                    else:
                        site_status_v[current_site] = 0
                        site_status_v[int(pointer_v[current_site])] = 1

                    # now we need to set the pointer vectors back to empty, start with the partner site
                    pointer_place_holder = pointer_v[current_site]
                    pointer_v[int(pointer_v[current_site])] = -.42
                    pointer_v[current_site] = -.42

                    # now update the new transition matrix and lsit of state id's as usual
                    current_site_state = site_status_v[current_site]
                    current_globalstate_id = get_state_id(site_status_v, pointer_v)
                    # check if the newly formed global state is on the list and update if not
                    list_of_state_ids, transition_m, occupancy_key, dist_transition_m,particle_count_key \
                        = update_states(current_globalstate_id, list_of_state_ids, transition_m, occupancy_key,
                                        dist_transition_m, site_status_v, coord_v, pointer_v, transition_m_name,particle_count_key)
                    # add a transition to the transition matrix
                    from_state_index = list_of_state_ids.index(revert_state_id)
                    to_state_index = list_of_state_ids.index(current_globalstate_id)
                    dist_transition_m[from_state_index, to_state_index] = dist_m[int(current_site),
                                                                                 int(pointer_place_holder)]
                    dist_transition_m[to_state_index, from_state_index] = dist_m[int(current_site),
                                                                                 int(pointer_place_holder)]
                    transition_m[from_state_index, to_state_index] = k_bi_mono
                    transition_m[to_state_index, from_state_index] = k_mono_bi  # by symmetry, we can add the
                    # reverse transition each time this happens as well!

                    bfsq = update_queue(list_of_state_ids.index(current_globalstate_id), bfsq, visited_states)
                    site_status_v, pointer_v = generate_state_from_id(revert_state_id)  # REVERT!
                    current_globalstate_id = get_state_id(site_status_v, pointer_v)
                    

            else:  # add a monovalent binding
                # update status of converted mono to empty site
                site_status_v[int(current_site)] = 1
                current_site_state = site_status_v[int(current_site)]
                current_globalstate_id= get_state_id(site_status_v, pointer_v)
                # check if the newly formed global state is on the list and update if not
                list_of_state_ids, transition_m, occupancy_key, dist_transition_m, particle_count_key \
                    = update_states(current_globalstate_id, list_of_state_ids, transition_m, occupancy_key,
                                    dist_transition_m, site_status_v, coord_v, pointer_v, transition_m_name, particle_count_key)
                # add a transition to the transition matrix
                from_state_index = list_of_state_ids.index(revert_state_id)
                to_state_index = list_of_state_ids.index(current_globalstate_id)
                transition_m[from_state_index, to_state_index] = k_on
                transition_m[to_state_index, from_state_index] = k_off  # by symmetry, we can add the reverse
                # transition each time this happens as well!
                # print transition_matrix

                bfsq = update_queue(list_of_state_ids.index(current_globalstate_id), bfsq, visited_states)
                site_status_v, pointer_v = generate_state_from_id(revert_state_id)  # REVERT!
                current_globalstate_id = get_state_id(site_status_v, pointer_v)

            site_status_v, pointer_v = generate_state_from_id(revert_state_id)

        visited_states.append(bfsq[0])
        bfsq = bfsq[1::]

        try:
            site_status_v, pointer_v = generate_state_from_id(list_of_state_ids[bfsq[0]])
            current_globalstate_id = get_state_id(site_status_v, pointer_v)
            revert_state_id = current_globalstate_id
        except:
            continue

    if not os.path.exists(str(os.getcwd()) + '/transition_matrices'):  #test to se if a folder for
        # storing the data exists
        os.makedirs(str(os.getcwd()) + '/transition_matrices')  # if no , create the folder
    np.save(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name), transition_m)
    np.save(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'occupancy_key', occupancy_key)
    np.save(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'particle_count_key', particle_count_key)
    np.save(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'distance_transition_matrix',dist_transition_m)
    
    
    
    
    np.savetxt(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name), transition_m)
    np.savetxt(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'occupancy_key', occupancy_key)
    np.savetxt(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'particle_count_key', particle_count_key)
    np.savetxt(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'distance_transition_matrix', dist_transition_m)
    #plot_transition_m_continuous(transition_m, occupancy_key, transition_m_name, dist_transition_m)


def plot_transition_m_continuous(transition_m, occupancy_key, transition_m_name, dist_transition_m):
    count = 0
    for i in range(len(transition_m)):
        for j in range(len(transition_m)):
            if transition_m[i, j] != 0:
                count = count + 1
            else:
                pass
    print(len(transition_m), count, occupancy_key)
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    ax.imshow(transition_m, interpolation='nearest', cmap='terrain_r')
    # plt.colorbar(label='transition rate \n $\lambda ^{.25}$')
    # ax.set_xlabel("to state
    ax.set_xlabel('to state j ', fontsize=200)
    ax.xaxis.set_label_coords(.5, -0.05)
    # ax.axis('off')
    ax.set_ylabel('from state i', fontsize=200)
    ax.yaxis.set_label_coords(-0.05, .5)
    startx, endx = ax.get_xlim()
    starty, endy = ax.get_xlim()
    stepsize = 1
    ax.xaxis.set_ticks(np.arange(0-0.5, endx-0.5, stepsize))
    ax.yaxis.set_ticks(np.arange(0-0.5, endy-0.5, stepsize))
    ax.grid(color='k', linestyle='-', linewidth=1.0)
    labels = []
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title('states: ' + str(len(transition_m)) + '\n transitions: ' + str(count), fontsize=200)
    if not os.path.exists(str(
            os.getcwd()) + '/' + 'State_Space/' +
                                  transition_m_name):  # test to se if a folder for storing the data exists
        os.makedirs(str(os.getcwd()) + '/' + 'State_Space' + '/' + transition_m_name)  # if no , create the folder
    fig.savefig(str(os.getcwd()) + '/' + 'State_Space' + '/' + transition_m_name + '/' + "transition_matrix.svg")
    fig.savefig(str(os.getcwd()) + '/' + 'State_Space' + '/' + transition_m_name + '/' + "transition_matrix.jpg")
    plt.close()

    plt.imshow((distance_transition_matrix),interpolation='nearest',cmap='coolwarm')
    plt.colorbar(label='transition rate \n $\lambda ^{.25}$')
    plt.xlabel("to state n")
    plt.axis('off')
    plt.ylabel("from state m")
    plt.title('states: ' + str(len(transition_matrix)) + '\n transitions: ' + str(count) +
              '\n occupancy key: ' + str(occupancy_key), fontsize=8)
    plt.savefig(str(os.getcwd()) +'/' + str(directory_name) +'/' + str(today)+
                "transition_matrix"+str(time.time())+".png")
    plt.close()
    ##################################################################################
    print("ASDFASDFASDF")
    if len(transition_m) < 500:
        visualize_state_space_network(occupancy_key, transition_m, transition_m_name)

    if not os.path.exists(str(os.getcwd()) + '/transition_matrices'):  #test to se if a folder for
        # storing the data exists
        os.makedirs(str(os.getcwd()) + '/transition_matrices')  # if no , create the folder
    np.save(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name), transition_m)
    np.save(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'occupancy_key', occupancy_key)
    np.save(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name) + 'distance_transition_matrix',
            dist_transition_m)


def visualize_state_space_network(occupancy_key, transition_m, transition_m_name):
    labels = {}
    G = nx.DiGraph()
    # my_plot_format(20, 20, 30)
    for node in range(0, len(occupancy_key)):
        labels[node] = "$" + str(int(occupancy_key[node])) + "$"
        G.add_node(1)

    edge_counter = 0
    for i in range(0, len(transition_m)):
        for j in range(0, len(transition_m)):
            if transition_m[i, j] == 0:
                pass
            elif transition_m[i, j] == 1:
                G.add_edge(i, j, rate='$k_{on},k_{off}$')
                edge_counter = edge_counter + 1
            elif transition_m[i, j] == 2:
                G.add_edge(i, j, rate='$k_{on},k_{off}$')
                edge_counter = edge_counter + 1
            elif transition_m[i, j] == 3:
                G.add_edge(i, j, rate='$k_{mb},k_{bm}$')
                edge_counter = edge_counter + 1
            elif transition_m[i, j] == 4:
                G.add_edge(i, j, rate='$k_{mb},k_{bm}$')
                edge_counter = edge_counter + 1
            else:
                pass
    edgelabels = {}
    for u, v, data in G.edges(data=True):
        edgelabels[(u, v)] = data['rate']
    # G=nx.from_numpy_matrix(transition_matrix)
    val_map = {'A': 1.0,
               'D': 0.5714285714285714,
               'H': 0.0}
    values = [val_map.get(node, 0.25) for node in G.nodes()]
    plt.figure(figsize=(len(transition_m), len(transition_m)))
    pos = nx.spring_layout(G, iterations=10000)
    nx.draw(G, pos=pos, cmap=plt.get_cmap('jet'), node_color='white', arrows=True, edge_color='k', node_size=1000)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edgelabels, font_size=16)
    plt.savefig(str(os.getcwd()) + '/' + 'State_Space' + '/' + transition_m_name + '/' + "graph.png")
    plt.savefig(str(os.getcwd()) + '/' + 'State_Space' + '/' + transition_m_name + '/' + "graph.svg")
    plt.close()


def alpha(distance): # the function which determines the distance dependence of k mono to bi,

    if distance >= 0 and distance < 2:
        slope_m = .0553
        intercept_b = -2.082e-17
    elif distance < 2.4:
        slope_m = .210
        intercept_b = -.310
    elif distance < 3.4:
        slope_m = .805
        intercept_b = -1.738
    elif distance < 6.8:
        slope_m = -.007
        intercept_b = 1.0247
    elif distance < 14.28:
        slope_m = .003
        intercept_b = .955
    elif distance < 15.8:
        slope_m = -.010
        intercept_b = 1.138
    elif distance < 16.74:
        slope_m = -.482
        intercept_b = 8.599
    elif distance < 21:
        slope_m = -.109
        intercept_b = 2.358
    else:
        slope_m = 0
        intercept_b = 0
    a = distance*slope_m+intercept_b
    print('alpha:', alpha)
    print('distance:', distance)
    return a


def import_transition_matrix(transition_m_name):
    print('import_transition_matrix')
    transition_m = np.load(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name)+'.npy')
    occupancy_key = np.load(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name)+'occupancy_key.npy')
    dist_transition_m = np.load(str(os.getcwd()) + '/' + 'transition_matrices' + '/' + str(transition_m_name)+'distance_transition_matrix.npy')
    return transition_m, occupancy_key, dist_transition_m


def fetch_and_normalize_data(data_file_path,normalization_constant): #norm_const is experimental
    print('fetch_and_normalize_data')
    with open(str(data_file_path), "r") as dataFile:
        data = dataFile.read()
    actual_run = np.genfromtxt(StringIO(data), delimiter="\n")
    actual_run = actual_run/normalization_constant
    return actual_run


def transition_rates(rate_on, rate_off, rate_mono_to_double, rate_double_to_mono, transition_m, dist_transition_m):
    print('transitionRates')
    transition_rate_m = np.zeros((len(transition_m), len(transition_m)))
    for i in range(0, len(transition_m)):
        for j in range(0, len(transition_m)):
            if transition_m[i, j] == 0:
                pass
            elif transition_m[i, j] == 1:
                transition_rate_m[i, j] = rate_on
            elif transition_m[i, j] == 2:
                transition_rate_m[i, j] = rate_off
            elif transition_m[i, j] == 3:
                transition_rate_m[i, j] = rate_mono_to_double * alpha(dist_transition_m[i, j])
                print('distance dependence modified rate: ', transition_rate_m[i, j])
                print('rate_mono_to_double: ', rate_mono_to_double)
            elif transition_m[i, j] == 4:
                transition_rate_m[i, j] = rate_double_to_mono
            else: pass

    print('\n set of unique distances: ', np.unique(dist_transition_m))
    print('\n set of transition modes: ', np.unique(transition_m))
    print('\n set of unique rates: ', np.unique(transition_rate_m))
    return transition_rate_m


def uniformize(transition_m):
    print("uniformize")
    inf_generator_m = transition_m
    uniformization_r = 0
    for i in range(len(transition_m)):
        inf_generator_m[i, i] = -1*np.sum(transition_m[i, :])
        uniformization_r = max(uniformization_r, (-inf_generator_m[i,i]))

    unif_discr_t_mc = np.identity(len(transition_m)) + inf_generator_m/uniformization_r

    return unif_discr_t_mc, uniformization_r, inf_generator_m


def spr_program(program, state_labels, time_samples, ssc):
    print('SPR_program', 'steadyStateConcentration=', ssc, 'timeSamples=', time_samples)
    if program == 'standard':
        time_samples = 2869
        time_interval = time_samples + 1
        time_points = np.array([0, 84, 384, 475, 775, 866, 1166, 1257, 1557, 1656, 1956, time_samples])
        probability_vectors = np.zeros((time_interval, len(state_labels)))
        probability_vectors[0][0] = 1.000
        concentrations = np.array([0,  # Molar concentration
                                   0.025e-9,
                                   0,
                                   0.05e-9,
                                   0,
                                   0.1e-9,
                                   0,
                                   0.25e-9,
                                   0,
                                   0.5e-9,
                                   0])
    elif program == 'SteadyState':
        time_samples = time_samples
        time_interval = time_samples + 1
        time_points = np.array([0, time_samples])
        probability_vectors = np.zeros((time_interval, len(state_labels)))
        probability_vectors[0][0] = 1.000
        concentrations = np.array([ssc,  # Molar concentration
                                   ssc])

    else:
        print("program broke because wrong program name was inputted")

    return time_samples, time_points, concentrations, probability_vectors, time_interval

def make_contact_sheet(fnames, (ncols, nrows), (photow, photoh), (marl, mart, marr, marb), padding):

    """\
    Make a contact sheet from a group of filenames:

    fnames       A list of names of the image files

    ncols        Number of columns in the contact sheet
    nrows        Number of rows in the contact sheet
    photow       The width of the photo thumbs in pixels
    photoh       The height of the photo thumbs in pixels

    marl         The left margin in pixels
    mart         The top margin in pixels
    marr         The right margin in pixels
    marl         The left margin in pixels

    padding      The padding between images in pixels

    returns a PIL image object.
    """
    # print 'make_contact_sheet'

    # Read in all images and resize appropriately
    imgs = [Image.open(fn).resize((photow, photoh)) for fn in fnames]

    # Calculate the size of the output image, based on the
    #  photo thumb sizes, margins, and padding
    marw = marl + marr
    marh = mart + marb

    padw = (ncols - 1) * padding
    padh = (nrows - 1) * padding
    isize = (ncols * photow + marw + padw, nrows * photoh + marh + padh)

    # Create the new image. The background doesn't have to be white
    white = (255, 255, 255)
    inew = Image.new('RGB', isize, white)

    # Insert each thumb:
    for irow in range(nrows):
        for icol in range(ncols):
            left = marl + icol * (photow + padding)
            right = left + photow
            upper = mart + irow * (photoh + padding)
            lower = upper + photoh
            bbox = (left, upper, right, lower)
            try:
                img = imgs.pop(0)
            except:
                break
            inew.paste(img, bbox)
    return inew


def make_bar_chart_of_states(probabilityVectors,directory_name,transition_matrix_name_):
    print('make_bar_chart_of_states')
    # bar chart of the state probabilities in the final timepoint of the run
    final_ranked_state_probabilities = np.zeros((len(probabilityVectors[-1, :]), 2))
    final_ranked_state_probabilities[:, 0] = final_ranked_state_probabilities[:, 0] + probabilityVectors[-1, :]
    final_ranked_state_probabilities[:, 1] = final_ranked_state_probabilities[:, 1] + np.arange(
        len(final_ranked_state_probabilities)) + 1
    final_ranked_state_probabilities = final_ranked_state_probabilities[np.argsort(
        final_ranked_state_probabilities[:, 0])]
    fig = plt.figure(figsize=(len(final_ranked_state_probabilities[:, 0]), 20))
    ax = fig.add_subplot(111)
    labels = final_ranked_state_probabilities[:, 1][::-1].astype(int)
    ax.set_xticks(np.arange((len(final_ranked_state_probabilities)))[::-1])
    tickrange = np.arange((len(final_ranked_state_probabilities)))[::-1]
    state_truncation_value = len(final_ranked_state_probabilities[:, 0])
    plt.xlim(0,len(final_ranked_state_probabilities[:, 0]))
    plt.ylim(0,1)
    ax.set_xlim([-0.5, state_truncation_value + 1])
    ax.bar(tickrange[0:state_truncation_value], final_ranked_state_probabilities[0:state_truncation_value, 0],
           alpha=1,
           color='black',
           width=1
           )
    fnames = []
    for bar_state in ((range(0, state_truncation_value))):
        # lowerCorner = ax.transData.transform(
        #     (0, 100 * np.max(final_ranked_state_probabilities[:, 0])))
        # upperCorner = ax.transData.transform((bar_state + 1, 0))
        # limits = ax.transData.transform((state_truncation_value * 1.05, state_truncation_value * 1.075))
        # bbox_image = BboxImage(Bbox([[lowerCorner[0] + bar_state * limits[0] / state_truncation_value,
        #                               upperCorner[1]+80],
        #                              [lowerCorner[0] + bar_state * limits[0] / state_truncation_value + 80,
        #                               upperCorner[1] - 0],
        #                              ]),
        #                        norm=None,
        #                        origin=None,
        #                        clip_on=False)
        img = imread(os.getcwd() + '/' + 'state_configurations/' + str(transition_matrix_name_) + '/' + str(
            transition_matrix_name_) + str(labels[bar_state]) + '.png')
        fname = os.getcwd() + '/' + 'state_configurations/' + str(transition_matrix_name_) + '/' + str(
            transition_matrix_name_) + str(labels[bar_state]) + '.png'
        fnames = fnames + [fname]
        # bbox_image.set_data(img)
        # ax.add_artist(bbox_image)
        montage = make_contact_sheet(fnames, (len(final_ranked_state_probabilities[:, 0]),1), (80, 80),
                           (0, 0, 0, 0),
                           0)
    montage.save(str(os.getcwd()) + '/' + str(directory_name) + '/' + "state_montage" + str(
        transition_matrix_name_) + ".png")
    plt.ylabel("state probability at steady state", fontsize=50)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(50)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30)
    fig.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + "partitionSpace" + str(
        transition_matrix_name_) + ".png", bbox_inches='tight')
    fig.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + "partitionSpace" + str(
        transition_matrix_name_) + ".svg", bbox_inches='tight')
    plt.close()
    return 0


def print_comparison_occupancy_curves(timeSamples,occupancy,actual_run,directory_name,run_name,transition_matrix_name_):
    print('print_comparison_occupancy_curves')
    plotms.occupancy_curve()
    plotms.uniform_figure_fonts()
    plt.plot(range(timeSamples), actual_run, label='normalized SPR signal',c='#B0E0E6',linestyle='-',lw=5)
    plt.plot(range(timeSamples + 1), occupancy, label='theor. occupancy', c='#800080', linestyle='-',lw=1)
    plt.rcParams.update(
        {'font.size': 15, 'figure.figsize': (6, 6), 'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
         'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'})
    plt.ylabel("occupancy [AB's/structure]")
    plt.ylim((0, 4.0))
    plt.xlabel("t (sec)")
    plotms.occupancy_curve()
    plt.xlim((0, timeSamples + 1))
    plt.legend(bbox_to_anchor=(0.1, 1.02), loc=3, borderaxespad=0., fontsize=12)
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "occupancy_curve" + str(run_name) + str(time.time()) + ".png")
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "occupancy_curve" + str(run_name) + str(time.time()) + ".svg")
    plt.close()
    return 0


def print_stacked_occupancy_curves(timeSamples, directory_name, run_name,probabilityVectors,occupancy_key,transition_matrix_name_):
    print('print_stacked_occupancy_curves')
    random.seed(a=2)

    plotms.occupancy_curve()

    fig, ax = plt.subplots()
    weighted_probability_vectors = np.zeros((len(occupancy_key),len(probabilityVectors)))

    colors = np.zeros((len(occupancy_key)),np.ndarray)
    for state in range (0,len(occupancy_key)):
        weighted_probability_vectors[state] = probabilityVectors[:,state]*occupancy_key[state]
        colors[state]= cm.get_cmap('Spectral')(random.random())
    ax.stackplot(range(timeSamples + 1), weighted_probability_vectors,colors=colors)

    # for state in range(0,len(occupancy_key)):
        # plt.plot(range(timeSamples + 1), probabilityVectors[:,state]*occupancy_key[state], label='theor. occupancy', linestyle='-',lw=1)
    plt.ylabel("occupancy [AB's/structure]")
    plt.ylim((0, 4.0))
    plt.xlabel("t (sec)")
    plt.xlim((0, timeSamples+1))
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels)#bbox_to_anchor=(0.1, 1.02), loc=3, borderaxespad=0., fontsize=12)

    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "stacked_trajectory" + str(run_name) + str(time.time()) + ".png")
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "stacked_trajectory" + str(run_name) + str(time.time()) + ".svg")
    plt.close()
    return 0


def print_stacked_macro_occupancy_curves(timeSamples, directory_name, run_name,probabilityVectors,occupancy_key,transition_matrix_name_):

    print('print_stacked_macro_occupancy_curves')
    random.seed(a=2)

    # ipdb.set_trace()
    plotms.occupancy_curve()

    fig, ax = plt.subplots()
    weighted_probability_vectors = np.zeros((int(np.max(occupancy_key))+1,len(probabilityVectors)))

    colors = np.zeros((7),np.ndarray)
    for i in range(1,len(colors)+1):
        colors[i-1] = cm.get_cmap('inferno')(1./i)
    for state in range (0,len(occupancy_key)):
        weighted_probability_vectors[int(occupancy_key[state])] = weighted_probability_vectors[int(occupancy_key[state])]+probabilityVectors[:,state]*occupancy_key[state]
        # colors[int(occupancy_key[state])]= cm.get_cmap('Jet')(random.random())
    ax.stackplot(range(timeSamples + 1), weighted_probability_vectors,colors=colors)


    # for state in range(0,len(occupancy_key)):
        # plt.plot(range(timeSamples + 1), probabilityVectors[:,state]*occupancy_key[state], label='theor. occupancy', linestyle='-',lw=1)
    plt.ylabel("occupancy [AB's/structure]")
    plt.ylim((0, 4.0))
    plt.xlabel("t (sec)")
    plt.xlim((0, timeSamples+1))
    plt.tight_layout()
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(bbox_to_anchor=(0.1, 1.02), loc=3, borderaxespad=0., fontsize=12)
    # ax.legend()#bbox_to_anchor=(0.1, 1.02), loc=3, borderaxespad=0., fontsize=12)

    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "macrostacked_trajectory" + str(run_name) + str(time.time()) + ".png")
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "macrostacked_trajectory" + str(run_name) + str(time.time()) + ".svg")
    plt.close()
    return 0


def print_stacked_macroprobability_curves(timeSamples, directory_name, run_name,probabilityVectors,occupancy_key,transition_matrix_name_):
    print('print_stacked_macroprobability_curves')
    random.seed(a=2)

    plotms.occupancy_curve()

    fig, ax = plt.subplots()
    colors = np.zeros((7),np.ndarray)
    macro_probability_vectors = np.zeros((int(np.max(occupancy_key)) + 1, len(probabilityVectors)))
    for i in range(0,len(colors)):
        colors[i] = cm.get_cmap('inferno')(1./(i+1))
    for state in range (0,len(occupancy_key)):
        macro_probability_vectors[int(occupancy_key[state])] = macro_probability_vectors[int(occupancy_key[state])]+probabilityVectors[:,state]
        # weighted_probability_vectors[state] = probabilityVectors[:,state]*occupancy_key[state]
        # colors[state]= cm.get_cmap('Spectral')(random.random())

    ax.stackplot(range(timeSamples + 1), macro_probability_vectors,colors=colors)

    # for state in range(0,len(occupancy_key)):
        # plt.plot(range(timeSamples + 1), probabilityVectors[:,state]*occupancy_key[state], label='theor. occupancy', linestyle='-',lw=1)
    plt.ylabel("p")
    plt.ylim((0, 1.0))
    plt.xlabel("t (sec)")
    plt.xlim((0, timeSamples+1))
    plt.tight_layout()

    # plt.legend(handles=)

    # ax.legend(handles, labels)#bbox_to_anchor=(0.1, 1.02), loc=3, borderaxespad=0., fontsize=12)

    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "stacked_probability" + str(run_name) + str(time.time()) + ".png")
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "stacked_probability" + str(run_name) + str(time.time()) + ".svg")
    plt.close()
    return 0


def print_stacked_probability_curves(timeSamples, directory_name, run_name,probabilityVectors,occupancy_key,transition_matrix_name_):
    print('print_stacked_probability_curves')
    random.seed(a=2)
    # ipdb.set_trace()
    plotms.occupancy_curve()

    fig, ax = plt.subplots()
    colors = np.zeros((len(occupancy_key)),np.ndarray)
    for state in range (0,len(occupancy_key)):
        # weighted_probability_vectors[state] = probabilityVectors[:,state]*occupancy_key[state]
        colors[state]= cm.get_cmap('Spectral')(random.random())
    # ipdb.set_trace()
    ax.stackplot(range(timeSamples + 1), probabilityVectors.T,colors=colors)

    # for state in range(0,len(occupancy_key)):
        # plt.plot(range(timeSamples + 1), probabilityVectors[:,state]*occupancy_key[state], label='theor. occupancy', linestyle='-',lw=1)
    plt.ylabel("p")
    plt.ylim((0, 1.0))
    plt.xlabel("t (sec)")
    plt.xlim((0, timeSamples+1))
    plt.tight_layout()
    # plt.legend(handles=)

    # ax.legend(handles, labels)#bbox_to_anchor=(0.1, 1.02), loc=3, borderaxespad=0., fontsize=12)

    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "stacked_probability" + str(run_name) + str(time.time()) + ".png")
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "stacked_probability" + str(run_name) + str(time.time()) + ".svg")
    plt.close()
    return 0


def print_standalone_occupancy_curves(timeSamples, occupancy, directory_name, run_name,transition_matrix_name_):
    print('print_standalone_occupancy_curves')
    plotms.occupancy_curve()
    plt.plot(range(timeSamples + 1), occupancy, label='theor. occupancy', c='#800080', linestyle='-',lw=1)
    plt.ylabel("occupancy [AB's/structure]")
    plt.ylim((0, 4.0))
    plt.xlabel("t (sec)")
    plt.xlim((0, timeSamples+1))
    plt.legend(bbox_to_anchor=(0.1, 1.02), loc=3, borderaxespad=0., fontsize=12)
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "occupancy_curve" + str(run_name) + str(time.time()) + ".png")
    plt.savefig(str(os.getcwd()) + '/' + str(directory_name) + '/' + str(today) + str(
        transition_matrix_name_) + "occupancy_curve" + str(run_name) + str(time.time()) + ".svg")
    plt.close()
    return 0


def print_occupancy_dissection_curves(directory_name,today,transition_matrix_name_,run_name,probabilityVectors,occupancy_key,stateLabels,timeInterval):
    print('print_occupancy_dissection_curves')
    color = plt.cm.jet(np.linspace(0, 1, 7))
    sumcurve = np.zeros((len(probabilityVectors[:, 0]), int(np.max(occupancy_key) + 1)))
    # sumcurve_occupancy_key =  np.zeros((np.max(occupancy_key)+1))
    print(np.max(occupancy_key))
    for curve in range(0, len(stateLabels)):
        sumcurve[:, int(occupancy_key[curve])] = sumcurve[:, int(occupancy_key[curve])] + probabilityVectors[:, curve]
    for curve in range(0, int(np.max(occupancy_key)) + 1):
        c = color[int(range(0, int(np.max(occupancy_key)) + 1)[curve])]
        plt.plot(range(timeInterval), sumcurve[:, curve], label=str(range(0, int(np.max(occupancy_key)) + 1)[curve])+" AB's/structure",
                 c=c, lw=2)
        plt.ylabel("$p$")
        plt.xlabel("t (sec)")
        plt.legend(bbox_to_anchor=(1.05, 0.95), loc=2, borderaxespad=-0.5, fontsize=12)
    plt.xlim((0, len(probabilityVectors[:, 0])))
    plt.savefig(str(os.getcwd()) + "/" + str(directory_name) + "/" + str(today) + str(transition_matrix_name_) + str(
        run_name) + "state_probabilities" + str(time.time()) + ".png", bbox_inches='tight')
    plt.savefig(str(os.getcwd()) + "/" + str(directory_name) + "/" + str(today) + str(transition_matrix_name_) + str(
        run_name) + "state_probabilities" + str(time.time()) + ".svg", bbox_inches='tight')
    plt.close()
    return 0


def parse_parameters_and_normalize_rundata(fixed_params,run_data,optimization_params):
    print('parse_parameters_and_normalize_rundata')
    actual_run = run_data
    variable_and_fixed_parameters = fixed_params[0]
    index_of_fixed_params = fixed_params[1]
    index_of_variable_params = fixed_params[2]
    run_name = fixed_params[3]
    program = fixed_params[4]
    parameter_values = np.zeros((5))
    for index in index_of_fixed_params:
        parameter_values[index] = variable_and_fixed_parameters[0][index]
    counter = 0
    for index in index_of_variable_params:
        parameter_values[index] = optimization_params[counter]
        counter += 1
    normalization_constant = parameter_values[-1]
    timeSamples = len(actual_run)
    actual_run = actual_run / normalization_constant
    return timeSamples, actual_run,run_name,program

def SPR_run(optimization_params,fixed_params):

    print('SPR_run')
    run_data = run_data_
    transition_matrix = transition_matrix_
    occupancy_key = occupancy_key_
    print_graph = print_graph_
    non_monovalent = non_monovalent_
    stateLabels = stateLabels_
    timeSamples, actual_run, run_name, program = parse_parameters_and_normalize_rundata(fixed_params,run_data,optimization_params)
    timeSamples, time_points, concentrations, probabilityVectors, timeInterval = spr_program(program=program, state_labels=stateLabels,time_samples=timeSamples,ssc=0)
    #initialize the occupancy vector as well
    occupancy = np.zeros((timeInterval))
    probabilityVectors, occupancy = compute_transient_probabilities(time_points, concentrations, [np.exp(parameter_values[0]),parameter_values[1],parameter_values[2],parameter_values[3]],
                                                                    non_monovalent, transition_matrix,
                                                                    probabilityVectors, occupancy, occupancy_key,distance_transition_matrix,
                                                                    deltat=1.0, iteration_depth=30)
    sse = ((occupancy[0:-1]-actual_run)*((np.linspace(1,timeSamples,timeSamples))**0.1))**2
    master_error = np.sum(sse)


    if print_graph == True:
        #print out the stateID's of states with the greatest probability so that we can find images of their configuration
        np.savetxt(str(os.getcwd()) + '/' + (directory_name) + '/' + str(today) + 'probability_vectors' + str(time.time()) + '.txt', probabilityVectors, delimiter=';')

        #bar chart of the state probabilities in the final timepoint of the run
        make_bar_chart_of_states(probabilityVectors, directory_name,transition_matrix_name_)

        np.savetxt(str(os.getcwd())+ '/' + (directory_name) +'/' + str(today)+str(transition_matrix_name_)+'theoretical_occupancy_curve_'+str(run_name)+str(time.time())+'.txt', occupancy, delimiter='\n')
        np.savetxt(str(os.getcwd())+ '/' + (directory_name) +'/' + str(today)+str(transition_matrix_name_)+'parameter_fitting_result'+str(run_name)+str(time.time())+'.txt', np.concatenate([[master_error],[np.exp(parameter_values[0]),parameter_values[1],(parameter_values[2]),(parameter_values[3]),parameter_values[4]]]))
        print occupancy_key
        print_comparison_occupancy_curves(timeSamples, occupancy, actual_run, directory_name, run_name,transition_matrix_name_)
        print_occupancy_dissection_curves(directory_name, today, transition_matrix_name_, run_name, probabilityVectors,
                                          occupancy_key, stateLabels, timeInterval)
        print_stacked_occupancy_curves(timeSamples, occupancy, directory_name, run_name,probabilityVectors,occupancy_key,transition_matrix_name_)

    else: pass
    # print (1-np.absolute((np.exp(optimization_params[1]))))*1E-03, (1-np.absolute((np.exp(optimization_params[2]))))*5.96E-03

    return master_error

def constraint_equation(variable_parameters,otherargs):#enable this function in fmin_slsqp to restrict the Kd value for a monovalent fit
    import ipdb


    if np.min(otherargs[2])!=0: #then kon is fixed and the KD criterion should be ignored
        above_zero_criterion = 1.
    else:
        above_zero_criterion = np.log(variable_parameters[1] / np.exp(variable_parameters[0])) - np.log(1e-14)
    # ipdb.set_trace()
    return above_zero_criterion


def fit_to_experimental_data(variable_and_fixed_parameters,run_data__,transition_matrix__,occupancy_key__,print_graph__,non_monovalent__,transition_matrix_name__,run_name,distance_transition_matrix):
    global run_data_
    global transition_matrix_
    global occupancy_key_
    global print_graph_
    global stateLabels_
    global non_monovalent_
    global transition_matrix_name_
    print('fit_to_experimental_data')
    transition_matrix_name_ = transition_matrix_name__
    run_data_ = run_data__
    transition_matrix_ = transition_matrix__
    occupancy_key_ = occupancy_key__
    print_graph_ = print_graph__
    stateLabels_ = np.arange(len(transition_matrix_))
    non_monovalent_ = non_monovalent__
    fixed_params_array = np.asarray(variable_and_fixed_parameters[0])
    variable_parameters_array = np.asarray(variable_and_fixed_parameters[1])
    variable_parameters_numbers_only = variable_parameters_array[variable_parameters_array>=0]
    index_of_fixed_params = np.where(fixed_params_array>=0)
    index_of_variable_params = np.where(variable_parameters_array>=0)
    variable_parameter_bounds = []
    for variable in index_of_variable_params[0]:
        variable_parameter_bounds = variable_parameter_bounds + [variable_and_fixed_parameters[2][variable]]

    if np.count_nonzero(variable_parameters_numbers_only)>0:
        xopt = \
            opt.fmin_slsqp(SPR_run,
                      variable_parameters_numbers_only,
                      args=([variable_and_fixed_parameters,index_of_fixed_params[0],index_of_variable_params[0],run_name,'standard',distance_transition_matrix],),#note that you need the ,) on the end bc its a tuple
                      #ftol=.0000001,
                      # callback=callbackF,
                      #maxiter=500,
                      acc=0.000000001,
                      full_output=True,
                      epsilon=.01,
                      ieqcons=[constraint_equation],
                      bounds=variable_parameter_bounds,
                      #acc=0.1,
                      # full_output=True,
                      #retall=False
                      )
        print_graph_ = True

        SPR_run(xopt[0], [variable_and_fixed_parameters,index_of_fixed_params[0],index_of_variable_params[0],run_name,'standard',distance_transition_matrix])

        return xopt
    else:
        print_graph_ = True
        timeSamples, actual_run, run_name, program,distance_transition_matrix = parse_parameters_and_normalize_rundata(fixed_params=[variable_and_fixed_parameters,index_of_fixed_params[0],index_of_variable_params[0],run_name,'standard'],
                                                                                            run_data=run_data_,
                                                                                            optimization_params=fixed_params_array)
        single_run_no_fit(fixed_params_array,transition_matrix_,occupancy_key_,print_graph=True,non_monovalent=non_monovalent_,transition_matrix_name__=transition_matrix_name_,program='standard',run_name=run_name,actual_run=actual_run)
        # SPR_run(fixed_params_array,
        #         [variable_and_fixed_parameters, index_of_fixed_params[0], index_of_variable_params[0], run_name])
    return fixed_params_array


def single_run_no_fit(rate_constants, transition_matrix, occupancy_key ,print_graph, non_monovalent,transition_matrix_name__,program,run_name,distance_transition_matrix,actual_run=None,timeSamples=2869,steadyStateConcentration=0.5e-9):
    print 'single_run_no_fit'
    transition_matrix_name_ = transition_matrix_name__

    stateLabels = np.arange(len(transition_matrix))
    #probability vector initialization - this occurs outside the multiconcentration experiment loop


    #probabilityVectors[0][:]+1.0/len(probabilityVectors[0][:])#1.0000  # because everything is starting in state 0 at the beginning of the run

    #initialize the experimental conditions:
    timeSamples, time_points, concentrations, probabilityVectors, timeInterval = spr_program(program=program,state_labels=stateLabels,
                                                                                             time_samples=timeSamples,ssc=steadyStateConcentration) #timeSamples=2869
      # seconds
    # initialize the occupancy vector as well
    occupancy = np.zeros((timeInterval))
    deltat = 1.0
    probabilityVectors, occupancy = compute_transient_probabilities(time_points,concentrations,[np.exp(rate_constants[0]),rate_constants[1],rate_constants[2],rate_constants[3]],non_monovalent,transition_matrix,probabilityVectors,occupancy,occupancy_key,distance_transition_matrix,deltat=1.0,iteration_depth=20)
    if print_graph == True:
        #print out the stateID's of states with the greatest
        # probability so that we can find images of their configuration
        np.savetxt(str(os.getcwd()) + '/' + (directory_name) + '/' + str(today) + 'probability_vectors' + str(
            time.time()) + '.txt', probabilityVectors, delimiter=';')

        # make_bar_chart_of_states(probabilityVectors,directory_name,transition_matrix_name_)


        if actual_run == None:
            print_standalone_occupancy_curves(timeSamples, occupancy, directory_name, run_name,transition_matrix_name_)
        else:
            print_comparison_occupancy_curves(timeSamples, occupancy, actual_run, directory_name, run_name,transition_matrix_name_)
        print_standalone_occupancy_curves(timeSamples, occupancy, directory_name, run_name,transition_matrix_name_)
        print_occupancy_dissection_curves(directory_name, today, transition_matrix_name_, run_name, probabilityVectors,
                                          occupancy_key, stateLabels, timeInterval)
        print_stacked_occupancy_curves(timeSamples, directory_name, run_name, probabilityVectors,
                                       occupancy_key,transition_matrix_name_)
        print_stacked_probability_curves(timeSamples, directory_name, run_name, probabilityVectors,
                                         occupancy_key,transition_matrix_name_)
        print_stacked_macro_occupancy_curves(timeSamples, directory_name, run_name, probabilityVectors,
                                         occupancy_key,transition_matrix_name_)
        print_stacked_macroprobability_curves(timeSamples, directory_name, run_name, probabilityVectors,
                                         occupancy_key,transition_matrix_name_)
    else: pass
    return rate_constants

def compute_transient_probabilities(time_points,concentrations,rate_constants,non_monovalent,transition_matrix,probabilityVectors,occupancy,occupancy_key,distance_transition_matrix,deltat=1.0,iteration_depth=20):
    print('compute_transient_probabilities')
    for sub_run in (range(0, len(time_points) - 1)):
        t_lower_bound = time_points[sub_run]
        t_upper_bound = time_points[sub_run + 1]
        sub_run_concentration = concentrations[sub_run]
        rate_on = (rate_constants[0]) * sub_run_concentration
        rate_off = (rate_constants[1])
        if non_monovalent == True:
            rate_on = ((rate_constants[0])) * sub_run_concentration
            rate_off = ((rate_constants[1]))  # 5.87586768E-04
            rate_mono_to_double = ((rate_constants[2]))  # (1-np.absolute((np.exp(rate_constants[2]))))*1E-01
            rate_double_to_mono = ((rate_constants[3]))
        else:
            rate_mono_to_double = 0
            rate_double_to_mono = 0
        # initialization of the transition rate matrix according to current concentration and resultant rate
        transitionRateMatrix = transition_rates(rate_on, rate_off, rate_mono_to_double, rate_double_to_mono,
                                               transition_matrix,distance_transition_matrix)
        print(np.unique(transitionRateMatrix))
        # conversion of transition rate matrix to discrete time step transfer matrix of probabilities
        infinitessimal_generator_matrix = uniformize(transitionRateMatrix)[2]

        for t_step in range(t_lower_bound, t_upper_bound):
            # generator_product = np.identity(len(infinitessimal_generator_matrix))
            factorial_poisson_step = (1.0)
            iteration_depth = iteration_depth
            main_sum = np.zeros((len(infinitessimal_generator_matrix), len(infinitessimal_generator_matrix)))
            for poisson_step in range(0, iteration_depth):
                if poisson_step == 0:
                    factorial_poisson_step = 1
                    generator_product = np.identity(len(infinitessimal_generator_matrix))
                else:
                    factorial_poisson_step = factorial_poisson_step * poisson_step
                    generator_product = np.dot((infinitessimal_generator_matrix * deltat), generator_product)
                main_sum = main_sum + generator_product / factorial_poisson_step

            # if t_step == 1000:ipdb.set_trace()
            probabilityVectors[t_step + 1] = np.absolute(np.dot(probabilityVectors[t_step], main_sum))
            for state in range(0, len(occupancy_key)):
                occupancy[t_step + 1] = occupancy[t_step + 1] + probabilityVectors[t_step + 1, state] * occupancy_key[
                    state]
    return probabilityVectors, occupancy
