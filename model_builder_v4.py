# Generic import
import matplotlib
from fileinput import filename
from cv2 import split
import known_synapse_signs
from numpy import source
import tools
from scipy.sparse.linalg.isolve.tests.test_lsqr import normal
from pycparser.c_ast import Assignment
from known_cell_types import cell_remap
import known_neuron_patterns
from dnf.crypto import log_key_import
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os, math
import numpy as np
import re
import csv
import copy
import operator
import random
import subprocess
import json
import tensorflow as tf
import hexspline
import pickle
import scipy
import scipy.stats
import scipy.optimize
from joblib import Parallel, delayed
import time
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


# DVSC imports
import hexgrid_reference
import known_cell_types
import compat_temporal_offsets

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def parse_body_dataset_json(path, bodies, yrotate):
    json_connectome_bodies = None
    
    with open(path) as json_data:
        json_connectome_bodies = json.load(json_data)
        json_data.close()
    
    unknown_bodies = []
    unknown_offsets = []
    
    # Parse bodies
    for json_body in json_connectome_bodies['data']:
        if 'name' in json_body.keys():
            split_name = json_body['name'].replace('-', ' ').split(' ')
            # Translate cell alias to our model cell name
            if (split_name[0] in known_cell_types.cell_alias.keys()):
                split_name[0] = known_cell_types.cell_alias[split_name[0]]
            # Only add known cells to the model
            if (split_name[0] in known_cell_types.cells):
                offset = None
                if (split_name[-1] in hexgrid_reference.offset.keys()):
                    offset = hexgrid_reference.offset[split_name[-1]]
                else:
                    for potential_offset_alias in split_name:
                        if (potential_offset_alias in hexgrid_reference.offset.keys()):
                            offset = hexgrid_reference.offset[potential_offset_alias]
                
                name_modifier = ''
                if (split_name[-1] in known_cell_types.cell_modifiers.keys()):
                    name_modifier = known_cell_types.cell_modifiers[split_name[-1]]
                else:
                    for potential_name_modifier in split_name:
                        if (potential_name_modifier in known_cell_types.cell_modifiers.keys()):
                            name_modifier = known_cell_types.cell_modifiers[potential_name_modifier]

                if not offset == None:
                    if yrotate:
                        offset = hexgrid_reference.yrotate_offset(offset)
                    bodies[json_body['body ID']] = (split_name[0] + name_modifier, offset)
                else:
                    bodies[json_body['body ID']] = (split_name[0] + name_modifier, None)
                    print('Unknown offset; ID: ' + str(json_body['body ID']) + ', full name: ' + json_body['name'])
                    unknown_offsets.append(json_body['name'] + ' (ID: ' + str(json_body['body ID']) + ')')
            else:
                print('Unknown cell type (' + split_name[0] + '); ID: ' + str(json_body['body ID']) + ', full name: ' + json_body['name'])
                unknown_bodies.append(json_body['name'] + ' (ID: ' + str(json_body['body ID']) + ')')
    return unknown_bodies, unknown_offsets

def parse_synapse_dataset_json(path, bodies, synapses, yrotate):
    json_connectome_synapses = None

    with open(path) as json_data:
        json_connectome_synapses = json.load(json_data)
        json_data.close()
        
    # Parse synapses
    for json_synapse in json_connectome_synapses['data']:
        if ('T-bar' in json_synapse.keys() and 'partners' in json_synapse.keys() and len(json_synapse['partners']) > 0):
            for json_partner in json_synapse['partners']:
                if (json_synapse['T-bar']['body ID'] in bodies.keys() and json_partner['body ID'] in bodies.keys()):
                    target_body = bodies[json_partner['body ID']]
                    source_body = bodies[json_synapse['T-bar']['body ID']]
                    
                    synapse_key = (source_body[0], target_body[0])
                    
                    if not synapse_key in synapses:
                        synapses[synapse_key] = []
                    
                    new_synapses = []
                    found = False
                    location = json_synapse['T-bar']['location'] if ('location' in json_synapse['T-bar'].keys()) else None
                    for synapse_pair in synapses[synapse_key]:
                        if synapse_pair[0] == json_synapse['T-bar']['body ID'] and synapse_pair[1] == json_partner['body ID']:
                            locations = synapse_pair[3]
                            locations.append(location)
                            synapse_pair = (synapse_pair[0], synapse_pair[1], synapse_pair[2] + 1, locations)
                            found = True
                        new_synapses.append(synapse_pair)
                    if not found:
                        synapse_pair = (json_synapse['T-bar']['body ID'], json_partner['body ID'], 1, [location])
                        new_synapses.append(synapse_pair)
                    synapses[synapse_key] = new_synapses
            
    
def parse_body_dataset_txt(path, bodies, yrotate):   
    unknown_bodies = []
    unknown_offsets = []
    
    with open(path) as txt_data:
        for txt_line in txt_data:
            split_txt = txt_line.split(' -> ')
            body_id = int(split_txt[0])
            txt_body = split_txt[1].split('\'')[1]
            split_name = txt_body.replace('-', ' ').split(' ')
            # Translate cell alias to our model cell name
            if (split_name[0] in known_cell_types.cell_alias.keys()):
                split_name[0] = known_cell_types.cell_alias[split_name[0]]
            # Only add known cells to the model
            if (split_name[0] in known_cell_types.cells):
                offset = None
                if (split_name[-1] in hexgrid_reference.offset.keys()):
                    offset = hexgrid_reference.offset[split_name[-1]]
                else:
                    for potential_offset_alias in split_name:
                        if (potential_offset_alias in hexgrid_reference.offset.keys()):
                            offset = hexgrid_reference.offset[potential_offset_alias]
                            
                name_modifier = ''
                if (split_name[-1] in known_cell_types.cell_modifiers.keys()):
                    name_modifier = known_cell_types.cell_modifiers[split_name[-1]]
                else:
                    for potential_name_modifier in split_name:
                        if (potential_name_modifier in known_cell_types.cell_modifiers.keys()):
                            name_modifier = known_cell_types.cell_modifiers[potential_name_modifier]
                            
                if not offset == None:
                    if yrotate:
                        offset = hexgrid_reference.yrotate_offset(offset)
                    bodies[body_id] = (split_name[0] + name_modifier, offset)
                else:
                    bodies[body_id] = (split_name[0] + name_modifier, None)
                    print('Unknown offset; ID: ' + str(body_id) + ', full name: ' + txt_body)
                    unknown_offsets.append(txt_body + ' (ID: ' + str(body_id) + ')')
            else:
                print('Unknown cell type (' + split_name[0] + '); ID: ' + str(body_id) + ', full name: ' + txt_body)
                unknown_bodies.append(txt_body + ' (ID: ' + str(body_id) + ')')
    txt_data.close()
    return unknown_bodies, unknown_offsets

def export_bodies_csv(path, bodies):
    with open(path, 'wt') as f:
        writer = csv.writer(f)
        for body_key in bodies.keys():
            body = bodies[body_key]
            row = (body_key, body[0], body[1])
            writer.writerow(row)
        

def parse_synapse_dataset_txt(path, bodies, synapses, yrotate):
    src_body = None
    tar_body = None
    count = None
    with open(path) as txt_data:
        for txt_line in txt_data:
            if 'Cell' in txt_line:
                src_body = int(txt_line.split(' ')[1])
            if 'Out' in txt_line:
                split_line = txt_line.split(' ')
                tar_body = int(split_line[2])
                count = float(split_line[4])
                 
                if (src_body in bodies.keys() and tar_body in bodies.keys()):
                    source_body = bodies[src_body]
                    target_body = bodies[tar_body]
                                       
                    synapse_key = (source_body[0], target_body[0])
                    
                    if not synapse_key in synapses:
                        synapses[synapse_key] = []
                    
                    new_synapses = []
                    found = False
                    for synapse_pair in synapses[synapse_key]:
                        if synapse_pair[0] == src_body and synapse_pair[1] == tar_body:
                            locations = synapse_pair[3]
                            locations.append(None)
                            synapse_pair = (synapse_pair[0], synapse_pair[1], synapse_pair[2] + count, locations)
                            found = True
                        new_synapses.append(synapse_pair)
                    if not found:
                        synapse_pair = (src_body, tar_body, count, [])
                        new_synapses.append(synapse_pair)
                    synapses[synapse_key] = new_synapses

def parse_datasets_to_model(datasets):
    unknown_bodies_file = open('output/unknown_bodies.txt', 'w')
    unknown_offsets_file = open('output/unknown_offsets.txt', 'w')
    
    dataset_bodies = []
    dataset_synapses = []
    dataset_body_offsets = []
    for dataset in datasets:
        synapses = dict()
        bodies = dict()
        body_offsets = dict()

        for path in dataset[0]:
            filename, file_extension = os.path.splitext(path)
            unknown_bodies = None
            unknown_offsets = None
            if (file_extension == '.json'):
                unknown_bodies, unknown_offsets = parse_body_dataset_json(path, bodies, dataset[2])
            else:
                unknown_bodies, unknown_offsets = parse_body_dataset_txt(path, bodies, dataset[2])
            
            for item in sorted(unknown_bodies, key=str.lower):
                unknown_bodies_file.write("%s\n" % item)
            for item in sorted(unknown_offsets, key=str.lower):
                unknown_offsets_file.write("%s\n" % item)
                
        for body_key in bodies.keys():
            body = bodies[body_key]
            body_offset = body[1]
            if body[0] in body_offsets.keys():
                body_offsets[body[0]].append(body_offset)
            else:
                body_offsets[body[0]] = [body_offset]
        
        for path in dataset[1]:
            filename, file_extension = os.path.splitext(path)
            if (file_extension == '.json'):
                parse_synapse_dataset_json(path, bodies, synapses, dataset[2])
            else:
                parse_synapse_dataset_txt(path, bodies, synapses, dataset[2])
           
        dataset_bodies.append(bodies)
        dataset_body_offsets.append(body_offsets)
        dataset_synapses.append(synapses)
        
    unknown_bodies_file.close()
    unknown_offsets_file.close()
    return dataset_bodies, dataset_synapses

# Returns the updated log_likelihood with the normal distribution with a 5% cutoff
def norm_lpdf(dist, x, log_likelihood, min_prob=0.05):
    val = None
    if dist == None:
        # 5% log_likelihood lower cutoff
        val = min_prob
    elif dist[1] > 0.0:
        # Compute normal distribution with variance
        val = scipy.stats.norm(dist[0],dist[1]).pdf(x)
    elif dist[1] == 0.0 and dist[0] - x == 0.0:
        # No variance: Only perfect match gives 100% score
        val = 1.0
    else:
        # log_likelihood lower cutoff
        val = min_prob
        
    val = max(min_prob, val)
        
    if log_likelihood == None:
        return math.log(val)
    else:
        return log_likelihood + math.log(val)

def update_normal_maps(cell_pairs, hex_to_hex_offsets, dataset_known_positions, dataset_sorted_synapses):
    normal_map = dict()
    for i in range(0, len(dataset_known_positions)):
        for cell_pair in cell_pairs:
            normal_map[(i, cell_pair)] = dict()
            for offset in hex_to_hex_offsets:
                data = []
                # Collect data
                sorted_synapses = dataset_sorted_synapses[i]
                if cell_pair[0] in dataset_known_positions[i].keys():
                    for src_body in dataset_known_positions[i][cell_pair[0]]:
                        if cell_pair[1] in dataset_known_positions[i].keys():
                            for tar_body in dataset_known_positions[i][cell_pair[1]]:
                                if (tar_body[3][0] - src_body[3][0], tar_body[3][1] - src_body[3][1]) == offset:
                                    if ((src_body[1], tar_body[1]) in sorted_synapses.keys()):
                                        data.append(sorted_synapses[(src_body[1], tar_body[1])])
                                    else:
                                        data.append(0.0)

                # At least 1 data sample needed to compute mu, std values
                if len(data) > 1:
                    mu, std = scipy.stats.norm.fit(np.asarray(data))
                    normal_map[(i, cell_pair)][offset] = (mu, std)
                else:
                    normal_map[(i, cell_pair)][offset] = None
    return normal_map
    
def update_assignment_picks(i, uk_types, hex_to_hex_offsets, normal_maps, unknown_positions, known_positions,\
                            sorted_synapses, body_synapse_keys, body_remap, available_offsets):
    
    max_num_clusters = 5
    num_known_scom = 0
    for key in known_positions.keys():
        for k in range(0, len(known_positions[key])):
            k_body = known_positions[key][k]
            if not k_body[2] is None:
                num_known_scom += 1
    k_scoms = np.zeros((num_known_scom, max_num_clusters**2, 3))
    k_scoms_mask = np.zeros((num_known_scom, max_num_clusters**2))
    k_offsets = np.zeros((num_known_scom, 2))
    l = 0
    for key in known_positions.keys():
        for k in range(0, len(known_positions[key])):
            k_body = known_positions[key][k]
            if not k_body[2] is None:
                for m in range(0, max_num_clusters):
                    for o in range(0, max_num_clusters):
                        if o < len(k_body[2]):
                            k_scoms[l, m*max_num_clusters+o, :] = np.asarray(k_body[2][o])[1:4]
                            k_scoms_mask[l, m*max_num_clusters+o] = np.asarray(k_body[2][o])[0]
                        else:
                            k_scoms[l, m*max_num_clusters+o, :] = np.nan
                            k_scoms_mask[l, m*max_num_clusters+o] = np.nan
                k_offsets[l, :] = k_body[3]
                l += 1
    
    assignment_picks = []
    for uk_type in uk_types:
        type_available_offsets = available_offsets[uk_type]
        cost_matrix = np.zeros((len(unknown_positions[uk_type]),len(type_available_offsets)))
        if not uk_type in unknown_positions.keys():
            continue
        stime = time.time()
        for j in range(0, len(unknown_positions[uk_type])):
            uk_body = unknown_positions[uk_type][j]
            normalizer = 0.0
            max_log_likelihood = None
            # Calculate probabilties for all free offsets
            log_likelihood_per_offset = [math.log(1.0) for offset in type_available_offsets]
            for syn_key, cell_pair in []:#body_synapse_keys[uk_body[1]]:
                # print(cell_pair)
                # Interaction: The source cell is of the same type as the unknown cell
                count = sorted_synapses[syn_key]
                if uk_body[1] == syn_key[0]:
                    known, k, k_type = body_remap[syn_key[1]]
                    if known:
                        k_body = known_positions[k_type][k]
                        for offset_index in range(0, len(type_available_offsets)):
                            offset = type_available_offsets[offset_index]
                            delta_offset = (offset[0]-k_body[3][0],offset[1]-k_body[3][1])
                            if delta_offset in hex_to_hex_offsets:
                                dist = normal_maps[cell_pair][delta_offset]
                                log_likelihood_per_offset[offset_index] = norm_lpdf(dist, count, log_likelihood_per_offset[offset_index])
                    else:
                        k_body = unknown_positions[k_type][k]
                        for offset_index in range(0, len(type_available_offsets)):
                            log_likelihood_per_offset[offset_index] = norm_lpdf(None, count, log_likelihood_per_offset[offset_index])
            
                # Interaction: The target cell is of the same type as the unknown cell
                elif uk_body[1] == syn_key[1]:
                    known, k, k_type = body_remap[syn_key[0]]
                    if known:
                        k_body = known_positions[k_type][k]
                        for offset_index in range(0, len(type_available_offsets)):
                            offset = type_available_offsets[offset_index]
                            delta_offset = (k_body[3][0]-offset[0],k_body[3][1]-offset[1])
                            if delta_offset in hex_to_hex_offsets:
                                dist = normal_maps[cell_pair][delta_offset]
                                log_likelihood_per_offset[offset_index] = norm_lpdf(dist, count, log_likelihood_per_offset[offset_index])
                    else:
                        k_body = unknown_positions[k_type][k]
                        for offset_index in range(0, len(type_available_offsets)):
                            log_likelihood_per_offset[offset_index] = norm_lpdf(None, count, log_likelihood_per_offset[offset_index])
        
            max_residual = 0.0
            residuals = np.zeros(len(type_available_offsets))
            if not uk_body[2] is None:
                uk_scoms = np.zeros((1, max_num_clusters**2, 3))
                uk_scoms_mask = np.zeros((1, max_num_clusters**2))
                for m in range(0, max_num_clusters):
                    # Broadcast
                    if m < len(uk_body[2]):
                        uk_scoms[0, m*max_num_clusters:(m+1)*(max_num_clusters), :] = np.asarray(uk_body[2][m])[1:4]
                        uk_scoms_mask[0, m*max_num_clusters:(m+1)*(max_num_clusters)] = np.asarray(uk_body[2][m])[0]
                    else:
                        uk_scoms[0, m*max_num_clusters:(m+1)*(max_num_clusters), :] = np.nan
                        uk_scoms_mask[0, m*max_num_clusters:(m+1)*(max_num_clusters)] = np.nan

                delta_offsets = np.zeros((len(type_available_offsets), num_known_scom))  
                # Subtraction becomes [NNNNN..., 1234512345123451234512345, UVW] - [N, 1111122222333334444455555, UVW]
                # which means we get pairwise subtraction of all possible combinations (25)       
                delta_scoms = np.linalg.norm(k_scoms - uk_scoms, axis=2)
                # Only account for deltas where the source and target synapse center of mass stem from the same cluster
                delta_scoms[np.where((k_scoms_mask - uk_scoms_mask) != 0.0)] = np.nan
                # Only count smallest delta, which means the closest synapse clusters
                delta_scoms = np.nanmin(delta_scoms, axis=1)
                
                for offset_index in range(0, len(type_available_offsets)):
                    offset = type_available_offsets[offset_index]
                    delta_offsets[offset_index,:] = np.linalg.norm(np.dot(hexgrid_reference.hex_to_cartesian_R, np.transpose(k_offsets - np.asarray(offset))), axis=0)
                    
                for offset_index in range(0, len(type_available_offsets)):
                    scale = np.nanmean(delta_offsets[offset_index]/delta_scoms)
                    # print(scale)
                    residuals[offset_index] = np.nansum(np.abs(scale * delta_scoms - delta_offsets[offset_index]))
                    if residuals[offset_index] > max_residual:
                        max_residual = residuals[offset_index]
            
            # print(np.min(residuals))
            # print(np.max(residuals))
            
            for offset_index in range(0, len(type_available_offsets)):
                log_likelihood = log_likelihood_per_offset[offset_index]
                if max_residual > 0.0:
                    # 5% cutoff to circumvent math domain errors of log
                    log_likelihood += math.log(max(1.00 - residuals[offset_index]/max_residual, 0.05))

                log_likelihood_per_offset[offset_index] = log_likelihood
                if max_log_likelihood == None or max_log_likelihood < log_likelihood:
                    max_log_likelihood = log_likelihood
            
            probability_per_offset = dict()
            for offset_index in range(0, len(type_available_offsets)):
                probability_per_offset[offset_index] = 0.0
                log_likelihood = log_likelihood_per_offset[offset_index]
                if not log_likelihood == None:
                    # Normalize largest exponent to 0
                    log_likelihood -= max_log_likelihood    
                    normalizer += math.exp(log_likelihood)
                    probability_per_offset[offset_index] = math.exp(log_likelihood)
                    
            for offset_index in range(0, len(type_available_offsets)):
                probability = probability_per_offset[offset_index]
                if not (probability == None or normalizer == 0.0):
                    cost_matrix[j, offset_index] = (1.0 - probability/normalizer)
                else:
                    cost_matrix[j, offset_index] = 1.0
                    
        # print(cost_matrix)
        etime = time.time()
        #print('Time per cell type: ' + str(etime-stime))
                    
        if (len(unknown_positions[uk_type])) > 0:
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            assignment_matrix = np.zeros(cost_matrix.shape)
            assignment_matrix[row_ind, col_ind] = 1.0
            
            cost_assignment_matrix = np.multiply(assignment_matrix, cost_matrix)
            
            cost_per_body = np.sum(cost_assignment_matrix, axis=1)
            pick_j_val = None
            pick_j = None
            offset_index = None
            for j in range(0, len(unknown_positions[uk_type])):
                if pick_j_val == None or pick_j_val > cost_per_body[j]:
                    pick_j_val = cost_per_body[j]
                    pick_j = j
                    offset_index = col_ind[j]

            total_cost = np.sum(cost_per_body)/len(unknown_positions[uk_type])
                        
            for j in range(0, len(unknown_positions[uk_type])):
                assignment_pick = ((1.0/total_cost) if j == pick_j else 0.0, i, uk_type, unknown_positions[uk_type][j][0], unknown_positions[uk_type][j][1], unknown_positions[uk_type][j][2], type_available_offsets[offset_index])
                assignment_picks.append(assignment_pick)
            
    return assignment_picks

def synapse_clusters(locations, min_clusters=1, max_clusters=5, debug=False, epsilon=0.65, compute_cluster_pca=False):
    final_centers = None
    final_labels = None
    final_pcas = None
    
    locs = np.asarray(locations)
    # Single synapse -> synapse location is only cluster location
    if len(locs) == 0:
        return (None, None, None)
    if len(locs) == 1:
        return (locs.tolist(), [0 for i in range(0, len(locs.tolist()))], None)
    distortions = []
    silhouettes = []
    centers = []
    labels = []
    # Between 1 and either 5 or len(locs)-1 clusters
    for n_clusters in range(min_clusters, min(max_clusters+1, len(locs))):
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        labels.append(clusterer.fit_predict(locs))
        #clusterer.fit(locs)
        if n_clusters > 1:
            silhouettes.append(silhouette_score(locs, labels[len(labels)-1]))
        else:
            silhouettes.append(0.0)
        #print(silhouette_avg)
        distortions.append(sum(np.min(cdist(locs, clusterer.cluster_centers_, 'euclidean'), axis=1)) / locs.shape[0])
        centers.append(clusterer.cluster_centers_)
                
    if debug:
        plt.plot(range(min_clusters, max_clusters+1), distortions, 'bx-')
        plt.show()
        plt.pause(1)
        plt.plot(range(min_clusters, max_clusters+1), silhouettes, 'bx-')
        plt.show()
        plt.pause(1)
        
    if np.max(silhouettes) >= epsilon:
        # Suitable cluster count found via silhouettes -> return best scoring #clusters
        final_centers = centers[np.argmax(silhouettes)].tolist()
        final_labels = labels[np.argmax(silhouettes)]
    else:
        # No suitable count found; return smallest #clusters
        final_centers = centers[0].tolist()
        final_labels = labels[0]
        
    if compute_cluster_pca:
        final_pcas = []
        cluster_data = [[] for i in range(0, len(final_centers))]
        for location, label in zip(locations, final_labels):
            cluster_data[label].append(location)
        for cluster in cluster_data:
            pca = PCA(n_components=3)
            pca.fit(cluster)
            final_pcas.append(pca)
        
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(locs[:,0].tolist(), locs[:,1].tolist(), locs[:,2].tolist(), c=final_labels)
        if compute_cluster_pca:
            for pca in final_pcas:
                for length, vector in zip(pca.explained_variance_, pca.components_):
                    v = vector * 3 * np.sqrt(length)
                    v0 = pca.mean_
                    v1 = pca.mean_ + v
                    a = Arrow3D([v0[0], v1[0]], [v0[1], v1[1]], 
                                [v0[2], v1[2]], mutation_scale=20, 
                                lw=3, arrowstyle="-|>", color="r")
                    ax.add_artist(a)
        plt.show()
        plt.pause(1)    
    return (final_centers, final_labels, final_pcas)

def optimize_neuron_positions(dataset_bodies, dataset_synapses, output_name, n_threads=16):
    np.seterr(all='ignore')
    
    hex_offsets = hexgrid_reference.hex_area(6)
    hex_to_hex_offsets = set()
    for hex_offset_1 in hex_offsets:
        for hex_offset_2 in hex_offsets:
            hex_to_hex_offsets.add((hex_offset_2[1]-hex_offset_1[1],hex_offset_2[0]-hex_offset_1[0]))

    # [dataset][cell_type][(index, id, (u,v,w), (y,x))]
    dataset_known_positions = []
    # [dataset][cell_type][(index, id, (u,v,w))]
    dataset_unknown_positions = []
    # [dataset][cell_type][offsets]
    dataset_available_offsets = []
    # [dataset][body_id][synapse_keys]
    dataset_body_synapse_keys = []
    for i in range(0, len(dataset_bodies)):
        sub_known_positions = dict()
        sub_unknown_positions = dict()
        bodies = dataset_bodies[i]
        dataset_available_offsets.append(dict())
        dataset_body_synapse_keys.append(dict())
        
        total_coords = []
        body_scoms = []
        
        for body_key in sorted(list(bodies.keys())):
            body = bodies[body_key]
            coords = []
            # Compute synapse (pre and post) center of mass (position) for the body
            scoms = None
            synapses = dataset_synapses[i]
            for synapse_key in list(synapses.keys()):
                for j in range(0, len(synapses[synapse_key])):
                    if synapses[synapse_key][j][0] == body_key or synapses[synapse_key][j][1] == body_key:
                        for location in synapses[synapse_key][j][3]:
                            coords.append(location)
                        if not body_key in dataset_body_synapse_keys[i]:
                            dataset_body_synapse_keys[i][body_key] = [((synapses[synapse_key][j][0], synapses[synapse_key][j][1]), synapse_key)]
                        else:
                            dataset_body_synapse_keys[i][body_key].append(((synapses[synapse_key][j][0], synapses[synapse_key][j][1]), synapse_key))
                        
            scoms, labels, _ = synapse_clusters(coords, debug=False, compute_cluster_pca=False)
            body_scoms.append(scoms)
            if not scoms is None:
                total_coords += scoms
                
        _, total_labels, _ = synapse_clusters(total_coords, debug=False, compute_cluster_pca=False, epsilon=0.40)
        
        labeled_body_scoms = []
        k = 0
        for scoms in body_scoms:
            if scoms is None:
                labeled_body_scoms.append(None)
            else:
                labeled_scoms = []
                for scom in scoms:
                    labeled_scom = [total_labels[k], scom[0], scom[1], scom[2]]
                    k += 1
                    labeled_scoms.append(labeled_scom)
                labeled_body_scoms.append(labeled_scoms)
        
        index = 0
        for body_key in sorted(list(bodies.keys())):
            body = bodies[body_key]
            
            body_cell_types = [body[0]]
            if body_cell_types[0] in known_cell_types.cell_remap.keys():
                body_cell_types = []
                for body_cell_type in list(known_cell_types.cell_remap.keys()):
                    body_cell_types.append(body_cell_type)
            # Remap cells which have an unsure type to the possible types it could be
            for body_cell_type in body_cell_types:
                if not body_cell_type in dataset_available_offsets[i].keys():
                    dataset_available_offsets[i][body_cell_type] = [offset for offset in hex_offsets]
                if not body_cell_type in sub_known_positions:
                    sub_known_positions[body_cell_type] = []
                if not body_cell_type in sub_unknown_positions:
                    sub_unknown_positions[body_cell_type] = []
                    
                # Only use bodies connected to synapses with at least one position
                if not labeled_body_scoms[index] is None:
                    if body[1] == None or len(body_cell_types) > 1:
                        # Either unknown position or known position but more than one possible cell type
                        sub_unknown_positions[body_cell_type].append((index, body_key, labeled_body_scoms[index]))
                    else:
                        sub_known_positions[body_cell_type].append((index, body_key, labeled_body_scoms[index], body[1]))
                        # Offset no longer available
                        try:
                            dataset_available_offsets[i][body_cell_type].remove(body[1])
                        except:
                            print("Double allocation of offset: ", body[1], body_cell_type)
            index = index + 1   
        dataset_known_positions.append(sub_known_positions) 
        dataset_unknown_positions.append(sub_unknown_positions)
        
    
    dataset_sorted_synapses = []
    for i in range(0, len(dataset_synapses)):
        sorted_synapses = dict()
        synapses = dataset_synapses[i]
        for synapse_key in list(synapses.keys()):
            for synapse_pair in synapses[synapse_key]:
                sorted_synapses[(synapse_pair[0], synapse_pair[1])] = float(synapse_pair[2])
        dataset_sorted_synapses.append(sorted_synapses)
    
    dataset_cell_pairs = []
    cell_pairs = []
    for synapses in dataset_synapses:
        set_cell_pairs = []
        for synapse_key in list(synapses.keys()):
            if not synapse_key in cell_pairs:
                cell_pairs.append(synapse_key)
            if not synapse_key in set_cell_pairs:
                set_cell_pairs.append(synapse_key)
            set_cell_pairs.sort()
        dataset_cell_pairs.append(set_cell_pairs)
    cell_pairs.sort()
    
    dataset_normal_maps = [dict() for i in range(0, len(dataset_bodies))]
    for i in range(0, len(dataset_normal_maps)):
        for cell_pair in cell_pairs:
            dataset_normal_maps[i][cell_pair] = dict()
    
    assignment_picks = []
    num_assignments_left = 1

    while (num_assignments_left > 0):
        with Parallel(n_jobs=n_threads) as parallel:
            # Compute fit based on cells with known positions
            t0 = time.time()
            #norm_maps_updates = parallel(delayed(update_normal_maps)(sub_cell_pairs, hex_to_hex_offsets,\
            #                                                         dataset_known_positions, dataset_sorted_synapses)\
            #                              for sub_cell_pairs in tools.split(cell_pairs, n_threads))
            #for norm_maps_update in norm_maps_updates:
            #    for set_id, update_key in norm_maps_update.keys():
            #        dataset_normal_maps[set_id][update_key] = norm_maps_update[(set_id, update_key)]
                           
            t1 = time.time()
            print('Time', t1 - t0)
            t0 = time.time()
            
            # Loop over all datasets
            dataset_body_remap = [dict() for i in range(0, len(dataset_unknown_positions))]
            for i in range(0, len(dataset_unknown_positions)):
                
                # Create fast lookup map for known/unknown positions based on BodyId
                for cell_type in dataset_known_positions[i].keys():
                    k = 0
                    for body in dataset_known_positions[i][cell_type]:
                        dataset_body_remap[i][body[1]] = (True, k, cell_type)
                        k += 1
                
                for cell_type in dataset_unknown_positions[i].keys():
                    k = 0
                    for body in dataset_unknown_positions[i][cell_type]:
                        dataset_body_remap[i][body[1]] = (False, k, cell_type)
                        k += 1
                        
                # Loop over all cells with unknown positions
                uk_types = []
                for uk_type in dataset_unknown_positions[i].keys():
                    if len(dataset_unknown_positions[i][uk_type]) > 0:
                        uk_types.append(uk_type)
                    
                assignment_picks_updates = parallel(delayed(update_assignment_picks)(i, uk_per_thread_types, hex_to_hex_offsets,\
                                                                                     dataset_normal_maps[i], dataset_unknown_positions[i],\
                                                                                     dataset_known_positions[i], dataset_sorted_synapses[i],
                                                                                     dataset_body_synapse_keys[i], \
                                                                                     dataset_body_remap[i], dataset_available_offsets[i])\
                                                    for uk_per_thread_types in tools.split(uk_types, n_threads))
                for assignment_picks_update in assignment_picks_updates:
                    for assignment_pick in assignment_picks_update:
                        if not assignment_pick == None:
                            found = False
                            for k in range(0, len(assignment_picks)):
                                if assignment_picks[k][4] == assignment_pick[4]:
                                    assignment_picks[k] = assignment_pick
                                    found = True
                            if not found:
                                assignment_picks.append(assignment_pick)
            t1 = time.time()
            print('Time', t1-t0)
           
            assignment_picks.sort(reverse=True, key=lambda tup: tup[0])
            
            # Can update from multiple datasets simultaneously
            for i in range(0, len(dataset_bodies)):
                assignment_pick = None
                for tmp_assignment_pick in assignment_picks:
                    # print(tmp_assignment_pick[1])
                    if i == tmp_assignment_pick[1]:
                        assignment_pick = tmp_assignment_pick
                        # print(assignment_pick)
                        break
                if not (assignment_pick is None):
                    # Body now has a known position
                    dataset_available_offsets[assignment_pick[1]][assignment_pick[2]].remove(assignment_pick[6])
                    dataset_unknown_positions[assignment_pick[1]][assignment_pick[2]].remove((assignment_pick[3], assignment_pick[4], assignment_pick[5]))
                    
                    # Remove other unknown positions carrying the same index and cell ID:
                    for cell_type in list(dataset_unknown_positions[assignment_pick[1]].keys()):
                        try:
                            dataset_unknown_positions[assignment_pick[1]][cell_type].remove((assignment_pick[3], assignment_pick[4], assignment_pick[5]))
                        except:
                            pass
                    dataset_known_positions[assignment_pick[1]][assignment_pick[2]].append((assignment_pick[3], assignment_pick[4], assignment_pick[5], assignment_pick[6]))
                    assignment_picks.remove(assignment_pick)
                    print('Pick:', len(assignment_picks), assignment_pick)
            
            num_assignments_left = len(assignment_picks)
    pickle.dump((dataset_known_positions,  dataset_normal_maps), open(output_name, 'wb'), pickle.HIGHEST_PROTOCOL)
   
    
def get_node_pattern(cell_type, dataset_known_positions):
    allocated_offsets = [dict() for i in range(0, len(dataset_known_positions))]
    for i in range(0, len(dataset_known_positions)):
        known_positions = dataset_known_positions[i]
        for known_position in known_positions:
            allocated_offsets[i][known_position[3]] = 1 + (0 if (not known_position[3] in allocated_offsets[i].keys()) else allocated_offsets[i][known_position[3]])
    # TODO: Find a way to get the real pattern. Data seems insufficient to dedict that at the moment
    # Assume, for now, all cells are synperiodic/columnar
    return ('stride', (1, 1)) if not cell_type in known_neuron_patterns.known_neuron_patterns.keys() else known_neuron_patterns.known_neuron_patterns[cell_type]
    
    
def generate_dvsc_model(template_nodes, template_edges, template_input_units, template_output_units,
                        input_name, dataset_bodies, dataset_synapses,
                        datasets_for_pattern=[0], datasets_for_model=[0],
                        n_threads=32):
    
    # Copy from templates
    nodes = copy.deepcopy(template_nodes) if not template_nodes == None else []
    edges = copy.deepcopy(template_edges) if not template_edges == None else []
    input_units = copy.deepcopy(template_input_units) if not template_input_units == None else []
    output_units = copy.deepcopy(template_output_units) if not template_output_units == None else []

    dataset_known_positions, normal_maps = pickle.load(open(input_name, 'rb'))
    
    cell_patterns = dict()
    cell_types = set()
    for known_positions in dataset_known_positions:
        for cell_type in list(known_positions.keys()):
            if len(known_positions[cell_type]) > 0:
                cell_types.add(cell_type)
   
    for cell_type in list(cell_types):
        cell_patterns[cell_type] = get_node_pattern(cell_type, [dataset_known_positions[i][cell_type] if cell_type in dataset_known_positions[i].keys() else [] for i in datasets_for_pattern])
    
    for known_positions in [dataset_known_positions[i] for i in datasets_for_model]:
        for cell_type in list(known_positions.keys()):
            if len(known_positions[cell_type]) > 0:
                found = False
                for node in nodes:
                    if node[0] == cell_type:
                        found = True
                if not found:
                    nodes.append((cell_type, cell_patterns[cell_type], 'relu', 3.5))
        
    hex_offsets = hexgrid_reference.hex_area(6)
    hex_to_hex_offsets = set()
    for hex_offset_1 in hex_offsets:
        for hex_offset_2 in hex_offsets:
            hex_to_hex_offsets.add((hex_offset_2[1]-hex_offset_1[1],hex_offset_2[0]-hex_offset_1[0]))
            
    dataset_sorted_synapses = []
    for i in range(0, len(dataset_synapses)):
        sorted_synapses = dict()
        synapses = dataset_synapses[i]
        for synapse_key in list(synapses.keys()):
            for synapse_pair in synapses[synapse_key]:
                sorted_synapses[(synapse_pair[0], synapse_pair[1])] = float(synapse_pair[2])
        dataset_sorted_synapses.append(sorted_synapses)
    
    dataset_cell_pairs = []
    cell_pairs = []
    for synapses in dataset_synapses:
        set_cell_pairs = []
        for synapse_key in list(synapses.keys()):
            if not synapse_key in cell_pairs:
                cell_pairs.append(synapse_key)
            if not synapse_key in set_cell_pairs:
                set_cell_pairs.append(synapse_key)
            set_cell_pairs.sort()
        dataset_cell_pairs.append(set_cell_pairs)
    cell_pairs.sort()
    
    with Parallel(n_jobs=n_threads) as parallel:
        normal_maps = dict()
        norm_maps_updates = parallel(delayed(update_normal_maps)(sub_cell_pairs, hex_to_hex_offsets,
                                     [dataset_known_positions[j] for j in datasets_for_model],
                                     [dataset_sorted_synapses[j] for j in datasets_for_model]) \
                                     for sub_cell_pairs in tools.split(cell_pairs, n_threads))
        for norm_maps_update in norm_maps_updates:
            for update_key in norm_maps_update.keys():
                normal_maps[update_key] = norm_maps_update[update_key]
        
    avg_lmbd = 0.0
    avg_count = 0
    temp_edges = []
    for normal_map_key in list(normal_maps.keys()):
        normal_map = normal_maps[normal_map_key]
        cv = 0.0
        count = 0
        edge_offsets = []
        for offset in list(normal_map.keys()):
            if (not normal_map[offset] == None):
                mu, sigma = normal_map[offset]
                cv = cv + sigma/mu
                count = count + 1
                edge_offsets.append((offset, mu))
        
        # Clamp lambda values
        lmbd = None
        if count > 0 and cv/float(count) > 0.0:
            lmbd = max(min(1.0/(cv/float(count)), 1e6), 0.0)
            avg_lmbd = avg_lmbd + lmbd
            avg_count = avg_count + 1

        if count > 0:
            temp_edges.append((normal_map_key[0], normal_map_key[1], edge_offsets,
                               known_synapse_signs.get_sign(normal_map_key[0], normal_map_key[1]),
                               lmbd))
          
    avg_lmbd = avg_lmbd / float(avg_count)
    
    # Rewrite edges, add lambda if necessary (< 3 observations and cv > 0.0)
    for i in range(0, len(temp_edges)):
        temp_edges[i] = (temp_edges[i][0], temp_edges[i][1], temp_edges[i][2],
                         temp_edges[i][3], avg_lmbd if temp_edges[i][4] == None else temp_edges[i][4])
    
    # Rewrite template edges with new alpha and lambda, prune edges with zero weight
    for i in range(0, len(edges)):
        edges[i] = (edges[i][0], edges[i][1], edges[i][2],
                    known_synapse_signs.get_sign(edges[i][0], edges[i][1]),
                    avg_lmbd)
    
    for temp_edge in temp_edges:
        found = False
        for edge in edges:
            if (temp_edge[0] == edge[0] and temp_edge[1] == edge[1]):
                found = True
        if found == False:
            edges.append(temp_edge)
            
    # Prune edges with zero weight
    remove_edges = []
    for i in range(0, len(edges)):
        new_edge_offsets = []
        for edge_offset in edges[i][2]:
            if not edge_offset[1] == 0.0:
                new_edge_offsets.append(edge_offset)
        edges[i] = (edges[i][0], edges[i][1], new_edge_offsets,
                    edges[i][3], edges[i][4])
        if len(new_edge_offsets) == 0:
            remove_edges.append(edges[i])
    # Remove empty edges
    for edge in remove_edges:
        edges.remove(edge)
            
    # Prune nodes without edges from/to it
    remove_nodes = []
    for i in range(0, len(nodes)):
        found = False
        for j in range(0, len(edges)):
            if nodes[i][0] == edges[j][0] or nodes[i][0] == edges[j][1]:
                found = True
                break
        if not found:
            remove_nodes.append(nodes[i])
    # Remove empty nodes
    for node in remove_nodes:
        nodes.remove(node)
    
    return nodes, edges, input_units, output_units
    
# Create the DVSC specification for the new simulation
def write_dvsc_model(output_pickle, output_name, nodes, edges, input_units, output_units):
    if (output_pickle):
        # Pickle serialize
        pickle.dump((nodes, edges, input_units, output_units), open(output_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        # Text serialize to active python code
        file = open(output_name, 'w')
        text = '# Python active code serialized model\n'
        text = text + '################################################################################\n'
        text = text + '# Input units\n'
        text = text + 'input_units = ['
        for i in range(0, len(input_units)):
            text = text + '\'' + input_units[i] + '\''
            if i < len(input_units)-1:
                text = text + ', '
        text = text + ']\n'
        
        text = text + '################################################################################\n'
        text = text + '# Output units\n'
        text = text +'output_units = ['
        for i in range(0, len(output_units)):
            text = text + '\'' + output_units[i] + '\''
            if i < len(output_units)-1:
                text = text + ', '
        text = text + ']\n'
        
        text = text + '################################################################################\n'
        text = text + '# Nodes\n'
        text = text + 'nodes = []\n'
        for node in nodes:
            text = text + 'nodes.append(' + str(node) + ')\n'
        
        text = text + '################################################################################\n'
        text = text + '# Edges\n'
        text = text + 'edges = []\n'
        for edge in edges:
            text = text + 'edges.append(' + str(edge) + ')\n'
            
        file.write(text)
        file.close()
