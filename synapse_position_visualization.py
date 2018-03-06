# Generic import
import sys, os, math, tempfile, atexit, shutil
import numpy as np
import pickle
from joblib import Parallel, delayed
import hexgrid_reference
import tools
import model_builder
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib_tools
import affine_transform


def plot_synapse_positions(input_name, output_path, dataset_bodies, dataset_synapses, dataset_names):
    cartesian_micrometers = 2.7918
    light_jet = matplotlib_tools.cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    radius = cartesian_micrometers/(2.0*math.cos(math.pi/5.0))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    
    os.makedirs(output_path, exist_ok=True)
    dataset_known_positions, _ = pickle.load(open(input_name, 'rb'))
    
    for i in range(0, len(dataset_synapses)):        
        
        synapses = dataset_synapses[i]
        point_data = []
        for synapse_key in list(synapses.keys()):
            for synapse in synapses[synapse_key]:
                for location in synapse[3]:
                    if not location == None and len(location) == 3:
                        point_data.append(location)
        
        point_arr = np.array(point_data)     
        pca = PCA(n_components=3).fit(point_arr)
                        
        cell_types = set()
        known_positions = dataset_known_positions[i]
        for cell_type in list(known_positions.keys()):
            cell_types.add(cell_type)
        
        for cell_type in cell_types:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 6))
            bodies = known_positions[cell_type]
            body_ids = []
            per_cell_points = []
            per_cell_coms = []
            per_cell_positions = []
            
            trans_lhs = []
            trans_rhs = []
            for body in bodies:
                per_body_point_data = []
                for synapse_key in list(synapses.keys()):
                    if synapse_key[0] == cell_type or synapse_key[1] == cell_type:
                        for synapse in synapses[synapse_key]:
                            if synapse[0] == body[1] or synapse[1] == body[1]:
                                for location in synapse[3]:
                                    if not location == None and len(location) == 3:
                                        per_body_point_data.append(location)

                if len(per_body_point_data) > 0 and not body[2] == None:
                    body_ids.append(body[1])
                    per_body_point_arr = np.array(per_body_point_data)
                
                    # Most of the variance is explained along the axis going through the visual system's layers
                    # Therefore we look at the two lesser axis, which show the columnar pattern
                    per_body_point_arr = np.transpose(np.dot(pca.components_[1:3,:], np.transpose(per_body_point_arr)))
                    # print(per_body_point_arr.shape)
                    
                    com = np.average(per_body_point_arr, axis=0)
                    # print(com)
                                        
                    per_cell_points.append(per_body_point_arr)
                    per_cell_coms.append(com)
                    per_cell_positions.append(cartesian_micrometers*np.array(list(hexgrid_reference.hex_to_cartesian(body[2]))))
                    for k in range(0, np.shape(per_body_point_arr)[0]):
                        trans_lhs.append(per_body_point_arr[k,:])
                        trans_rhs.append(per_cell_positions[-1])
            
            trans = affine_transform.Affine_Fit(trans_lhs, trans_rhs)
            trans = affine_transform.Affine_Fit(per_cell_coms, per_cell_positions)

            if trans == False:
                continue
            
            colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(body_ids)))

            bodies_annotated = []
            # patches = []
            for j in range(0, len(body_ids)):
                offset = per_cell_positions[j]
                v, u = (offset[0],offset[1])
                body_annotated = False
                for body_key in dataset_bodies[i].keys():
                    body = dataset_bodies[i][body_key]
                    if body[0] == cell_type and not body[1] == None and np.array_equal(cartesian_micrometers*np.array(list(hexgrid_reference.hex_to_cartesian(body[1]))), offset):
                        body_annotated = True
                polygon = mpatches.RegularPolygon((u, v), 6, radius=radius, orientation=math.pi/2.0, zorder=0, alpha=0.3,
                                                  facecolor=colors[j], edgecolor='black', hatch=('' if body_annotated else  '/'))
                # patches.append(polygon)
                ax.add_patch(polygon)
                bodies_annotated.append(body_annotated)
            
            #collection = PatchCollection(patches, alpha=0.3, zorder=0, edgecolor='black', match_original=True)
            #ax.add_collection(collection)
            
            for j in range(0, len(body_ids)):
                per_cell_points[j] = np.array(trans.Transform(np.transpose(per_cell_points[j])))
                # print(per_cell_points[j])
                # print(per_cell_points[j])
                ax.scatter(per_cell_points[j][1,:], per_cell_points[j][0,:], c=colors[j], s=2, zorder=1)
            
            for j in range(0, len(body_ids)):
                per_cell_coms[j] = trans.Transform(per_cell_coms[j])
                ax.scatter(per_cell_coms[j][1], per_cell_coms[j][0], c=colors[j], s=40, zorder=2)

            legend_labels = []
            for l in (0,1,2):
                for j in range(0, len(body_ids)):
                    suffix = ''
                    if l == 0:
                        suffix = ' (annotated)' if bodies_annotated[j] else ' (algorithm)'
                    if l == 2:
                        suffix = ' (center)' 
                    legend_labels.append(str(body_ids[j]) + suffix)
            plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1.04, 0.5), ncol=int(math.ceil((len(body_ids)*3)/22.0)))
            
            ax.set_ylabel(r'$v\; [\mu m]$')
            ax.set_xlabel(r'$u\; [\mu m]$')
            ax.set_title(dataset_names[i]+': '+cell_type)
            
            fig.canvas.draw()
            plt.gca().invert_yaxis()   
            plt.savefig(output_path+dataset_names[i]+'_'+str(cell_type)+'_synsc.pdf', bbox_inches='tight')
            
            # plt.show()
            # plt.pause(1)
            
            plt.cla()
            plt.clf()
            plt.close()
        
        print(pca.components_)
