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
import affine_transform

def list_strip_none(list_in):
    list_out = []
    for entry in list_in:
        if not entry is None:
            list_out.append(entry)
    return list_out

def plot_synapse_positions_parallel(cell_types, i, known_positions, output_path, dataset_bodies,
                                    dataset_synapses, dataset_names):
    mpldir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, mpldir)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(mpldir, 0o777 & ~umask)
    os.environ['HOME'] = mpldir
    os.environ['MPLCONFIGDIR'] = mpldir
    import matplotlib
    # We need to give matplotlib a kick, otherwise it will not work with multiple processes
    # This will modify the TexManager to have a different cache path per thread, to avoid
    # Lock timeouts
    class TexManager(matplotlib.texmanager.TexManager):
        texcache = os.path.join(mpldir, 'tex.cache')
    matplotlib.texmanager.TexManager = TexManager
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    import matplotlib_tools
    
    cartesian_micrometers = 2.7918
    light_jet = matplotlib_tools.cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    radius = cartesian_micrometers/(2.0*math.cos(math.pi/5.0))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    synapses = dataset_synapses[i] 

    for cell_type in cell_types:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 6))
        bodies = known_positions[cell_type]
        body_ids = []
        
        #trans_lhs = []
        #trans_rhs = []
        per_cell_type_point_data = []
        for body in bodies:
            for synapse_key in list(synapses.keys()):
                if synapse_key[0] == cell_type or synapse_key[1] == cell_type:
                    for synapse in synapses[synapse_key]:
                        if synapse[0] == body[1] or synapse[1] == body[1]:
                            for location in synapse[3]:
                                if not location == None and len(location) == 3:
                                    per_cell_type_point_data.append(location)
        scoms, labels, _ = model_builder.synapse_clusters(np.asarray(per_cell_type_point_data), debug=False, compute_cluster_pca=False, epsilon=0.65)
        
        if scoms == None:
            continue

        per_cell_points = [[] for k in range(0, len(scoms))]
        per_cell_coms = [[] for k in range(0, len(scoms))]
        trans = [None for k in range(0, len(scoms))]
        per_cell_positions = [[] for k in range(0, len(scoms))]
        per_cell_positions_collected = []
        
        lidx = 0
        for body in bodies:
            per_body_point_data = [[] for k in range(0, len(scoms))]
            for synapse_key in list(synapses.keys()):
                if synapse_key[0] == cell_type or synapse_key[1] == cell_type:
                    for synapse in synapses[synapse_key]:
                        if synapse[0] == body[1] or synapse[1] == body[1]:
                            for location in synapse[3]:
                                if not location == None and len(location) == 3:
                                    per_body_point_data[labels[lidx]].append(location)
                                    lidx += 1

            if len(per_body_point_data) > 0 and not body[3] == None:
                body_ids.append(body[1])
                for k in range(0, len(scoms)):
                    per_body_point_arr = np.array(per_body_point_data[k])
                    com = np.average(per_body_point_arr, axis=0)
                    if np.count_nonzero(~np.isnan(com)) > 0:
                        per_cell_points[k].append(per_body_point_arr)
                        per_cell_coms[k].append(com.tolist())
                        per_cell_positions[k].append(cartesian_micrometers*np.array(list(hexgrid_reference.hex_to_cartesian(body[3]))+[0]))
                    else:
                        per_cell_points[k].append(None)
                        per_cell_coms[k].append(None)
                        per_cell_positions[k].append(None)
                per_cell_positions_collected.append(cartesian_micrometers*np.array(list(hexgrid_reference.hex_to_cartesian(body[3]))+[0]))

        for k in range(0, len(scoms)):
            if (len(list_strip_none(per_cell_coms[k])) > 2):
                trans[k] = affine_transform.Affine_Fit(list_strip_none(per_cell_coms[k]), list_strip_none(per_cell_positions[k]))
        
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(body_ids)))

        bodies_annotated = []
        # patches = []
        for j in range(0, len(body_ids)):
            offset = per_cell_positions_collected[j]
            v, u = (offset[0],offset[1])
            body_annotated = False
            for body_key in dataset_bodies[i].keys():
                body = dataset_bodies[i][body_key]
                if body[0] == cell_type and not body[1] == None and np.array_equal(cartesian_micrometers*np.array(list(hexgrid_reference.hex_to_cartesian(body[1]))+[0]),
                                                                                   offset):
                    body_annotated = True
            polygon = mpatches.RegularPolygon((u, v), 6, radius=radius, orientation=math.pi/2.0, zorder=0, alpha=0.3,
                                              facecolor=colors[j], edgecolor='black', hatch=('' if body_annotated else  '/'))
            # patches.append(polygon)
            ax.add_patch(polygon)
            bodies_annotated.append(body_annotated)
        
        for j in range(0, len(body_ids)):
            per_cell_points_concat = []
            for k in range(0, len(scoms)):
                if not trans[k] is None and not trans[k] == False and not per_cell_coms[k][j] is None and len(per_cell_coms[k][j]) > 0:
                    per_cell_points[k][j] = np.transpose(np.array(trans[k].Transform(np.transpose(per_cell_points[k][j]))))
                    per_cell_points_concat += per_cell_points[k][j].tolist()
            if (len(per_cell_points_concat) > 0):
                per_cell_points_concat = np.array(per_cell_points_concat)
                ax.scatter(per_cell_points_concat[:,1], per_cell_points_concat[:,0], c=colors[j], s=2, zorder=1)
        
        for j in range(0, len(body_ids)):
            count = 0
            per_cell_coms_avg = [0.0, 0.0]
            for k in range(0, len(scoms)):
                if not trans[k] is None and not trans[k] == False and not per_cell_coms[k][j] is None and len(per_cell_coms[k][j]) > 0:
                    per_cell_coms[k][j] = trans[k].Transform(per_cell_coms[k][j])
                    per_cell_coms_avg[0] += per_cell_coms[k][j][0]
                    per_cell_coms_avg[1] += per_cell_coms[k][j][1]
                    count += 1
            count = max(count, 1)
            ax.scatter(per_cell_coms_avg[1]/count, per_cell_coms_avg[0]/count, c=colors[j], s=40, zorder=2)

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
        name = output_path + tools.filename_strip(dataset_names[i]+'_'+str(cell_type))+'_synsc.pdf'
        print(name)
        plt.savefig(name, bbox_inches='tight')
        
        # plt.show()
        # plt.pause(1)
        
        plt.cla()
        plt.clf()
        plt.close()


def plot_synapse_positions(input_name, output_path, dataset_bodies, dataset_synapses, dataset_names, n_threads=16):
    os.makedirs(output_path, exist_ok=True)
    dataset_known_positions, _ = pickle.load(open(input_name, 'rb'))
    
    with Parallel(n_jobs=n_threads) as parallel:
        for i in range(0, len(dataset_bodies)):                           
            cell_types = set()
            known_positions = dataset_known_positions[i]
            for cell_type in list(known_positions.keys()):
                cell_types.add(cell_type)
        
            parallel(delayed(plot_synapse_positions_parallel)(sub_cell_types, i, known_positions,
                                                              output_path, dataset_bodies,
                                                              dataset_synapses, dataset_names) \
                     for sub_cell_types in tools.split(list(cell_types), n_threads))
            
        
