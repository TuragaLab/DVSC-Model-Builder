# Generic import
import sys, os, math, tempfile, atexit, shutil
import numpy as np
import pickle
from joblib import Parallel, delayed
import hexgrid_reference
import tools
from ipykernel.pickleutil import cell_type
import model_builder


def plot_normal_map_parallel(plot_idx, n_datasets, normal_map_keys, normal_maps,
                             cartesian_micrometers, dataset_name, output_path):
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
    
    # print(matplotlib.get_configdir())
    # print(matplotlib.get_cachedir())
    # print(matplotlib.get_home())
    # print(matplotlib.get_data_path())
    
    dataset_range = [plot_idx-1] if plot_idx > 0 else [p for p in range(0, n_datasets)]
    
        
    light_jet = matplotlib_tools.cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    
    for normal_map_key in normal_map_keys:
        normal_map = normal_maps[normal_map_key]
        offsets = list()
        for dr in dataset_range:
            new_offsets = list(normal_map[dr].keys())
            offsets = offsets + list(set(new_offsets) - set(offsets))
        num_offsets = len(offsets)
        v_data = []
        u_data = []
        mu_data = []
        sigma_data = []
        patches = []
        for i in range(0, num_offsets):
            # Flip the offset to make them target centered instead of source centered (filter style)
            y, x = hexgrid_reference.hex_to_cartesian((-offsets[i][0],-offsets[i][1]))
            for dr in dataset_range:
                mu = 0.0
                sigma = 0.0
                if offsets[i] in normal_map[dr].keys() and not normal_map[dr][offsets[i]] == None:
                    mu_temp, sigma_temp = normal_map[dr][offsets[i]]
                    if mu_temp > mu:
                        mu = mu_temp
                        sigma = sigma_temp
            if mu > 0.0 or sigma > 0.0:
                u_data.append(x*cartesian_micrometers) # Convert to micrometers
                v_data.append(y*cartesian_micrometers) # Convert to micrometers
                mu_data.append(mu)
                sigma_data.append(sigma)
                polygon = mpatches.RegularPolygon((u_data[-1], v_data[-1]), 6,
                                                   radius=cartesian_micrometers/(2.0*math.cos(math.pi/5.0)), orientation=math.pi/2.0, zorder=0)
                patches.append(polygon)
                
        num_valid_offsets = len(mu_data)
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 5))
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        radius = cartesian_micrometers/(2.0*math.cos(math.pi/5.0))
        minr = min(np.min(v_data), np.min(u_data))-radius if num_valid_offsets > 0 else -1
        maxr = max(np.max(v_data), np.max(u_data))+radius if num_valid_offsets > 0 else 1
        ax.set_aspect('equal')
        ax.set_xlim([minr,maxr])
        ax.set_ylim([minr,maxr])
        plt.title(dataset_name + ': ' + normal_map_key[0] + r' $\rightarrow$ ' + normal_map_key[1])
        circle_sizes = [((cartesian_micrometers/2.0*mud/np.max(mu_data)) if (mud > 0.0 or sid > 0.0) else 0.2) for mud,sid in zip(mu_data, sigma_data)]
        scatter = ax.scatter(u_data, v_data, s=circle_sizes, c=sigma_data, cmap=light_jet, zorder=2)
        for i in range(0, num_valid_offsets):
            if mu_data[i] > 0.0 or sigma_data[i] > 0.0:
                ax.annotate('$\mu='+'{:.3f}'.format(mu_data[i])+'$\n$\sigma='+'{:.3f}'.format(sigma_data[i])+'$', (u_data[i],v_data[i]), ha='center', va='center')
        cbar = fig.colorbar(scatter, ax=ax)
        ax.set_ylabel(r'$\Delta v\; [\mu m]$')
        ax.set_xlabel(r'$\Delta u\; [\mu m]$')
        cbar.ax.set_ylabel('$\sigma$ (standard deviation)')
        collection = PatchCollection(patches, alpha=1, zorder=0, edgecolor='gray', facecolor='none')
        ax.add_collection(collection)
        fig.canvas.draw()
        # Calculate radius in pixels :
        rr_pix = (ax.transData.transform(np.vstack([circle_sizes, circle_sizes]).T) -
                  ax.transData.transform(np.vstack([np.zeros(num_valid_offsets), np.zeros(num_valid_offsets)]).T))
        rpix, _ = rr_pix.T
        # Calculate and update size in points:
        size_pt = (2*rpix/fig.dpi*72)**2
        scatter.set_sizes(size_pt)
        plt.gca().invert_yaxis()
        
        plt.savefig(output_path+tools.filename_strip(dataset_name+str(normal_map_key))+'.pdf', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()


def plot_normal_map_results(input_name, output_path, dataset_bodies, dataset_synapses, dataset_names, n_threads=32):
    cartesian_micrometers = 2.7918

    os.makedirs(output_path, exist_ok=True)
    dataset_known_positions, _ = pickle.load(open(input_name, 'rb'))
    
    hex_offsets = hexgrid_reference.hex_area(8)
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
        for i in range(0, len(dataset_known_positions)+1):
            normal_maps = dict()
            norm_maps_updates = parallel(delayed(model_builder.update_normal_maps)(sub_cell_pairs, hex_to_hex_offsets,
                                        [dataset_known_positions[j] for j in (range(0, len(dataset_known_positions)) if i == 0 else [i-1])], None,
                                        [dataset_sorted_synapses[j] for j in (range(0, len(dataset_sorted_synapses)) if i == 0 else [i-1])], count_zero=True) \
                                        for sub_cell_pairs in tools.split(cell_pairs, n_threads))
            for norm_maps_update in norm_maps_updates:
                for update_key in norm_maps_update.keys():
                    if not update_key[1] in normal_maps.keys():
                        normal_maps[update_key[1]] = [None for p in range(0, len(dataset_bodies) if i == 0 else 1)]
                    normal_maps[update_key[1]][update_key[0]] = norm_maps_update[update_key]


            dataset_name = ''
            if i == 0:
                for j in range(0, len(dataset_known_positions)):
                    dataset_name = dataset_name + dataset_names[j]
                    if j < len(dataset_known_positions) - 1:
                        dataset_name = dataset_name + ' + '
            else:
                dataset_name = dataset_names[i - 1]
            
            parallel(delayed(plot_normal_map_parallel)(i, len(dataset_bodies), normal_map_keys,
                                                       normal_maps, cartesian_micrometers, dataset_name, output_path) \
                 for normal_map_keys in tools.split(list(normal_maps.keys()), n_threads))
        
        

def plot_cell_map_parallel(cell_types, dataset_known_positions, hex_offsets, cartesian_micrometers, dataset_bodies, dataset_names, output_path):
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
    
    
    light_jet = matplotlib_tools.cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    radius = cartesian_micrometers/(2.0*math.cos(math.pi/5.0))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    
    minr = 0.0
    maxr = 0.0
    for offset in hex_offsets:
        v, u = hexgrid_reference.hex_to_cartesian(offset)
        v = v * cartesian_micrometers
        u = u * cartesian_micrometers
        minr = min(minr, min(v-radius,u-radius))
        maxr = max(maxr, max(v+radius,u+radius))
                
    for cell_type in cell_types:
        allocated_offsets = [dict() for i in range(0,len(dataset_known_positions))]
        for i in range(0, len(dataset_known_positions)):
            if cell_type in dataset_known_positions[i].keys():
                known_positions = dataset_known_positions[i][cell_type]
                for known_position in known_positions:
                    allocated_offsets[i][known_position[3]] = 1 + (0 if (not known_position[3] in allocated_offsets[i].keys()) else allocated_offsets[i][known_position[3]])
            
        dataset_patches = [[] for i in range(0,len(dataset_known_positions))]
        for i in range(0, len(dataset_known_positions)):
            for offset in allocated_offsets[i].keys():
                # Figure out if this was a pre-annotated body
                body_annotated = False
                for body_key in dataset_bodies[i].keys():
                    body = dataset_bodies[i][body_key]
                    if body[0] == cell_type and body[1] == offset:
                        body_annotated = True
                v, u = hexgrid_reference.hex_to_cartesian(offset)
                v = v * cartesian_micrometers
                u = u * cartesian_micrometers
                polygon = mpatches.RegularPolygon((u, v), 6, radius=radius, orientation=math.pi/2.0, zorder=1, facecolor=light_jet(norm(1 if body_annotated else 0)))
                dataset_patches[i].append(polygon)
            
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(5*2, 5))
        for ax in axs:
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.set_aspect('equal')
            ax.set_xlim([minr, maxr])
            ax.set_ylim([minr, maxr])
            
        for i in range(0, len(axs)):
            for offset in list(allocated_offsets[i].keys()):
                v, u = hexgrid_reference.hex_to_cartesian(offset)
                v = v * cartesian_micrometers
                u = u * cartesian_micrometers
                axs[i].annotate(allocated_offsets[i][offset], (u, v), ha='center', va='center')

        for patches,ax,title in zip(dataset_patches,axs,dataset_names):
            ax.set_ylabel(r'$v\; [\mu m]$')
            ax.set_xlabel(r'$u\; [\mu m]$')
            ax.set_title(title)
            collection = PatchCollection(patches, alpha=1, zorder=1, edgecolor='gray', match_original=True)
            ax.add_collection(collection)
        
        fig.suptitle(cell_type)

        leg = plt.figlegend(['Algorithm', 'Annotated'], loc = 'lower center', ncol=2)
        leg.legendHandles[0].set_color(light_jet(norm(0)))
        leg.legendHandles[0].set_linewidth(5.0)
        leg.legendHandles[1].set_color(light_jet(norm(1)))
        leg.legendHandles[1].set_linewidth(5.0)

        fig.canvas.draw()
        plt.gca().invert_yaxis()       
        plt.savefig(output_path+tools.filename_strip(str(cell_type))+'.pdf', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()
    
            
def plot_cell_map_results(input_name, output_path, dataset_bodies, dataset_names, n_threads=32):
    with Parallel(n_jobs=n_threads) as parallel:
        cartesian_micrometers = 2.7918
    
        os.makedirs(output_path, exist_ok=True)
        dataset_known_positions, _ = pickle.load(open(input_name, 'rb'))
        
        cell_types = set()
        for known_positions in dataset_known_positions:
            for cell_type in list(known_positions.keys()):
                cell_types.add(cell_type)
        
        hex_offsets = hexgrid_reference.hex_area(8)            
        parallel(delayed(plot_cell_map_parallel)(cell_types, dataset_known_positions, hex_offsets, cartesian_micrometers,
                                                 dataset_bodies, dataset_names,output_path) \
                 for cell_types in tools.split(list(cell_types), n_threads))
        
    
