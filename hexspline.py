import matplotlib
import hexgrid_reference
import tools
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import scipy
import tensorflow as tf

# Numpy proof-of-concept hexspline implementation
def hexspline(x, y, n):
    sumval = 0.0
    for k1 in range(-n, n + 1):
        for k2 in range(-n, n + 1):
            c = 0.0
            t = 0.0
            for i in range(max(k1,k2,0), min(n+k1,n+k2,n) + 1):
                c = c + (-1)**(k1+k2+i)*scipy.misc.comb(n, i-k1, exact=False)*scipy.misc.comb(n, i-k2, exact=False)*scipy.misc.comb(n,i, exact=False)
            for d in range(0, n):
                t = t + scipy.misc.comb(n-1+d, d, exact=False)*1.0/(math.factorial(2*n-1+d)*math.factorial(n-1-d)) * abs(2.0*y/math.sqrt(3.0)+k1-k2)**(n-1-d) * max(0.0, x-(k1+k2)/2.0-abs(y/math.sqrt(3.0)+(k1-k2)/2.0))**(2*n-1+d)
            sumval = sumval + c * t
    return sumval

# Tensorflow (optimized) hexspline implementation
def tf_hexspline(coords, n, offsets=[(0.0, 0.0)]):
    tf_1_d_sqrt3 = tf.constant(1.0/math.sqrt(3.0))
    tf_2_d_sqrt3 = tf.constant(2.0/math.sqrt(3.0))
    xycoords = tf.unstack(coords, axis=-1)
    spline_components = []
    for offset in offsets:
        yoff_const = tf.constant(offset[0], dtype=tf.float32)
        xoff_const = tf.constant(offset[1], dtype=tf.float32)
        ycoords_offset = (xycoords[0] - yoff_const)
        xcoords_offset = (xycoords[1] - xoff_const)
        partial_sums = []
        for k1 in range(-n, n + 1):
            for k2 in range(-n, n + 1):
                c = 0.0
                for i in range(max(k1,k2,0), min(n+k1,n+k2,n) + 1):
                    c = c + (-1)**(k1+k2+i)*scipy.misc.comb(n, i-k1, exact=False)*scipy.misc.comb(n, i-k2, exact=False)*scipy.misc.comb(n,i, exact=False)
                for d in range(0, n):
                    t = scipy.misc.comb(n-1+d, d, exact=False)*1.0/(math.factorial(2*n-1+d)*math.factorial(n-1-d))
                    partial_sum = tf.scalar_mul(tf.constant(c/n*t, dtype=tf.float32),
                                                tf.pow(tf.abs(tf.scalar_mul(tf_2_d_sqrt3, xcoords_offset)
                                                              + tf.constant(float(k1-k2))),
                                                       tf.constant(float(n-1-d)))
                                                *
                                                tf.pow(tf.nn.relu(ycoords_offset
                                                                  - tf.constant((k1+k2)/2.0)
                                                                  - tf.abs(tf.scalar_mul(tf_1_d_sqrt3, xcoords_offset)
                                                                           + tf.constant((k1-k2)/2.0))),
                                                       tf.constant(float(2*n-1+d))))
                    partial_sums.append(partial_sum)
        spline_components.append(tf.add_n(partial_sums))
    return spline_components

def optimize_neuron_positions(dataset_bodies, dataset_synapses, output_name, iters=100000, debug=False, snapshot=20000,
                              batch_size=32, debug_img_size=100, gamma=0.5, rho=10.0, learning_rate=0.001):
    # Cartesian indices of large hexagonal grid with side length of 6 (91 components)
    hex_offsets = [hexgrid_reference.hex_to_cartesian(offset) for offset in hexgrid_reference.hex_area(5)]
    
    min_cart_y = None
    min_cart_x = None
    max_cart_y = None
    max_cart_x = None
    for offset in hex_offsets:
        if min_cart_y == None or offset[0] < min_cart_y:
            min_cart_y = offset[0]
        if min_cart_x == None or offset[1] < min_cart_x:
            min_cart_x = offset[1]
        if max_cart_y == None or offset[0] > max_cart_y:
            max_cart_y = offset[0]
        if max_cart_x == None or offset[1] > max_cart_x:
            max_cart_x = offset[1]
    max_r = 0.75*max(max(abs(min_cart_y),abs(max_cart_y)),max(abs(min_cart_x),abs(max_cart_x)))
            
    src_count = batch_size
    tar_count = batch_size

    dataset_count = len(dataset_bodies)
    print('Number of distinct datasets:', dataset_count)
    
    sorted_bodies = []
    bodies_yx_vals = []
    for i in range(0, len(dataset_bodies)):
        index = 0
        sub_sorted_bodies = dict()
        bodies = dataset_bodies[i]
        for body_key in list(bodies.keys()):
            body = bodies[body_key]
            if body[0] in sub_sorted_bodies.keys():
                sub_sorted_bodies[body[0]].append((index, body_key, body[1]))
            else:
                sub_sorted_bodies[body[0]] = [(index, body_key, body[1])]
            index = index + 1    
        sorted_bodies.append(sub_sorted_bodies)
        bodies_yx_vals.append(np.zeros((index, 2)))
        for body_key in sub_sorted_bodies.keys():
            for body in sub_sorted_bodies[body_key]:
                r, theta = [random.random()*max_r, 2*math.pi*random.random()]
                y = math.sqrt(r) * math.cos(theta) 
                x = math.sqrt(r) * math.sin(theta)
                bodies_yx_vals[i][body[0], :] = np.asarray([y,x]) if body[2] == None else np.asarray(list(hexgrid_reference.hex_to_cartesian(body[2])))
                
    sorted_synapses = []
    for i in range(0, len(dataset_synapses)):
        sub_sorted_synapses = dict()
        synapses = dataset_synapses[i]
        for synapse_key in list(synapses.keys()):
            for synapse_pair in synapses[synapse_key]:
                sub_sorted_synapses[(synapse_pair[0], synapse_pair[1])] = float(synapse_pair[2])
        sorted_synapses.append(sub_sorted_synapses)

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

    cell_pair_vals = dict()
    for cell_pair in cell_pairs:
        cell_pair_vals[cell_pair] = np.ones((1, len(hex_offsets)), dtype=np.float32)
    
    # Debug
    x = np.linspace(1.2*min_cart_x, 1.2*max_cart_x, int(debug_img_size))
    y = np.linspace(1.2*min_cart_y, 1.2*max_cart_y, int(debug_img_size))
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    img = np.zeros((int(debug_img_size), int(debug_img_size)))    

    plot_obj = None
    if debug:
        plt.ion()
        plot_obj = plt.imshow(img)
        plt.show()
        plt.pause(0.001)
        
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    
    with tf.Session(config=config) as session:
        synapse_vars = tf.get_variable('synapse_var', [tar_count * src_count], dtype=tf.float32, trainable=False)
        prob_vars = tf.get_variable('prob_var', [1, len(hex_offsets)], dtype=tf.float32)
        src_yx_vars = tf.get_variable('src_yx_var', [src_count, 2], dtype=tf.float32)
        src_ids_vars = tf.get_variable('src_id', [src_count], dtype=tf.int32, trainable=False)
        tar_yx_vars = tf.get_variable('tar_yx_var', [tar_count, 2], dtype=tf.float32)
        tar_ids_vars = tf.get_variable('tar_id', [tar_count], dtype=tf.int32, trainable=False)

        print("TF: Position variables")
        stacked_src_list = []
        for i in range(0, tar_count):
            stacked_src_list.append(src_yx_vars)
        
        stacked_tar_list = []
        tar_yx_vars_slices = []
        tar_ids_vars_slices = []
        for i in range(0, tar_count):
            tar_yx_vars_slices.append(tf.slice(tar_yx_vars, [i, 0], [1, 2]))
            tar_ids_vars_slices.append(tf.slice(tar_ids_vars, [i], [1]))

        for i in range(0, tar_count):
            for j in range(0, src_count):
                stacked_tar_list.append(tar_yx_vars_slices[i])
                
        src_yx_vars_slices = []
        src_ids_vars_slices = []
        for i in range(0, src_count):
            src_yx_vars_slices.append(tf.slice(src_yx_vars, [i, 0], [1, 2]))
            src_ids_vars_slices.append(tf.slice(src_ids_vars, [i], [1]))

        
        stacked_src_yx_vars = tf.concat(stacked_src_list, 0)
        stacked_tar_yx_vars = tf.concat(stacked_tar_list, 0)
        
        print("TF: Initializing hexagonal spline grid")
        spline_components = hexspline.tf_hexspline(stacked_tar_yx_vars - stacked_src_yx_vars, 2, offsets=hex_offsets)
         
        spline_vars = tf.matmul(prob_vars, spline_components)
        
        # Penalty term, for when cells get too close to each other
        print("TF: Initializing penalty")
        penalty_terms = []
        enable_terms = []
        for i in range(0, src_count):
            for j in range(i, src_count):
                penalty_terms.append(src_yx_vars_slices[i] - src_yx_vars_slices[j])
                enable_terms.append(tf.not_equal(src_ids_vars_slices[i], src_ids_vars_slices[j]))
        for i in range(0, tar_count):
            for j in range(i, tar_count):
                penalty_terms.append(tar_yx_vars_slices[i] - tar_yx_vars_slices[j])  
                enable_terms.append(tf.not_equal(tar_ids_vars_slices[i], tar_ids_vars_slices[j]))

        penalty = tf.multiply(tf.cast(tf.concat(enable_terms, 0), tf.float32), hexspline.tf_hexspline(tf.multiply(tf.constant(gamma, dtype=tf.float32),tf.concat(penalty_terms, 0)), 2)[0])
        
        
        # Debug
        spline_debug_components = []
        for offset in hex_offsets:
            spline_debug_components.append(hexspline.tf_hexspline(tf.subtract(tf.convert_to_tensor(pos, dtype=tf.float32), tf.constant(list(offset))), 2))
        spline_debug_vars = tf.matmul(prob_vars, tf.reshape(spline_debug_components, shape=(len(hex_offsets),debug_img_size*debug_img_size)))
        spline_debug_img = tf.reshape(spline_debug_vars, shape=(debug_img_size, debug_img_size))
        
                                    
        # Objective function; optimize prob_vars, y_vars, x_vars
        objective_function = tf.nn.l2_loss(tf.subtract(synapse_vars, spline_vars)) + tf.multiply(tf.constant(rho, dtype=np.float32),tf.nn.l2_loss(penalty))

        print("TF: Optimizer")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(objective_function)
        
        batch_cell_pair_vals = np.zeros((1, len(hex_offsets)))
        batch_synapse_vals = np.zeros((src_count * tar_count))
        batch_src_yx_vals = np.zeros((src_count, 2))
        batch_src_ids_vals = np.zeros((src_count), dtype=np.int32)
        batch_tar_yx_vals = np.zeros((tar_count, 2))
        batch_tar_ids_vals = np.zeros((tar_count), dtype=np.int32)

        ph_batch_cell_pair_vals = tf.placeholder(tf.float32, shape=batch_cell_pair_vals.shape)
        ph_batch_synapse_vals = tf.placeholder(tf.float32, shape=batch_synapse_vals.shape)
        ph_batch_src_yx_vals = tf.placeholder(tf.float32, shape=batch_src_yx_vals.shape)
        ph_batch_src_ids_vals = tf.placeholder(tf.int32, shape=batch_src_ids_vals.shape)
        ph_batch_tar_yx_vals = tf.placeholder(tf.float32, shape=batch_tar_yx_vals.shape)
        ph_batch_tar_ids_vals = tf.placeholder(tf.int32, shape=batch_tar_ids_vals.shape)

        assign_prob_vars = prob_vars.assign(ph_batch_cell_pair_vals)
        assign_synapse_vars = synapse_vars.assign(ph_batch_synapse_vals)
        assign_src_yx_vars = src_yx_vars.assign(ph_batch_src_yx_vals)
        assign_src_ids_vars = src_ids_vars.assign(ph_batch_src_ids_vals)
        assign_tar_yx_vars = tar_yx_vars.assign(ph_batch_tar_yx_vals)
        assign_tar_ids_vars = tar_ids_vars.assign(ph_batch_tar_ids_vals)

        
        print("TF: Global variable initialization")
        init = tf.global_variables_initializer()
        session.run(init)
        
        print("TF: Starting at", "obj:", session.run(objective_function))
        
        for step in range(iters + 1):
            if step % snapshot == 0:
                collection = (sorted_bodies, sorted_synapses, dataset_cell_pairs, hex_offsets, bodies_yx_vals, cell_pair_vals)
                pickle.dump(collection, open(output_name + '_' + str(step) + '.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
                
            # Select the dataset
            sel_dataset = np.random.randint(dataset_count)
            # Select the cell pair (which exists in the dataset)
            sel_cell_pair = np.random.randint(len(dataset_cell_pairs[sel_dataset]))
            cell_pair_key = dataset_cell_pairs[sel_dataset][sel_cell_pair]
                     
            print("TF: Loading values", (sel_dataset, cell_pair_key))
                     
            src_idxs = np.random.randint(len(sorted_bodies[sel_dataset][cell_pair_key[0]]), size=src_count)
            tar_idxs = np.random.randint(len(sorted_bodies[sel_dataset][cell_pair_key[1]]), size=tar_count)
            
            
            for i in range(0, len(src_idxs)):
                batch_src_ids_vals[i] = sorted_bodies[sel_dataset][cell_pair_key[0]][src_idxs[i]][0]
                batch_src_yx_vals[i,:] = bodies_yx_vals[sel_dataset][batch_src_ids_vals[i],:]
            
            for i in range(0, len(tar_idxs)):
                batch_tar_ids_vals[i] = sorted_bodies[sel_dataset][cell_pair_key[1]][tar_idxs[i]][0]
                batch_tar_yx_vals[i,:] = bodies_yx_vals[sel_dataset][batch_tar_ids_vals[i],:]
        
            for i in range(0, len(tar_idxs)):
                for j in range(0, len(src_idxs)):
                    src_body_id = sorted_bodies[sel_dataset][cell_pair_key[0]][src_idxs[j]][1]
                    tar_body_id = sorted_bodies[sel_dataset][cell_pair_key[1]][tar_idxs[i]][1]
                    batch_synapse_vals[i*src_count+j] = sorted_synapses[sel_dataset][(src_body_id, tar_body_id)] if (src_body_id, tar_body_id) in sorted_synapses[sel_dataset] else 0.0
            
            # Copy in
            print("TF: Assign variables")
            batch_cell_pair_vals[:] = cell_pair_vals[cell_pair_key]
            session.run(assign_prob_vars, feed_dict={ph_batch_cell_pair_vals: batch_cell_pair_vals})
            session.run(assign_synapse_vars, feed_dict={ph_batch_synapse_vals: batch_synapse_vals})
            session.run(assign_src_yx_vars, feed_dict={ph_batch_src_yx_vals: batch_src_yx_vals})
            session.run(assign_tar_yx_vars, feed_dict={ph_batch_tar_yx_vals: batch_tar_yx_vals})
            session.run(assign_src_ids_vars, feed_dict={ph_batch_src_ids_vals: batch_src_ids_vals})
            session.run(assign_tar_ids_vars, feed_dict={ph_batch_tar_ids_vals: batch_tar_ids_vals})
            
            # Train
            print("TF: Training step")
            session.run(train)
            
            # Copy back: Spline parameters and body (y,x) positions
            print("TF: Store results")
                        
            cell_pair_vals[cell_pair_key] = prob_vars.eval()

            batch_src_yx_vals = src_yx_vars.eval()
            for i in range(0, len(src_idxs)):
                if sorted_bodies[sel_dataset][cell_pair_key[0]][src_idxs[i]][2] == None:
                    bodies_yx_vals[sel_dataset][sorted_bodies[sel_dataset][cell_pair_key[0]][src_idxs[i]][0],:] = batch_src_yx_vals[i,:]
            
            batch_tar_yx_vals = tar_yx_vars.eval()
            for i in range(0, len(tar_idxs)):
                if sorted_bodies[sel_dataset][cell_pair_key[1]][tar_idxs[i]][2] == None:
                    bodies_yx_vals[sel_dataset][sorted_bodies[sel_dataset][cell_pair_key[1]][tar_idxs[i]][0],:] = batch_tar_yx_vals[i,:]
            
                        
            print("TF: Step", step, "obj:", session.run(objective_function))
            
            if debug:
                plot_obj.set_data(spline_debug_img.eval())
                plot_obj.autoscale()
                plt.draw()
                plt.pause(0.001)
            
def plot_optimization_results(input_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    collection = pickle.load(open(input_name, 'rb'))
    sorted_bodies, sorted_synapses, dataset_cell_pairs, hex_offsets, bodies_yx_vals, cell_pair_vals = collection
        
    min_cart_y = None
    min_cart_x = None
    max_cart_y = None
    max_cart_x = None
    for offset in hex_offsets:
        if min_cart_y == None or offset[0] < min_cart_y:
            min_cart_y = offset[0]
        if min_cart_x == None or offset[1] < min_cart_x:
            min_cart_x = offset[1]
        if max_cart_y == None or offset[0] > max_cart_y:
            max_cart_y = offset[0]
        if max_cart_x == None or offset[1] > max_cart_x:
            max_cart_x = offset[1]
        
    img_size = 100
    # Debug
    x = np.linspace(1.2*min_cart_x, 1.2*max_cart_x, int(img_size))
    y = np.linspace(1.2*min_cart_y, 1.2*max_cart_y, int(img_size))
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    img = np.zeros((int(img_size), int(img_size)))    

    plt.ion()
    plot_obj = plt.imshow(img)
    plt.show()
    plt.pause(0.001)
    
    mark_size_plus = 1
    
    for sel_dataset in range(0, len(bodies_yx_vals)):
        body_keys = sorted_bodies[sel_dataset].keys()
        for body_key in body_keys:
            img = np.zeros((int(img_size), int(img_size)))    
            body_sets = sorted_bodies[sel_dataset][body_key]
            for body in body_sets:
                yx = bodies_yx_vals[sel_dataset][body[0],:]
                yv = yx[0]
                xv = yx[1]
                yidx = int(((yv-y[0])/(y[-1]-y[0]))*img_size)
                xidx = int(((xv-x[0])/(x[-1]-x[0]))*img_size)
                for xr in range(xidx-mark_size_plus, xidx+mark_size_plus+1):
                    for yr in range(yidx-mark_size_plus, yidx+mark_size_plus+1):
                        if yr >= 0 and yr < img_size and xr > 0 and xr < img_size:
                            img[yr,xr] = 1
            plot_obj.set_data(img)
            plot_obj.autoscale()
            plt.draw()
            plt.pause(0.01)
            fig = plt.gcf()
            fig.savefig(output_path+tools.filename_strip(str(sel_dataset)+'_'+body_key)+'.eps', format='eps')
            fig.savefig(output_path+tools.filename_strip(str(sel_dataset)+'_'+body_key)+'.png', format='png')
    
    ph_batch_cell_pair_vals = tf.placeholder(tf.float32, shape=(1,len(hex_offsets)))
        
    with tf.Session() as session:
        prob_vars = tf.get_variable('prob_var', [1, len(hex_offsets)], dtype=tf.float32)
        assign_prob_vars = prob_vars.assign(ph_batch_cell_pair_vals)

        spline_components = []
        for offset in hex_offsets:
            spline_components.append(hexspline.tf_hexspline(tf.subtract(tf.convert_to_tensor(pos, dtype=tf.float32), tf.constant(list(offset))), 2))
        spline_vars = tf.matmul(prob_vars, tf.reshape(spline_components, shape=(len(hex_offsets),img_size*img_size)))
        spline_img = tf.reshape(spline_vars, shape=(img_size, img_size))

        for cell_pair_key in cell_pair_vals.keys():
            batch_cell_pair_vals = cell_pair_vals[cell_pair_key]
            session.run(assign_prob_vars, feed_dict={ph_batch_cell_pair_vals: batch_cell_pair_vals})

            plot_obj.set_data(spline_img.eval())
            plot_obj.autoscale()
            plt.draw()
            plt.pause(0.01)
            fig = plt.gcf()
            fig.savefig(output_path+tools.filename_strip(str(cell_pair_key))+'.eps', format='eps')
            fig.savefig(output_path+tools.filename_strip(str(cell_pair_key))+'.png', format='png')


# Demo/testing
def main():
    x = np.linspace(-2., 2., int(1e2))
    y = np.linspace(-2., 2., int(1e2))
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    
    #img = np.zeros((100,100))
    #for xi in range(0,100):
    #    for yi in range(0,100):
    #        img[xi,yi] = boxspline(x[xi],y[yi],2)
    
    with tf.Session() as session:
        vals = tf.convert_to_tensor(pos, dtype=tf.float32)
        img = tf_hexspline(vals, 2)[0].eval()
                
    plt.imshow(np.transpose(img))
    plt.show()

if __name__ == "__main__":
    main()