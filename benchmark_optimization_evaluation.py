import model_builder
import sys, os, math
import pickle
import matplotlib
from matplotlib.pyplot import tight_layout
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib_tools

def main():
    
    input_name = 'output/optimized_cell_positions_benchmark_'
    output_name = 'output/optimizer/'
    os.makedirs(output_name, exist_ok=True)
    
    datasets = []
    # List of body files, list of synapse files, right eye (False) or left eye (True)
    datasets.append((['../raw_connectome/FIB-25/annotations-body.json'],  ['../raw_connectome/FIB-25/annotations-synapse.json'], False))
    datasets.append((['../raw_connectome/FIB-19/body.txt'], ['../raw_connectome/FIB-19/synapse.txt'], False))
    # Parse
    dataset_bodies, dataset_synapses = model_builder.parse_datasets_to_model(datasets)
    
    # [dataset][cell_type][(index, id, (y,x))]
    orig_dataset_known_positions = []
    # [dataset][cell_type][(index, id)]
    orig_dataset_unknown_positions = []
    for i in range(0, len(dataset_bodies)):
        index = 0
        sub_known_positions = dict()
        sub_unknown_positions = dict()
        bodies = dataset_bodies[i]
        for body_key in list(bodies.keys()):
            body = bodies[body_key]
            if not body[0] in sub_known_positions:
                sub_known_positions[body[0]] = []
            if not body[0] in sub_unknown_positions:
                sub_unknown_positions[body[0]] = []
            if body[1] == None:
                sub_unknown_positions[body[0]].append((index, body_key))
            else:
                sub_known_positions[body[0]].append((index, body_key, body[1]))
            index = index + 1   
        orig_dataset_known_positions.append(sub_known_positions) 
        orig_dataset_unknown_positions.append(sub_unknown_positions) 


    results = []
    
    for i in range(0,11):
        dataset_known_positions, _ = pickle.load(open(input_name+str(i)+'.pickle', 'rb'))
        results.append(dataset_known_positions)
    
    cell_types = set()
    for known_positions in results[0]:
        for cell_type in list(known_positions.keys()):
            cell_types.add(cell_type)
    
    
    count_correct = [[0 for j in range(0, 1+2*(len(results[0])+1))] for i in range(0, len(results))]
    
    total_position_count = 0
    known_position_count = 0
    dataset_known_position_count = [0 for i in range(0, len(dataset_bodies))]
    dataset_total_position_count = [0 for i in range(0, len(dataset_bodies))]
    for i in range(0, len(dataset_bodies)):
        bodies = dataset_bodies[i]
        for body_key in list(bodies.keys()):
            body = bodies[body_key]
            if not body[1] == None:
                known_position_count += 1
                dataset_known_position_count[i] += 1 
            total_position_count += 1
            dataset_total_position_count[i] += 1
    
    for cell_type in cell_types:
        for i in range(0, len(results[0])):
            if cell_type in results[0][i].keys():
                reference_known_positions = results[0][i][cell_type]
                for reference_known_position in reference_known_positions:
                    count_correct[0][1] += 1
                    count_correct[0][2+i] += 1
                    for j in range(1,11):
                        for compare_known_position in results[j][i][cell_type]:
                            if (compare_known_position == reference_known_position):
                                count_correct[j][1] += 1
                                count_correct[j][2+i] += 1
                                
    for cell_type in cell_types:
        for i in range(0, len(orig_dataset_known_positions)):
            if cell_type in orig_dataset_known_positions[i].keys():
                reference_known_positions = orig_dataset_known_positions[i][cell_type]
                for reference_known_position in reference_known_positions:
                    for j in range(0,11):
                        for compare_known_position in results[j][i][cell_type]:
                            if (compare_known_position == reference_known_position):
                                count_correct[j][len(orig_dataset_known_positions)+2] += 1
                                count_correct[j][len(orig_dataset_known_positions)+3+i] += 1

        
    xlabels = [i*10 for i in range(0,11)]
    for j in reversed(range(0,11)):
        count_correct[j][0] = (known_position_count-(known_position_count/10)*j)/total_position_count * 100.0
        for i in range(1, 1+2*(len(results[0])+1)):
            count_correct[j][i] = count_correct[j][i]/count_correct[0][i] * (known_position_count/total_position_count if i > 3 else 1.0) * 100.0
    
    plt.figure(figsize=(6.2,6.2))
    plt.plot(xlabels, count_correct)
    plt.legend(['Reference positions', 'Combined', 'FIB-25', 'FIB-19', 'Combined (annotated)', 'FIB-25 (annotated)', 'FIB-19 (annotated)'])
    plt.xlabel('Known positions removed [\%]')
    plt.ylabel('Matching positions [\%]')
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.savefig(output_name+'/optimization_benchmark.pdf', tight_layout=True)
    plt.show()
    
if __name__ == "__main__":
    main()