import model_builder
import copy
import random
from joblib import Parallel, delayed

def main():   
    datasets = []
    datasets.append((['../raw_connectome/FIB-25/annotations-body.json'],  ['../raw_connectome/FIB-25/annotations-synapse.json'], False))
    datasets.append((['../raw_connectome/FIB-19/body.txt'], ['../raw_connectome/FIB-19/synapse.txt'], False))
    
    dataset_bodies, dataset_synapses = model_builder.parse_datasets_to_model(datasets)
    
    dataset_bodies_reduced = []
    
    removable_indices = []
    
    total_position_count = 0
    known_position_count = 0
    
    for i in range(0, len(dataset_bodies)):
        bodies = dataset_bodies[i]
        for body_key in list(bodies.keys()):
            body = bodies[body_key]
            if not body[1] == None:
                known_position_count += 1
                removable_indices.append((i, body_key))
            total_position_count += 1
                
    print(known_position_count)
    print(total_position_count)
        
    dataset_bodies_reduced.append(copy.deepcopy(dataset_bodies))
    for i in range(0, 10):
        # Remove 10%
        select = random.sample(range(len(removable_indices)), int(known_position_count/10))
        choices = [removable_indices[j] for j in select]
        print(select)
        for choice in choices:
            body = dataset_bodies[choice[0]][choice[1]]
            dataset_bodies[choice[0]][choice[1]] = (body[0], None)
            print(choice)
            removable_indices.remove(choice)
        dataset_bodies_reduced.append(copy.deepcopy(dataset_bodies))
    
    with Parallel(n_jobs=len(dataset_bodies_reduced)) as parallel:
        parallel(delayed(model_builder.optimize_neuron_positions)(dataset_bodies_reduced[i], dataset_synapses, 'output/optimized_cell_positions_benchmark_'+str(i)+'.pickle')\
                 for i in range(0, len(dataset_bodies_reduced)))
    


if __name__ == "__main__":
    main()