import model_builder
import optimization_visualization
import synapse_position_visualization

def main():
    datasets = []
    # List of body files, list of synapse files, right eye (False) or left eye (True)
    datasets.append((['../raw_connectome/FIB-25/annotations-body.json'],  ['../raw_connectome/FIB-25/annotations-synapse.json'], False, 'FIB-25'))
    datasets.append((['../raw_connectome/FIB-19/body.txt'], ['../raw_connectome/FIB-19/synapse.txt'], False, 'FIB-19'))
    
    # Parse
    dataset_bodies, dataset_synapses = model_builder.parse_datasets_to_model(datasets)
        
    # Optimize
    model_builder.optimize_neuron_positions(dataset_bodies, dataset_synapses, 'output/optimized_cell_positions_2.pickle')

    # Generate graphics
    # optimization_visualization.plot_normal_map_results('output/optimized_cell_positions_benchmark_0.pickle', 'output/optimizer/',
    #                                                    dataset_bodies, dataset_synapses, [datasets[0][3], datasets[1][3]])
    # optimization_visualization.plot_cell_map_results('output/optimized_cell_positions_benchmark_0.pickle', 'output/optimizer/',
    #                                                  dataset_bodies, [datasets[0][3], datasets[1][3]])

    # synapse_position_visualization.plot_synapse_positions('output/optimized_cell_positions_benchmark_0.pickle', 'output/optimizer/',
    #                                                       [dataset_bodies[0]],[dataset_synapses[0]],[datasets[0][3]])


if __name__ == "__main__":
    main()