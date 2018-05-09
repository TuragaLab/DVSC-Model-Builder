import model_builder
import optimization_visualization
import synapse_position_visualization
import model_template_v0

def main():
    datasets = []
    # List of body files, list of synapse files, right eye (False) or left eye (True)
    datasets.append((['../raw_connectome/FIB-25/annotations-body.json'],  ['../raw_connectome/FIB-25/annotations-synapse.json'], False, 'FIB-25'))
    datasets.append((['../raw_connectome/FIB-19/body.txt'], ['../raw_connectome/FIB-19/export-fib19.json'], False, 'FIB-19'))
    
    # Parse
    dataset_bodies, dataset_synapses = model_builder.parse_datasets_to_model(datasets)
    
    model_builder.export_bodies_csv('output/bodies-fib19.csv', dataset_bodies[1])
        
    # Optimize
    model_builder.optimize_neuron_positions(dataset_bodies, dataset_synapses, 'output/optimized_cell_positions.pickle')

    # Generate graphics
    #optimization_visualization.plot_normal_map_results('output/optimized_cell_positions.pickle', 'output/optimizer/',
    #                                                   dataset_bodies, dataset_synapses, [datasets[0][3], datasets[1][3]])
    #optimization_visualization.plot_cell_map_results('output/optimized_cell_positions.pickle', 'output/optimizer/',
    #                                                 dataset_bodies, [datasets[0][3], datasets[1][3]])
    synapse_position_visualization.plot_synapse_positions('output/optimized_cell_positions.pickle', 'output/optimizer/',
                                                          dataset_bodies, dataset_synapses, [datasets[0][3], datasets[1][3]])


    # Generate model
    nodes, edges, input_units, output_units = model_builder.generate_dvsc_model(
        model_template_v0.nodes, model_template_v0.edges, model_template_v0.input_units, model_template_v0.output_units,
        'output/optimized_cell_positions.pickle', dataset_bodies, dataset_synapses,
        datasets_for_pattern=[0,1], datasets_for_model=[0,1])

    model_builder.write_dvsc_model(True, 'output/model.pickle', nodes, edges, input_units, output_units)
    model_builder.write_dvsc_model(False, 'output/model.py', nodes, edges, input_units, output_units)

if __name__ == "__main__":
    main()