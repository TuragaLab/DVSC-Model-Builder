import pickle, numpy as np, csv
import model_base

def main():
    model_data = pickle.load(open('output/model_FIB25_FIB19.pickle', 'rb'))

    nodes = model_base.node_natural_sort(model_data['nodes'])

    csv_file = open('output/one_column_model.csv', 'w')
    csv_writer = csv.writer(csv_file)


    neuron_list = []
    num_nodes = 0
    for src_node in nodes:
        if not src_node.name == 'CT1G' and not src_node.name == 'CT1L':
            neuron_list.append(src_node.name)
            num_nodes += 1
    
    csv_writer.writerow(['PRE/POST']+neuron_list)

            
    value_matrix = np.zeros((num_nodes, num_nodes))
        
    i = 0
    for src_node in nodes:
        if not src_node.name == 'CT1G' and not src_node.name == 'CT1L':
            row = [src_node.name]
            j = 0
            for tar_node in nodes:
                if not tar_node.name == 'CT1G' and not  tar_node.name == 'CT1L':
                    for edge in model_data['edges']:
                        if src_node.name == edge.src and tar_node.name == edge.tar:
                            for offset in edge.offsets:
                                if offset[0] == (0,0):
                                    value_matrix[i,j] = offset[1]*edge.alpha
                    j += 1
            for j in range(0, num_nodes):
                row.append(value_matrix[i,j])
            csv_writer.writerow(row)
            i += 1


    csv_file.flush()
    csv_file.close()

if __name__ == "__main__":
    main()