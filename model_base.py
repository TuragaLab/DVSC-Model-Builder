import pickle, re

class Node:
    def __init__(self,
                 name=None,
                 pattern=('stride', (1, 1)),
                 activation='relu',
                 bias=3.5,
                 bias_fixed = False,
                 time_constant=None,
                 time_constant_fixed=False):
        self.name = name
        self.pattern = pattern
        self.activation = activation
        self.bias = bias
        self.bias_fixed = bias_fixed
        self.time_constant = time_constant
        self.time_constant_fixed = time_constant_fixed
    
    # Tuple-like accessors for backward compatibility
    def __delitem__(self, key):
        if isinstance(key, int):
            key = self.decode_index(key)
        self.__delattr__(key)
        
    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.decode_index(key)
        return self.__getattribute__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self.decode_index(key)
        self.__setattr__(key, value)
    
    def decode_index(self, index):
        return ['name', 'pattern', 'activation', 'bias', 'time_constant'][index]
    
    def from_dict(self, dict_src):
        for key in dict_src.key():
            if key in self.__dict__.keys():
                self[key] = dict_src[key]
        return self
    
    def from_tuple(self, tpl_src):
        for idx in range(0, len(tpl_src)):
            self[idx] = tpl_src[idx]
        return self
    
class Edge:
    def __init__(self,
                 src=None,
                 tar=None,
                 offsets=[],
                 alpha=1.0,
                 alpha_fixed=False,
                 alpha_references=[],
                 time_constant=None,
                 time_constant_fixed=False,
                 lambda_mult=None,
                 edge_type='chem'):
        self.src = src
        self.tar = tar
        self.offsets = offsets
        self.alpha = alpha
        self.alpha_fixed = alpha_fixed
        self.alpha_references = alpha_references
        self.time_constant = time_constant
        self.time_constant_fixed = time_constant_fixed
        self.lambda_mult = lambda_mult
        self.edge_type = edge_type

    # Tuple-like accessors for backward compatibility
    def __delitem__(self, key):
        if isinstance(key, int):
            key = self.decode_index(key)
        self.__delattr__(key)
        
    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.decode_index(key)
        return self.__getattribute__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self.decode_index(key)
        self.__setattr__(key, value)
        
    def decode_index(self, index):
        return ['src', 'tar', 'offsets', 'alpha', 'lambda_mult', 'time_constant'][index]
    
    def from_dict(self, dict_src):
        for key in dict_src.key():
            if key in self.__dict__.keys():
                self[key] = dict_src[key]
        return self
    
    def from_tuple(self, tpl_src):
        for idx in range(0, len(tpl_src)):
            self[idx] = tpl_src[idx]
        return self
            
class Receptor():
    def __init__(self, name=None, time_constant=None, time_constant_fixed=False):
        self.name = name
        self.time_constant = time_constant
        self.time_constant_fixed = time_constant_fixed

    # Tuple-like accessors for backward compatibility
    def __delitem__(self, key):
        if isinstance(key, int):
            key = self.decode_index(key)
        self.__delattr__(key)
        
    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.decode_index(key)
        return self.__getattribute__(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self.decode_index(key)
        self.__setattr__(key, value)
        
    def decode_index(self, index):
        return ['name', 'time_constant'][index]
    
    def from_dict(self, dict_src):
        for key in dict_src.key():
            if key in self.__dict__.keys():
                self[key] = dict_src[key]
        return self
    
    def from_tuple(self, tpl_src):
        for idx in range(0, len(tpl_src)):
            self[idx] = tpl_src[idx]
        return self
    
def to_string_val(val):
    if isinstance(val, str):
        return '\''+val+'\''
    else:
        return str(val)
    
def serialize(output_name, receptors=[], nodes=[], edges=[], input_units=[], output_units=[], output_pickle=False):
    if output_pickle:
        # Pickle serialize
        data = dict()
        data['nodes'] = nodes
        data['edges'] = edges
        data['receptors'] = receptors
        data['input_units'] = input_units
        data['output_units'] = output_units
        
        pickle.dump(data, open(output_name, 'wb'), pickle.HIGHEST_PROTOCOL)  
    else:
        # Text serialize to active python code
        file = open(output_name, 'w')
        text = '# Python active code serialized model\n'
        text = text + 'import model_base\n'
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
            node_serial = ''
            for key in node.__dict__.keys():
                node_serial += key + '=' + to_string_val(node.__dict__[key]) + ', '
            text = text + 'nodes.append(model_base.Node(' + node_serial + '))\n'
        
        text = text + '################################################################################\n'
        text = text + '# Edges\n'
        text = text + 'edges = []\n'
        for edge in edges:
            edge_serial = ''
            for key in edge.__dict__.keys():
                edge_serial += key + '=' + to_string_val(edge.__dict__[key]) + ', '
            text = text + 'edges.append(model_base.Edge(' + edge_serial + '))\n'
            
        text = text + '################################################################################\n'
        text = text + '# Receptors\n'
        text = text + 'receptors = []\n'
        for receptor in receptors:
            receptor_serial = ''
            for key in receptor.__dict__.keys():
                receptor_serial += key  + '=' + to_string_val(receptor.__dict__[key]) + ', '
            text = text + 'receptors.append(model_base.Receptor(' + receptor_serial + '))\n'
            
        file.write(text)
        file.close()
    
def node_natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key.name) ] 
    return sorted(l, key = alphanum_key)

def edge_natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key.src+key.tar) ] 
    return sorted(l, key = alphanum_key)
    
    