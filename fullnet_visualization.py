import numpy as np
import random
import tools
import subprocess
import matplotlib
import csv
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib_tools

def generate_fullnet_visualization(file_name, nodes, edges,
                                   weight_files, bias_map_file, weight_map_file, draw_graph=True, draw_table=True):
    
    # Reproducable results
    random.seed(0)
    
    normalizer = 1.0
    
    available_cells = []
    for node in nodes:
        available_cells.append(node[0]);
    
    cell_layers = []
    cell_layers.append(sorted(['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']))
    cell_layers.append(sorted(['L1', 'L2', 'L3', 'L4', 'L5', 'C2', 'C3', 'Lawf2', 'Am']))
    cell_layers.append(sorted(['Mi1', 'Mi4', 'Mi9', 'Mi15', 'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm6', 'Tm9', 'Tm20', 'TmY5a', 'Dm2', 'Dm8', 'T1', 'T2', 'T3', 'T2a']))
    cell_layers.append(sorted(['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']))
    #cell_layers.append(sorted(['']))
    
    cell_layer_names = ['Retina', 'Lamina', 'Medulla', 'Lobula Plate']#, 'Lobula']
    
    cell_layers_filtered = []
    for cell_layer in cell_layers:
        cell_layer_filtered = []
        for cell in cell_layer:
            if cell in available_cells:
                cell_layer_filtered.append(cell)
        cell_layers_filtered.append(cell_layer_filtered)
    cell_layer = cell_layers_filtered
    
    min_neuron_val = None
    max_neuron_val = None
    possible_neuron_vals = []
        
    min_synapse_val = None
    max_synapse_val = None
    possible_synapse_vals = []
    
    neuron_values = dict()
    synapse_values = dict()
    for node in nodes:
        neuron_values[node[0]] = 0.0
    for edge in edges:
        synapse_values[(edge[0], edge[1])] = 0.0
    
    weight_row_list = []
    
    weight_map = dict()
    bias_map = dict()
                
    with open(weight_map_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            weight_map[int(row[2])] = (row[1], row[0])
            
    with open(bias_map_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            bias_map[int(row[1])] = row[0]
    
    
    for fid in range(0, len(weight_files)):
        weight_file = weight_files[fid]
        weight_rows = []
        with open(weight_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                weight_rows.append(row)
        weight_row_list.append(weight_rows)
       
    if (len(weight_row_list) == 1):
        for wid in range(0, len(weight_row_list[0])):
            curr_val = float(weight_row_list[0][wid][2])
            if int(weight_row_list[0][wid][0]) == 0:
                neuron_pair_tag = weight_map[int(weight_row_list[0][wid][1])]
                synapse_values[neuron_pair_tag] = synapse_values[neuron_pair_tag] + abs(curr_val)*normalizer
                neuron_values[neuron_pair_tag[1]] = neuron_values[neuron_pair_tag[1]] + abs(curr_val)*normalizer
            # Do not report BIAS
            # else:
            #    neuron_tag = bias_map[int(weight_row_list[0][wid][1])]
            #    neuron_values[neuron_tag] = neuron_values[neuron_tag] + abs(curr_val)*normalizer
       
    elif (len(weight_row_list) == 2):
        neuron_curr_values = dict()
        neuron_last_values = dict()
        synapse_curr_values = dict()
        synapse_last_values = dict()
        for node in nodes:
            neuron_curr_values[node[0]] = []
            neuron_last_values[node[0]] = []
        for edge in edges:
            synapse_curr_values[(edge[0], edge[1])] = []
            synapse_last_values[(edge[0], edge[1])] = []
        for wid in range(0, len(weight_row_list[0])):
            curr_val = float(weight_row_list[1][wid][2])
            last_val = float(weight_row_list[0][wid][2])
            if int(weight_row_list[0][wid][0]) == 0:
                neuron_pair_tag = weight_map[int(weight_row_list[0][wid][1])]
                synapse_curr_values[neuron_pair_tag].append(curr_val)
                neuron_curr_values[neuron_pair_tag[1]].append(curr_val)
                synapse_last_values[neuron_pair_tag].append(last_val)
                neuron_last_values[neuron_pair_tag[1]].append(last_val)

            # Do not report BIAS
            # else:
            #    neuron_tag = bias_map[int(weight_row_list[0][wid][1])]
            #    neuron_values[neuron_tag] = neuron_values[neuron_tag] + abs(curr_val)
            #    neuron_norm_values[neuron_tag] = neuron_norm_values[neuron_tag] + abs(norm_val)
        for neuron_tag in neuron_values.keys():
            # neuron_values[neuron_tag], _ = scipy.stats.pearsonr(np.asarray(neuron_curr_values[neuron_tag]), np.asarray(neuron_last_values[neuron_tag]));
            neuron_values[neuron_tag] = np.dot(np.asarray(neuron_curr_values[neuron_tag]),np.asarray(neuron_last_values[neuron_tag]))/(np.sqrt(np.dot(np.asarray(neuron_curr_values[neuron_tag]),np.asarray(neuron_curr_values[neuron_tag]))*np.dot(np.asarray(neuron_last_values[neuron_tag]),np.asarray(neuron_last_values[neuron_tag]))))
            if np.isnan(neuron_values[neuron_tag]):
                neuron_values[neuron_tag] = 1.0
        for synapse_tag in synapse_values.keys():
            synapse_values[synapse_tag] = np.dot(np.asarray(synapse_curr_values[synapse_tag]),np.asarray(synapse_last_values[synapse_tag]))/(np.sqrt(np.dot(np.asarray(synapse_curr_values[synapse_tag]),np.asarray(synapse_curr_values[synapse_tag]))*np.dot(np.asarray(synapse_last_values[synapse_tag]),np.asarray(synapse_last_values[synapse_tag]))))
            if np.isnan(synapse_values[synapse_tag]):
                synapse_values[synapse_tag] = 1.0
                
    for edge_tag in synapse_values.keys():
        curr_val = synapse_values[edge_tag]
        possible_synapse_vals.append(curr_val)
                   
    for neuron_tag in neuron_values.keys():
        curr_val = neuron_values[neuron_tag]
        possible_neuron_vals.append(curr_val)
                
    for curr_val in possible_synapse_vals:
        if min_synapse_val == None:
            min_synapse_val = curr_val
        else:
            if curr_val < min_synapse_val:
                min_synapse_val = curr_val
        if max_synapse_val == None:
            max_synapse_val = curr_val
        else:
            if curr_val > max_synapse_val:
                max_synapse_val = curr_val
                                
    for curr_val in possible_neuron_vals:
        if min_neuron_val == None:
            min_neuron_val = curr_val
        else:
            if curr_val < min_neuron_val:
                min_neuron_val = curr_val
        if max_neuron_val == None:
            max_neuron_val = curr_val
        else:
            if curr_val > max_neuron_val:
                max_neuron_val = curr_val
    
    file = open(file_name+'.tex', 'w')
    overlay_text = ''
    text = ''
    #text = text + '\\documentclass[letterpaper]{article}\n'
    text = text + '\\RequirePackage{luatex85}\n'
    text = text + '\\documentclass[border=3pt]{standalone}\n'
    text = text + '\\usepackage[pass]{geometry}\n'
    text = text + '\\usepackage[table]{xcolor}\n'
    text = text + '\\usepackage{tikz}\n'
    text = text + '\\usepackage{float}\n'
    text = text + '\\usepackage[utf8]{inputenc}\n'
    text = text + '\\usepackage{subfig}\n'
    text = text + '\\usepackage[T1]{fontenc}\n'
    text = text + '\\usepackage{multirow}\n'
    text = text + '\\usetikzlibrary{arrows,positioning,shapes,decorations,calc,fit}\n'
    text = text + '\\pgfdeclarelayer{bg}\n'
    text = text + '\\pgfsetlayers{bg,main}\n'
    text = text + '\\pgfdeclaredecoration{dynamic rounded corners}{initial}{\n'
    text = text + '    \\state{initial}[width=\pgfdecoratedinputsegmentlength,\n'
    text = text + '        next state=middle,\n'
    text = text + '        persistent postcomputation=\\pgfmathsetmacro\\previousroundedendlength{min(\\pgfdecorationsegmentlength,\\pgfdecoratedinputsegmentlength)}\n'
    text = text + '    ]{}\n'
    text = text + '    \\state{middle}[width=\\pgfdecoratedinputsegmentlength,next state=middle,\n'
    text = text + '        persistent precomputation={\n'
    text = text + '            \\pgfmathsetmacro\\roundedstartlength{min(\\previousroundedendlength,\\pgfdecoratedinputsegmentlength/2)}\n'
    text = text + '        },\n'
    text = text + '        persistent postcomputation=\\pgfmathsetmacro\\previousroundedendlength{min(\\pgfdecorationsegmentlength,\\pgfdecoratedinputsegmentlength/2)}\n'
    text = text + '    ]{\n'
    text = text + '        \\pgfsetcornersarced{\\pgfpoint{\\roundedstartlength}{\\roundedstartlength}}\n'
    text = text + '        \\pgfpathlineto{\\pgfpoint{0pt}{0pt}}\n'
    text = text + '    }\n'
    text = text + '    \\state{final}[if input segment is closepath={\\pgfpathclose}]\n'
    text = text + '    {\n'
    text = text + '        \\pgfpathlineto{\\pgfpointdecoratedpathlast}\n'
    text = text + '    }\n'
    text = text + '}\n'
    text = text + '\\tikzset{\n'
    text = text + 'dynamic rounded corners/.style={\n'
    text = text + 'decorate,\n'
    text = text + 'decoration={\n'
    text = text + 'dynamic rounded corners,\n'
    text = text + 'segment length=#1\n'
    text = text + '}\n'
    text = text + '}\n'
    text = text + '}\n'
    text = text + '\\tikzstyle{solid}=                   [dash pattern=]\n'
    text = text + '\\tikzstyle{dotted}=                  [dash pattern=on \\pgflinewidth off 2pt]\n'
    text = text + '\\tikzstyle{densely dotted}=          [dash pattern=on \\pgflinewidth off 1pt]\n'
    text = text + '\\tikzstyle{loosely dotted}=          [dash pattern=on \\pgflinewidth off 4pt]\n'
    text = text + '\\tikzstyle{dashed}=                  [dash pattern=on 3pt off 3pt]\n'
    text = text + '\\tikzstyle{densely dashed}=          [dash pattern=on 3pt off 2pt]\n'
    text = text + '\\tikzstyle{loosely dashed}=          [dash pattern=on 3pt off 6pt]\n'
    text = text + '\\tikzstyle{dashdotted}=              [dash pattern=on 3pt off 2pt on \\the\\pgflinewidth off 2pt]\n'
    text = text + '\\tikzstyle{densely dashdotted}=      [dash pattern=on 3pt off 1pt on \\the\\pgflinewidth off 1pt]\n'
    text = text + '\\tikzstyle{loosely dashdotted}=      [dash pattern=on 3pt off 4pt on \\the\\pgflinewidth off 4pt]\n'
    text = text + '\\tikzstyle{box_gray} = [rectangle, draw, top color=white, bottom color = gray!30, draw=gray!50!black!100, rounded corners]\n'
    text = text + '\\tikzstyle{box_yellow} = [rectangle, draw, top color=white, bottom color = yellow!30, draw=yellow!50!black!100, rounded corners]\n'

    # Define necessary colors in LaTeX
    processed_val_codes = []
    for curr_val in possible_synapse_vals:
        val_code = tools.get_alphabetic_code(curr_val)
        if not val_code in processed_val_codes:
            color = plt.cm.jet((curr_val - min_synapse_val)/(max_synapse_val - min_synapse_val))
            r,g,b = jetcolor_to_rgb(color)
            text = text + '\\definecolor{CV'+val_code+'}{RGB}{'+str(r)+','+str(g)+','+str(b)+'}\n'
            text = text + '\\newcommand*{\\CV'+val_code+'}[1]{extcolor{CV'+val_code+'}{#1}}\n'
            text = text + '\\definecolor{CW'+val_code+'}{RGB}{'+str(255-r)+','+str(255-g)+','+str(255-b)+'}\n'
            text = text + '\\newcommand*{\\CW'+val_code+'}[1]{extcolor{CW'+val_code+'}{#1}}\n'
            processed_val_codes.append(val_code)
            
    processed_val_codes = []
    for curr_val in possible_neuron_vals:
        val_code = tools.get_alphabetic_code(curr_val)
        if not val_code in processed_val_codes:
            color = plt.cm.jet((curr_val - min_neuron_val)/(max_neuron_val - min_neuron_val))
            r,g,b = jetcolor_to_rgb(color)
            text = text + '\\definecolor{CVV'+val_code+'}{RGB}{'+str(r)+','+str(g)+','+str(b)+'}\n'
            text = text + '\\newcommand*{\\CVV'+val_code+'}[1]{extcolor{CVV'+val_code+'}{#1}}\n'
            text = text + '\\definecolor{CWW'+val_code+'}{RGB}{'+str(255-r)+','+str(255-g)+','+str(255-b)+'}\n'
            text = text + '\\newcommand*{\\CWW'+val_code+'}[1]{extcolor{CWW'+val_code+'}{#1}}\n'
            processed_val_codes.append(val_code)

    text = text + '\\begin{document}\n'
    
    if draw_graph:
        circle_size = 46
        
        def angle(p0,p1):
            x0, y0 = p0
            x1, y1 = p1
            inner_product = x0*x1 + y0*y1
            len1 = math.hypot(x0, y0)
            len2 = math.hypot(x1, y1)
            return math.acos(inner_product/(len1*len2))
        
        def in_circle(pos, circle_size, segment):
            x0 = float(pos[0])
            y0 = float(pos[1])
            x1 = float(segment[0][0])
            y1 = float(segment[0][1])
            x2 = float(segment[1][0])
            y2 = float(segment[1][1])

            distance = (abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1))/(math.sqrt((y2-y1)**2 + (x2-x1)**2))

            alpha = angle((x2-x1,y2-y1),(x0-x1,y0-y1))
            beta = angle((x1-x2,y1-y2),(x0-x2,y0-y2))

            return distance < float(circle_size)/2.0 and alpha < math.pi/2 and beta < math.pi/2
        
        def recursive_split(segment, circle_size, pos_list):
            for pos in pos_list:
                if in_circle(pos, circle_size, segment):
                    while (True):
                        r = random.uniform(0.7*float(circle_size), 0.9*float(circle_size))
                        a = random.uniform(0, 2*math.pi)
                        sx = math.cos(a) * r
                        sy = math.sin(a) * r
                        segment_0 = (segment[0],(pos[0]+sx,pos[1]+sy))
                        segment_1 = ((pos[0]+sx,pos[1]+sy),segment[1])
                        # return [segment_0, segment_1]
                        # print(in_circle(pos, circle_size, segment_0), in_circle(pos, circle_size, segment_1))
                        if (not in_circle(pos, circle_size, segment_0)) and (not in_circle(pos, circle_size, segment_1)):
                            return recursive_split(segment_0, circle_size, pos_list) + recursive_split(segment_1, circle_size, pos_list)
            return [segment]
        
        text = text + '\\tikzset{every loop/.style={min distance=1pt,looseness=5}}\n'
        
        text = text + '\\begin{tikzpicture}[>=latex,\n'
        text = text + 'neuron/.style={\n'
        text = text + 'draw,\n'
        text = text + 'circle,\n'
        text = text + 'minimum width={'+str(circle_size)+'pt},\n'
        text = text + '}, neuron_nodraw/.style={\n'
        text = text + 'circle,\n'
        text = text + 'minimum width={'+str(circle_size)+'pt},\n'
        text = text + '}, node distance='+str(circle_size/2)+'pt]\n'
        
        positions = dict()
        k = 0
        l = 0
        for i in range(0, len(cell_layer_names)):
            for j in range(0, len(cell_layers[i])):
                text = text + '\\node[neuron,fill=CVV'+tools.get_alphabetic_code(neuron_values[cell_layers[i][j]])
                if j % 8 > 0:
                    text = text + ',right=of '+cell_layers[i][j-1]
                    positions[cell_layers[i][j]] = (3*circle_size/2 * (j % 8), -(k * circle_size + (l - 1) * 3 * circle_size/2))
                text = text +'] ('+cell_layers[i][j]+')'
                if j % 8 == 0:
                    text = text + ' at (0,-'+str(k * circle_size + l * 3 * circle_size/2)+'pt)'
                    positions[cell_layers[i][j]] = (3*circle_size/2 * (j % 8), -(k * circle_size + l * 3 * circle_size/2))
                    l = l + 1
                text = text +'{\\textcolor{CWW'+tools.get_alphabetic_code(neuron_values[cell_layers[i][j]])+'}{\\textbf{'+cell_layers[i][j]+'}}};\n'
            k = k + 1
    
        for syn_val_key in synapse_values.keys():
            start = positions[syn_val_key[0]]
            end = positions[syn_val_key[1]]
            if start == end:
                text = text + '\\path ('+syn_val_key[0]+') edge [loop above,->,color=CV'+tools.get_alphabetic_code(synapse_values[syn_val_key])+ '] ('+syn_val_key[1]+');\n'
            else:
                pos_list = []
                for pos_key in positions.keys():
                    if not (pos_key == syn_val_key[0] or pos_key == syn_val_key[1]):
                        pos_list.append(positions[pos_key])
                segment = (start, end)
                segment_list = recursive_split(segment, circle_size, pos_list)
                text = text + '\\path [-latex,draw,rounded corners=5pt,color=CV'+tools.get_alphabetic_code(synapse_values[syn_val_key])+ ',line width = 0.5pt] ('+syn_val_key[0]+')'
                for i in range(0, len(segment_list)-1):
                    text = text + ' -- ('+str(segment_list[i][1][0])+'pt,'+str(segment_list[i][1][1])+'pt)'
                text = text + ' -- ('+syn_val_key[1]+');\n'
       
        for fnr in range(0,8):
            active = not (fnr == 2 or fnr == 3 or fnr == 4)
            text = text + '\\node['+('neuron,' if active else 'neuron_nodraw')+'] at ('+str(fnr * 3 * circle_size/2)+'pt,-'+str(k * circle_size + l * 3 * circle_size/2)+'pt) (FULLY_1_'+str(fnr)+') {'+('' if active else '\\ldots')+'};\n'
            if active:
                text = text + '\\draw[->] ('+str(fnr * 3 * circle_size/2)+'pt,-'+str(k * circle_size + (l-1) * 3 * circle_size/2)+'pt) to (FULLY_1_'+str(fnr)+');\n'
        text = text + '\\node[right=of FULLY_1_7, anchor=west] {128 neurons};\n'
        l = l + 1
    
        for fnr in range(0,6):
            active = not (fnr == 2 or fnr == 3)
            text = text + '\\node['+('neuron,' if active else 'neuron_nodraw')+'] at ('+str(fnr * 3 * circle_size/2)+'pt,-'+str(k * circle_size + l * 3 * circle_size/2)+'pt) (FULLY_2_'+str(fnr)+') {'+('' if active else '\\ldots')+'};\n'
            if active:
                for p in range(0,8):
                    if not (p == 2 or p == 3 or p == 4):
                        text = text + '\\draw[->] (FULLY_1_'+str(p)+') to (FULLY_2_'+str(fnr)+');\n'
        text = text + '\\node[right=of FULLY_2_5, anchor=west] {32 neurons};\n'
        l = l + 1
        
        for fnr in range(0,4):
            F3label = ''
            if fnr == 0:
                F3label = '$x$'
            elif fnr == 1:
                F3label = '$y$'
            elif fnr == 2:
                F3label = '$\Delta x$'
            elif fnr == 3:
                F3label = '$\Delta y$'
                
            text = text + '\\node[neuron] at ('+str(fnr * 3 * circle_size/2)+'pt,-'+str(k * circle_size + l * 3 * circle_size/2)+'pt) (FULLY_3_'+str(fnr)+') {'+F3label+'};\n'
            for p in range(0,6):
                if not (p == 2 or p == 3):
                    text = text + '\\draw[->] (FULLY_2_'+str(p)+') to (FULLY_3_'+str(fnr)+');\n'
        text = text + '\\node[right=of FULLY_3_3, anchor=west] {4 neurons};\n'
        
        text = text + '\\begin{pgfonlayer}{bg}\n'
        text = text + '\\node[box_yellow,fit=(R1)(R8)](box_0){};\n'
        text = text + '\\end{pgfonlayer}\n'
        text = text + '\\node[label={[label distance=0.5cm,text depth=-1ex,rotate=90]Retina}] at (box_0.west) {};\n'
        
        text = text + '\\begin{pgfonlayer}{bg}\n'
        text = text + '\\node[box_yellow,fit=(Lawf2)(L5)](box_1){};\n'
        text = text + '\\end{pgfonlayer}\n'
        text = text + '\\node[label={[label distance=0.5cm,text depth=-1ex,rotate=90]Lamina}] at (box_1.west) {};\n'
        
        text = text + '\\begin{pgfonlayer}{bg}\n'
        text = text + '\\node[box_yellow,fit=(Tm9)(T2)](box_2){};\n'
        text = text + '\\end{pgfonlayer}\n'
        text = text + '\\node[label={[label distance=0.5cm,text depth=-1ex,rotate=90]Medulla}] at (box_2.west) {};\n'
        
        text = text + '\\begin{pgfonlayer}{bg}\n'
        text = text + '\\node[box_yellow,fit=(T4a)(T5d)](box_3){};\n'
        text = text + '\\end{pgfonlayer}\n'
        text = text + '\\node[label={[label distance=0.5cm,text depth=-1ex,rotate=90]Lobula Plate}] at (box_3.west) {};\n'
        
        text = text + '\\end{tikzpicture}\n'
    
    if draw_table:
        tabconfig = '|c|c|'
        for node in nodes:
            tabconfig = tabconfig + 'c|'
        text = text + '\\begin{tabular}{'+tabconfig+'}\n'
        text = text + '\\hline\n&&'
        for i in range(0, len(cell_layer_names)):
            mdelim = 'c'
            if i == len(cell_layer_names) - 1:
                mdelim = '|c|'
            text = text + '\\multicolumn{'+str(len(cell_layers[i]))+'}{'+mdelim+'}{'+cell_layer_names[i]+'}'
            if not i == len(cell_layer_names) - 1:
                text = text + '&'
        text = text + '\\\\\\hline\n&&'
        for i in range(0, len(cell_layer_names)):
            for j in range(0, len(cell_layers[i])):
                text = text + '\\cellcolor{CVV'+tools.get_alphabetic_code(neuron_values[cell_layers[i][j]])+'}'
                text = text + '\\textcolor{CWW'+tools.get_alphabetic_code(neuron_values[cell_layers[i][j]])+'}{\\textbf{'+cell_layers[i][j]+'}}'
                if not (i == len(cell_layer_names) - 1 and j == len(cell_layers[i]) - 1):
                    text = text + '&'
        text = text + '\\\\\\hline\n'
        for i1 in range(0, len(cell_layer_names)):
            text = text + '\\multirow{'+str(len(cell_layers[i]))+'}{*}{\\rotatebox[origin=c]{90}{'+cell_layer_names[i1]+'}} &'
            for j1 in range(0, len(cell_layers[i1])):
                if not j1 == 0:
                    text = text + '&'
                text = text + '\\cellcolor{CVV'+tools.get_alphabetic_code(neuron_values[cell_layers[i1][j1]])+'}'
                text = text + '\\textcolor{CWW'+tools.get_alphabetic_code(neuron_values[cell_layers[i1][j1]])+'}{\\textbf{'+cell_layers[i1][j1]+'}} &'       
                for i2 in range(0, len(cell_layer_names)):
                    for j2 in range(0, len(cell_layers[i2])):
                        if ((cell_layers[i1][j1],cell_layers[i2][j2]) in synapse_values.keys()):
                            text = text + '\\cellcolor{CV'+tools.get_alphabetic_code(synapse_values[cell_layers[i1][j1],cell_layers[i2][j2]])+'}'
                            text = text + '\\textcolor{CW'+tools.get_alphabetic_code(synapse_values[cell_layers[i1][j1],cell_layers[i2][j2]])+'}'
                            text = text + '{\\textbf{'+'{0:.4f}'.format(synapse_values[cell_layers[i1][j1],cell_layers[i2][j2]])+'}}'
                        if not (i2 == len(cell_layer_names) - 1 and j2 == len(cell_layers[i2]) - 1):
                            text = text + '&'
                if j1 == len(cell_layers[i1]) - 1:
                    text = text + '\\\\\\hline\n'
                else:
                    text = text + '\\\\\\cline{2-'+str(len(nodes)+2)+'}\n'
        text = text + '\\end{tabular}\n'
    
    text = text + '\\minipage{100pt}\n'
    if (draw_graph):
        text = text + '\\vspace*{-1000pt}\n'
    text = text + '\\begin{figure}\n'
    #text = text + 'Neuron:\n'
    text = text + '\\subfloat[Neurons]{\\hspace*{8pt}\\includegraphics{'+file_name.split('/')[-1]+'_colorbar_n.pdf}\\hspace*{8pt}}\n'
    text = text + '\\newline\n'
    text = text + '\\subfloat[Synapses]{\\hspace*{8pt}\\includegraphics{'+file_name.split('/')[-1]+'_colorbar_s.pdf}\\hspace*{8pt}}\n'
    text = text + '\\end{figure}\n'
    text = text + '\\endminipage\n'

    text = text + '\\end{document}\n'
    file.write(text)
    file.close()
    
    a = np.array([[min_synapse_val,max_synapse_val]])
    plt.figure(figsize=(2, 3))
    img = plt.imshow(a, cmap="jet")
    img.set_visible(False)
    plt.gca().set_visible(False)
    plt.colorbar(orientation="vertical")
    plt.savefig(file_name + '_colorbar_s.pdf', bbox_inches='tight')
    
    a = np.array([[min_neuron_val,max_neuron_val]])
    plt.figure(figsize=(2, 3))
    img = plt.imshow(a, cmap="jet")
    img.set_visible(False)
    plt.gca().set_visible(False)
    plt.colorbar(orientation="vertical")
    plt.savefig(file_name + '_colorbar_n.pdf', bbox_inches='tight')
    
    cmd = ['lualatex', '-interaction', 'nonstopmode', file_name.split('/')[-1]+'.tex']
    cwd = ''
    for i in range(0,len(file_name.split('/'))-1):
        cwd += file_name.split('/')[i]+'/'
    proc = subprocess.Popen(cmd, cwd=cwd)
    proc.communicate()
        
    return text