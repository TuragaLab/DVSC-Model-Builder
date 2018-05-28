# Defining static nodes and edges not represented in the datasets
import hexgrid_reference
import model_base

hex_areas = []
for j in range(0, 41):
    hex_areas.append([offset for offset in hexgrid_reference.hex_area(j)])
    
nodes = []
edges = []

################################################################################
# Input units

input_units = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']

################################################################################
# Output units
# All cells that go into lobula layers plus T4a,b,c,d
output_units = ['Tm1', 'Tm2',
                'Tm3', 'Tm3-ant', 'Tm3-post',
                'Tm4', 'Tm4-ant', 'Tm4-post',
                'Tm5a', 'Tm5b', 'Tm5c', 'Tm5d',
                'Tm6', 'Tm9', 'Tm16', 'Tm20', 'Tm28',
                'TmY3', 'TmY4', 'Tm5Y', 'TmY5a', 'TmY9',
                'TmY10','TmY13', 'TmY14', 'TmY15', 'TmY17',
                'T1', 'T2', 'T2a', 'T3',
                'T4a', 'T4b', 'T4c', 'T4d',
                'T5a', 'T5b', 'T5c', 'T5d']


################################################################################
# Nodes

# Motion detection (intensity)
# The PR neurons target 6 different cartridges
nodes.append(model_base.Node().from_tuple(('R1', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('R2', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('R3', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('R4', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('R5', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('R6', ('stride', (1, 1)), 'relu', 3.5)))

# Color vision (30% blue, 70% yellow)
nodes.append(model_base.Node().from_tuple(('R7', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('R8', ('stride', (1, 1)), 'relu', 3.5)))

# Lamina monopolar cells
nodes.append(model_base.Node().from_tuple(('L1', ('stride', (1, 1)), 'relu', 3.5)))  # ON-pathway (Armin Bahl)
nodes.append(model_base.Node().from_tuple(('L2', ('stride', (1, 1)), 'relu', 3.5)))  # OFF-pathway (Armin Bahl)
nodes.append(model_base.Node().from_tuple(('L3', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('L4', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('L5', ('stride', (1, 1)), 'relu', 3.5)))
    
# Lamina wide-field neurons
# LAWF: Enhance neural coding by subtracting low-frequency signals from the inputs to motion detection circuits (Tuthill 2014)
# ~ 140 Lawf2 per optic lobe (780 ommatidia = 30(A-P)x26(D-V), 135 Lawf2 = 15(A-P)x9(D-V))
# ~ 5 Lawf2 connections to each cartridge
# Innervation in lamina: 28 cartridges, skewed along dorsal-ventral (A-P: 4, D-V: 7)
# Innervation in medulla: M1 127 cartridges (hexagon sidelength 7), M8-10 19 cartridges (hexagon sidelength 3)
nodes.append(model_base.Node().from_tuple(('Lawf2', ('stride', (3, 2)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('Lawf1', ('stride', (3, 2)), 'relu', 3.5)))

# Amacrine cell
nodes.append(model_base.Node().from_tuple(('Am', ('stride', (1, 1)), 'relu', 3.5)))

# Centrifugal medulla neurons (feedback)
nodes.append(model_base.Node().from_tuple(('C2', ('stride', (1, 1)), 'relu', 3.5)))
nodes.append(model_base.Node().from_tuple(('C3', ('stride', (1, 1)), 'relu', 3.5)))

# CT1 local and global
nodes.append(model_base.Node(name='CT1G', pattern=('single', 'center'), activation=None, bias=3.5))
nodes.append(model_base.Node(name='CT1L', pattern=('tile', 5), activation=None, bias=3.5))


################################################################################
# Edges

# Lamina connections; Rivera-Alba (2011)
# Changed model: R1 - R6 project into the same cartridge to simplify programming
# All photoreceptors use histamine as a neurotransmitter --> inhibitory (Hardie, 1989).

edges.append(model_base.Edge().from_tuple(('R1', 'L1', [((0, 0), 40.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R2', 'L1', [((0, 0), 43.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R3', 'L1', [((0, 0), 37.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R4', 'L1', [((0, 0), 38.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R5', 'L1', [((0, 0), 38.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R6', 'L1', [((0, 0), 45.0)], -1.0, 1.0))) # simplified, accounted for in input data

edges.append(model_base.Edge().from_tuple(('R1', 'L2', [((0, 0), 46.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R2', 'L2', [((0, 0), 45.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R3', 'L2', [((0, 0), 39.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R4', 'L2', [((0, 0), 41.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R5', 'L2', [((0, 0), 39.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R6', 'L2', [((0, 0), 47.0)], -1.0, 1.0))) # simplified, accounted for in input data

edges.append(model_base.Edge().from_tuple(('R1', 'L3', [((0, 0), 11.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R2', 'L3', [((0, 0), 10.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R3', 'L3', [((0, 0), 4.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R4', 'L3', [((0, 0), 8.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R5', 'L3', [((0, 0), 6.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R6', 'L3', [((0, 0), 12.0)], -1.0, 1.0))) # simplified, accounted for in input data
         
edges.append(model_base.Edge().from_tuple(('R1', 'Am', [((0, 0), 36.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R2', 'Am', [((0, 0), 39.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R3', 'Am', [((0, 0), 40.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R4', 'Am', [((0, 0), 37.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R5', 'Am', [((0, 0), 35.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R6', 'Am', [((0, 0), 40.0)], -1.0, 1.0))) # simplified, accounted for in input data

edges.append(model_base.Edge().from_tuple(('R1', 'T1', [((0, 0), 2.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R2', 'T1', [((0, 0), 2.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R3', 'T1', [((0, 0), 2.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R4', 'T1', [((0, 0), 2.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R5', 'T1', [((0, 0), 2.0)], -1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('R6', 'T1', [((0, 0), 2.0)], -1.0, 1.0))) # simplified, accounted for in input data

# L2 --> R1, R2, L1, L4, L4+x, L4-y, T1
edges.append(model_base.Edge().from_tuple(('L2', 'R1', [((0, 0), 1.0)], 1.0, 1.0)))  # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('L2', 'R2', [((0, 0), 2.0)], 1.0, 1.0)))  # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('L2', 'L1', [((0, 0), 3.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('L2', 'L4', [((0, 0), 5.0), ((-1, 1), 5.0), ((0, 1), 3.0)], 1.0, 1.0)))

edges.append(model_base.Edge().from_tuple(('L4', 'R1', [((1, -1), 1.0), ((0, -1), 1.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('L4', 'R2', [((0, -1), 1.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('L4', 'R3', [((1, -1), 2.0), ((0, -1), 2.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('L4', 'R4', [((1, -1), 1.0), ((0, -1), 1.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('L4', 'R5', [((0, 0), 1.0), ((0, -1), 3.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('L4', 'R6', [((1, -1), 2.0), ((0, -1), 2.0)], 1.0, 1.0))) # simplified, accounted for in input data

edges.append(model_base.Edge().from_tuple(('L4', 'L2', [((0, 0), 4.0), ((1, -1), 6.0), ((0, -1), 5.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('L4', 'L3', [((0, 0), 0.0), ((1, -1), 1.0), ((0, -1), 1.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('L4', 'L4', [((1, -1), 2.0), (( 0, -1), 3.0), ((-1, 0), 1.0), ((1, 0), 3.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('L4', 'L5', [((0, 0), 2.0), ((1, -1), 3.0), ((0, -1), 2.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('L4', 'Am', [((0, 0), 1.0), ((1, -1), 1.0), ((0, -1), 1.0)], 1.0, 1.0)))

# Am --> R1, R2, R4, R5, L1, L2, L3, L4, L5, Am, T1, C2, C3
edges.append(model_base.Edge().from_tuple(('Am', 'R1', [((0, 0), 1.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('Am', 'R2', [((0, 0), 1.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('Am', 'R4', [((0, 0), 2.0)], 1.0, 1.0))) # simplified, accounted for in input data
edges.append(model_base.Edge().from_tuple(('Am', 'R5', [((0, 0), 2.0)], 1.0, 1.0))) # simplified, accounted for in input data

edges.append(model_base.Edge().from_tuple(('Am', 'L1', [((0, 0), 1.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Am', 'L2', [((0, 0), 4.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Am', 'L3', [((0, 0), 14.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Am', 'L4', [((0, 0), 5.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Am', 'Am', [((0, 0), 2.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Am', 'T1', [((0, 0), 63.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Am', 'C2', [((0, 0), 1.0)], 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Am', 'C3', [((0, 0), 1.0)], 1.0, 1.0)))

# C2 --> L2, L5, Am, Lawf
# C2 is inhibitory (Michael Reiser, 27.01.2017)
edges.append(model_base.Edge().from_tuple(('C2', 'Am', [((0, 0), 8.0)], -1.0, 1.0)))

# C3 --> R3, L1, L2, L3, Am
# C3 is inhibitory (Michael Reiser, 27.01.2017)
edges.append(model_base.Edge().from_tuple(('C3', 'R3', [((-2, 1), 1.0)], -1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('C3', 'L1', [((0, 0), 6.0)], -1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('C3', 'L3', [((0, 0), 1.0)], -1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('C3', 'Am', [((0, 0), 22.0)], -1.0, 1.0)))

lawf2_lamina_range = [(-3,-1),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(3,-1),(4,-1),
                      (-4,0),(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0),(4,0),
                      (-4,1),(-3,1),(-2,1),(-1,1),(0,1),(1,1),(2,1),(3,1)]

# Tuthill 2014 (LAWF); "It is likely that wide-field neurons form synapses on most cell types in the lamina [...] high [...] on Am and L3"
# Lawf --> R1, R2, R3, R4, R6, L1, L2, L3, L5, Am, T1, C2, C3, Lawf
# Lawf2 range (distributed to ~5 processes of Lawf2 cells)
edges.append(model_base.Edge().from_tuple(('Lawf2', 'Lawf2', list(zip(lawf2_lamina_range, [7.0/5.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'R1', list(zip(lawf2_lamina_range, [1.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'R2', list(zip(lawf2_lamina_range, [3.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'R3', list(zip(lawf2_lamina_range, [1.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'R4', list(zip(lawf2_lamina_range, [1.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'R5', list(zip(lawf2_lamina_range, [0.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'R6', list(zip(lawf2_lamina_range, [1.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'L1', list(zip(lawf2_lamina_range, [2.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'L2', list(zip(lawf2_lamina_range, [9.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'L3', list(zip(lawf2_lamina_range, [21.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'L5', list(zip(lawf2_lamina_range, [1.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'Am', list(zip(lawf2_lamina_range, [36.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'T1', list(zip(lawf2_lamina_range, [6.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'C2', list(zip(lawf2_lamina_range, [6.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('Lawf2', 'C3', list(zip(lawf2_lamina_range, [2.0/11.0 for o in lawf2_lamina_range])), 1.0, 1.0)))

# Hypothesis 1: L1/L2 connect to Lawf2 in the M1 layer
edges.append(model_base.Edge().from_tuple(('L1', 'Lawf2', list(zip(hex_areas[3], [5.0 for o in hex_areas[3]])), 1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('L2', 'Lawf2', list(zip(hex_areas[3], [5.0 for o in hex_areas[3]])), 1.0, 1.0)))
    
# Hypothesis 2: C2/C3 connect to Lawf2 in the M8-M10 layers
edges.append(model_base.Edge().from_tuple(('C2', 'Lawf2', list(zip(hex_areas[3], [5.0 for o in hex_areas[3]])), -1.0, 1.0)))
edges.append(model_base.Edge().from_tuple(('C3', 'Lawf2', list(zip(hex_areas[3], [5.0 for o in hex_areas[3]])), -1.0, 1.0)))

# CT1 branches
# One global CT1 cell connects to all branches
edges.append(model_base.Edge(src='CT1G', tar='CT1L', edge_type='elec', alpha=1.0, alpha_fixed=True,
                             offsets=list(zip(hex_areas[40], [10.0 for o in hex_areas[40]]))))
edges.append(model_base.Edge(src='CT1L', tar='CT1G', edge_type='elec', alpha=1.0, alpha_fixed=True,
                             offsets=list(zip(hex_areas[40], [10.0 for o in hex_areas[40]]))))

# One local CT1 branch covers about 5x5 columns
edges.append(model_base.Edge(src='CT1L', tar='CT1', edge_type='elec', alpha=1.0, alpha_fixed=True,
                             offsets=list(zip(hex_areas[3], [10.0 for o in hex_areas[3]]))))
edges.append(model_base.Edge(src='CT1', tar='CT1L', edge_type='elec', alpha=1.0, alpha_fixed=True,
                             offsets=list(zip(hex_areas[3], [10.0 for o in hex_areas[3]]))))


