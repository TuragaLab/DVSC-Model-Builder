import copy, operator, math
import numpy as np

offset = dict()

# Home column (1 element)
offset['h'] = (0.0, 0.0)
# Ring 1 (6 elements):
offset['A'] = (-1.0, 0.0)
offset['B'] = (-1.0, 1.0)
offset['C'] = (0.0, 1.0)
offset['D'] = (1.0, 0.0)
offset['E'] = (1.0, -1.0)
offset['F'] = (0.0, -1.0)
# Ring 2 (12 elements):
offset['G'] = (-2.0, 0.0)
offset['H'] = (-2.0, 1.0)
offset['I'] = (-2.0, 2.0)
offset['J'] = (-1.0, 2.0)
offset['K'] = (0.0, 2.0)
offset['L'] = (1.0, 1.0)
offset['M'] = (2.0, 0.0)
offset['N'] = (2.0, -1.0)
offset['O'] = (2.0, -2.0)
offset['P'] = (1.0, -2.0)
offset['Q'] = (0.0, -2.0)
offset['R'] = (-1.0, -1.0)
# Ring 3 (18 elements):
offset['S'] = (-3.0, 0.0)
offset['T'] = (-3.0, 1.0)
offset['U'] = (-3.0, 2.0)
offset['V'] = (-3.0, 3.0)
offset['W'] = (-2.0, 3.0)
offset['X'] = (-1.0, 3.0)
offset['Y'] = (0.0, 3.0)
offset['Z'] = (1.0, 2.0)
offset['AA'] = (2.0, 1.0)
offset['AB'] = (3.0, 0.0)
offset['AC'] = (3.0, -1.0)
offset['AD'] = (3.0, -2.0)
offset['AE'] = (3.0, -3.0)
offset['AF'] = (2.0, -3.0)
offset['AG'] = (1.0, -3.0)
offset['AH'] = (0.0, -3.0)
offset['AI'] = (-1.0, -2.0)
offset['AJ'] = (-2.0, -1.0)

# Additional patched offsets to fix json dataset issues - those that are commented out should be optimized by the algorithm
offset['home'] = offset['h']
#offset['home?'] = offset['h']
#offset['B/home(rep)'] = offset['h']
# offset['A?'] = offset['A']
# offset['E?'] = offset['E']
# offset['I?'] = offset['I']
# offset['O?'] = offset['O']
# offset['P?'] = offset['P']
# offset['Q?'] = offset['Q']
offset['F(rep)']  = offset['F']
offset['E(rep)'] = offset['E']

# Finer grained offsets between the main columns - commented out - to be optimized by the algorithm
# Currently the datasets do not have enough of these annotations to derive any pattern...
# offset['PQ'] = offset['F']
# offset['P^Q'] = offset['F']
# offset['P^Q?'] = offset['F']
# offset['JK'] = offset['C']
# offset['NO'] = offset['N']
# offset['OP'] = offset['E']
# offset['O^P'] = offset['E']
# offset['IJ'] = offset['B']

def hex_area(side_length):
    # Nearest neighbors in a hexagon (6)
    base_offsets = [(-1.0,0.0),(-1.0,1.0),(0.0,1.0),(1.0,0.0),(1.0,-1.0),(0.0,-1.0)]
    new_area = set()
    new_area.add((0.0,0.0))
    for i in range(0, side_length - 1):
        old_area = copy.deepcopy(new_area)
        new_area = set()
        for offset in old_area:
            # print offset
            for base_offset in base_offsets:
                new_offset = tuple(map(operator.add, offset, base_offset))
                new_area.add(new_offset)
        for offset in old_area:
            new_area.add(offset)
    return new_area

def compute_area_coverage(offsets, side_length):
    area = hex_area(side_length)
    count = 0
    for area_off in area:
        if area_off in offsets:
            count = count + 1
    return float(count)/float(len(area))


# Rotate offsets along y axis (hex coordinates)
def yrotate_offset(offset):
    return (offset[0] + offset[1], -offset[1])


hex_to_cartesian_R = math.sqrt(2.0/math.sqrt(3.0))*np.asarray([[1.0,0.5],[0.0,math.sqrt(3.0)/2.0]])
cartesian_to_hex_R = np.linalg.inv(hex_to_cartesian_R)

#print(hex_to_cartesian_R)
#print(cartesian_to_hex_R)

# (y,x) -> (v,u)
def hex_to_cartesian(offset):
    return (hex_to_cartesian_R[0][0] * offset[0] + hex_to_cartesian_R[0][1] * offset[1],
            hex_to_cartesian_R[1][0] * offset[0] + hex_to_cartesian_R[1][1] * offset[1])

# (v,u) -> (y,x)
def cartesian_to_hex(offset):
    return (cartesian_to_hex_R[0][0] * offset[0] + cartesian_to_hex_R[0][1] * offset[1],
            cartesian_to_hex_R[1][0] * offset[0] + cartesian_to_hex_R[1][1] * offset[1])
    