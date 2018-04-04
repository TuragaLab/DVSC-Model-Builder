known_signs = dict()

# C2 and C3 are inhibitory (Michael Reiser, 27.01.2017)
known_signs[('C2', None)] = -1.0
known_signs[('C3', None)] = -1.0

# Mi4 is inhibitory (Michael Reiser, 27.01.2017)
known_signs[('Mi4', None)] = -1.0
# Mi9 is inhibitory (Michael Reiser, 27.01.2017)
known_signs[('Mi9', None)] = -1.0


# All photoreceptors use histamine as a neurotransmitter --> inhibitory (Hardie, 1989).
known_signs[('R1', None)] = -1.0
known_signs[('R2', None)] = -1.0
known_signs[('R3', None)] = -1.0
known_signs[('R4', None)] = -1.0
known_signs[('R5', None)] = -1.0
known_signs[('R6', None)] = -1.0
known_signs[('R7', None)] = -1.0
known_signs[('R8', None)] = -1.0


def get_sign(src, tar):
    if (src, tar) in known_signs.keys():
        return known_signs[(src, tar)]
    elif (src, None) in known_signs.keys():
        return known_signs[(src, None)]
    elif (None, tar) in known_signs.keys():
        return known_signs[(None, tar)]
    else:
        return 0.0