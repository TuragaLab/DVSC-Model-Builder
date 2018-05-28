known_signs = dict()

# All photoreceptors use histamine as a neurotransmitter --> inhibitory (Hardie, 1989).
known_signs[('R1', None)] = (-1.0, True, ['Hardie1989'])
known_signs[('R2', None)] = (-1.0, True, ['Hardie1989'])
known_signs[('R3', None)] = (-1.0, True, ['Hardie1989'])
known_signs[('R4', None)] = (-1.0, True, ['Hardie1989'])
known_signs[('R5', None)] = (-1.0, True, ['Hardie1989'])
known_signs[('R6', None)] = (-1.0, True, ['Hardie1989'])
known_signs[('R7', None)] = (-1.0, True, ['Hardie1989'])
known_signs[('R8', None)] = (-1.0, True, ['Hardie1989'])

# R8 is excitatory on Mi cells
known_signs[('R8', 'Mi1')] = (1.0, True, ['NernPC2018'])
known_signs[('R8', 'Mi4')] = (1.0, True, ['NernPC2018'])
known_signs[('R8', 'Mi9')] = (1.0, True, ['NernPC2018'])
known_signs[('R8', 'Mi15')] = (1.0, True, ['NernPC2018'])


# L1 is inhibitory, glutamatergic (Aljoscha Nern 21.05.2018)
known_signs[('L1', None)] = (-1.0, True, ['NernPC2018'])

# Am probably inhibitory
known_signs[('Am', None)] = (-1.0, False, ['NernPC2018'])
# Am excitatory onto L1 and T1
known_signs[('Am', 'L1')] = (1.0, True, ['NernPC2018'])
known_signs[('Am', 'T1')] = (1.0, True, ['NernPC2018'])


# L2/L3 are excitatory, cholinergic (Aljoscha Nern 21.05.2018)
known_signs[('L2', None)] = (1.0, True, ['NernPC2018'])
known_signs[('L3', None)] = (1.0, True, ['NernPC2018'])

# All T5 inputs express cholinergic markers and should be excitatory (Aljoscha Nern 21.05.2018)
known_signs[(None, 'T5a')] = (1.0, True, ['NernPC2018'])
known_signs[(None, 'T5b')] = (1.0, True, ['NernPC2018'])
known_signs[(None, 'T5c')] = (1.0, True, ['NernPC2018'])
known_signs[(None, 'T5d')] = (1.0, True, ['NernPC2018'])


# C2 and C3 are inhibitory (Michael Reiser, 27.01.2017)
known_signs[('C2', None)] = (-1.0, True, ['ReiserPC2017', 'NernPC2018'])
known_signs[('C3', None)] = (-1.0, True, ['ReiserPC2017', 'NernPC2018'])
# C2/C3 inputs to L1/L2 and maybe L3 are excitatory
known_signs[('C2', 'L1')] = (1.0, True, ['NernPC2018'])
known_signs[('C3', 'L1')] = (1.0, True, ['NernPC2018'])
known_signs[('C2', 'L2')] = (1.0, True, ['NernPC2018'])
known_signs[('C3', 'L2')] = (1.0, True, ['NernPC2018'])
known_signs[('C2', 'L3')] = (-1.0, False, ['NernPC2018'])
known_signs[('C3', 'L3')] = (-1.0, False, ['NernPC2018'])

# CT1 inhibitory
known_signs[('CT1', None)] = (-1.0, True, ['NernPC2018'])

# Mi1 is excitatory
known_signs[('Mi1', None)] = (1.0, False, ['NernPC2018'])

# Most MiX are inhibitory (Aljoscha Nern 21.05.2018)
known_signs[('Mi2', None)] = (-1.0, True, ['NernPC2018'])
known_signs[('Mi3', None)] = (-1.0, True, ['NernPC2018'])

# Mi4 is inhibitory (Michael Reiser, 27.01.2017) (Aljoscha Nern 21.05.2018)
known_signs[('Mi4', None)] = (-1.0, True, ['ReiserPC2017', 'NernPC2018'])
# Mi9 is inhibitory (Michael Reiser, 27.01.2017) (Aljoscha Nern 22.05.2018)
known_signs[('Mi9', None)] = (-1.0, True, ['ReiserPC2017', 'NernPC2018'])
known_signs[('Mi10', None)] = (-1.0, True, ['NernPC2018'])
known_signs[('Mi12', None)] = (-1.0, False, ['NernPC2018']) # Unsure about this annotation of cell type
known_signs[('Mi13', None)] = (-1.0, True, ['NernPC2018'])
known_signs[('Mi14', None)] = (-1.0, False, ['NernPC2018'])

# Mi15 is excitatory
known_signs[('Mi15', None)] = (1.0, False, ['NernPC2018'])

# Most Tm cells are excitatory
# TmY14 is inhibitory
known_signs[('TmY14', None)] = (-1.0, True, ['NernPC2018'])
# TmY15 is inhibitory
known_signs[('TmY15', None)] = (-1.0, True, ['NernPC2018'])

# TmY5a inhibitory
known_signs[('TmY5a', None)] = (-1.0, True, ['NernPC2018'])


# Tm5a/b/c potentially inhibitory
# Lin 2016
# "It is interesting to note that the 
# chromatic Tm neurons have mixed neurotransmitter phenotypes: Tm5c is glutamatergic 
# while Tm5a and Tm5b have a cholinergic phenotype"
known_signs[('Tm5a', None)] = (1.0, False, ['Karuppudurai2014', 'Lin2016'])
known_signs[('Tm5b', None)] = (1.0, False, ['Karuppudurai2014', 'Lin2016'])
known_signs[('Tm5c', None)] = (-1.0, False, ['Karuppudurai2014', 'Lin2016'])


# Unpublished data; Nern, Aljoscha, 01.05.2018, (Aljoscha Nern 21.05.2018)
# Most Pm and Dm inhibitory
known_signs[('Pm3', None)] = (-1.0, False, ['NernPC2018'])
known_signs[('Pm4', None)] = (-1.0, False, ['NernPC2018'])
known_signs[('Dm10', None)] = (-1.0, False, ['NernPC2018'])


# CT branches excitatory to each other
known_signs['CT1', 'CT1L'] = (1.0, True, [])
known_signs['CT1L', 'CT1'] = (1.0, True, [])
known_signs['CT1G', 'CT1L'] = (1.0, True, [])
known_signs['CT1L', 'CT1G'] = (1.0, True, [])



def get_sign(src, tar):
    if (src, tar) in known_signs.keys():
        # Exact match for pre-post pair
        return known_signs[(src, tar)]
    elif (None, tar) in known_signs.keys():
        # Generalization for postsynaptic match
        return known_signs[(None, tar)]
    elif (src, None) in known_signs.keys():
        # Generalization for presynaptic match
        return known_signs[(src, None)]
    else:
        # Assume default excitatory and non-fixed
        return (1.0, False, [])