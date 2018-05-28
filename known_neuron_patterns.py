known_neuron_patterns = dict()

# LAWF: Enhance neural coding by subtracting low-frequency signals from the inputs to motion detection circuits (Tuthill 2014)
# ~ 140 Lawf2 per optic lobe (780 ommatidia = 30(A-P)x26(D-V), 135 Lawf2 = 15(A-P)x9(D-V))
# ~ 5 Lawf2 connections to each cartridge
# Innervation in lamina: 28 cartridges, skewed along dorsal-ventral (A-P: 4, D-V: 7)
# Innervation in medulla: M1 127 cartridges (hexagon sidelength 7), M8-10 19 cartridges (hexagon sidelength 3)
known_neuron_patterns['Lawf2'] = ('stride', (3, 2))

known_neuron_patterns['Lawf1'] = ('stride', (3, 2))

# Nern 2015
# 750 columns

# Dm1: ~40 cells per OL
known_neuron_patterns['Dm1'] = ('stochastic', 40.0/750.0)

# Dm2: size and coverage pattern suggest ~ 1 cell per column
known_neuron_patterns['Dm2'] = ('stride', (1, 1))

# Dm3: not counted; could be up to ~ one per column
known_neuron_patterns['Dm3'] = ('stride', (1, 1))

# Dm4: ~40 cells per OL
known_neuron_patterns['Dm4'] = ('stochastic', 40.0/750.0)

# Dm6: ~30 cells per OL
known_neuron_patterns['Dm6'] = ('stochastic', 30.0/750.0)

# Dm8: coverage pattern consistent with nearly 1 cell per column
known_neuron_patterns['Dm8'] = ('stride', (1, 1))

# Dm9: ~110 cells per OL
known_neuron_patterns['Dm9'] = ('stochastic', 110.0/750.0)

# Dm10: ~300 cells per OL
known_neuron_patterns['Dm10'] = ('stochastic', 300.0/750.0)

# Dm11: ~70 cells per OL
known_neuron_patterns['Dm11'] = ('stochastic', 70.0/750.0)

# Dm12: ~110 cells per OL
known_neuron_patterns['Dm12'] = ('stochastic', 120.0/750.0)

# Dm13: ~15-20 cells per OL
known_neuron_patterns['Dm13'] = ('stochastic', ((15.0+20.0)/2.0)/750.0)

# Dm14: ~15 cells per OL
known_neuron_patterns['Dm14'] = ('stochastic', 15.0/750.0)

# Dm15: ~250 cells per OL
known_neuron_patterns['Dm15'] = ('stochastic', 250.0/750.0)

# Dm16: ~100 cells per OL
known_neuron_patterns['Dm16'] = ('stochastic', 100.0/750.0)

# Dm17: ~ 5 cells near the anterior edge of the medulla
known_neuron_patterns['Dm17'] = ('stochastic', 5.0/750.0)

# Dm18: ~20 per OL
known_neuron_patterns['Dm18'] = ('stochastic', 20.0/750.0)

# Dm19: ~15 cells per OL
known_neuron_patterns['Dm19'] = ('stochastic', 15.0/750.0)

# Dm20: ~50 cells per OL
known_neuron_patterns['Dm20'] = ('stochastic', 50.0/750.0)

# Taken as tiled (2x1) since found 4x in 7 columns (Takemura 2013 S1)
known_neuron_patterns['Tm6'] = ('stochastic', 0.5)
