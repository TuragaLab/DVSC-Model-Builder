import model_base

# Constructor test
node = model_base.Node(name='R1',
                       pattern=('stride', (1, 1)),
                       activation='relu',
                       bias=3.5,
                       time_constant=12.5,)

# Legacy accessors
print(node[0], node[1], node[2], node[3], node[4])

# Constructor test
edge = model_base.Edge(src='R1',
                       tar='L1',
                       offsets=[((0, 0), 40.0)],
                       alpha=-1.0,
                       lambda_mult=6.0,
                       time_constant=12.5,)

# Legacy accessors test
print(edge[0], edge[1], edge[2], edge[3], edge[4], edge[5])