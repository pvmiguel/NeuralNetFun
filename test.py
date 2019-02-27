import Network

net = Network.Network([2,3,1])

print("Layers = %f" % (net.n_layers))
print(net.size_layers)
print(net.biases)
print(net.weights[0])
print(net.weights[1])
