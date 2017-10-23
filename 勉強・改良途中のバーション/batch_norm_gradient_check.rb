require 'numo/narray'
require './layers_2.rb'
require './mnist.rb'
require './multi_layer_net_extend'

# ƒf[ƒ^‚Ì“Ç‚İ‚İ
x_train, t_train, x_test, t_test = load_mnist(normalize: true, one_hot_label: true)

network = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100], output_size: 10, use_batchnorm: true)

x_batch = x_train[0..1, false]
t_batch = t_train[0..1, false]

grad_backprop = network.gradient(x: x_batch, t: t_batch)
grad_numerical = network.numerical_gradients(x: x_batch, t: t_batch)

grad_numerical.keys.each{|key|
    diff = (grad_backprop[key] - grad_numerical[key]).abs.mean
    print(key + ":" + diff.to_s + "\n")
}