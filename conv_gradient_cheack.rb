require 'numo/narray'
require './Simple_ConvNet'

network = SimpleConvNet.new(input_dim: [1, 10, 10], 
                        conv_param: {'filter_num':10, 'filter_size':3, 'pad':0, 'stride':1},
                        hidden_size: 10, output_size: 10, weight_init_std: 0.01)

x = Numo::DFloat.new(100).rand.reshape(1, 1, 10, 10)
t = Numo::Int32.new(1,1).fill(1)

grad_num = network.numerical_gradients(x: x, t: t)
grad = network.gradient(x: x, t: t)

grad_num.each{|key, value| puts(key + ":" + (grad_num[key] - grad[key]).abs.mean.to_s) }