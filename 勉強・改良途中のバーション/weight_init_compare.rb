require 'numo/narray'
require 'numo/gnuplot'
require './mnist'
require './multi_layer_net_extend'
require './optimizers'


# 0:MNISTデータの読み込み==========
x_train, t_train, x_test, t_test = load_mnist(normalize: true)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1:実験の設定==========
weight_init_types = {'std=0.01': 0.01, 'Xavier': :sigmoid, 'He': :relu}
optimizer = SGD.new(lr: 0.01)

networks = {}
train_loss = {}
weight_init_types.each{|key, weight_type|
    networks[key] = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100, 100, 100],
                                  		output_size: 10, weight_init_std: weight_type)
    train_loss[key] = []
}

# 2:訓練の開始==========
(0..max_iterations - 1).each{|i|
    batch_mask = Numo::Int32.new(batch_size).rand(0, train_size)
    x_batch = x_train[batch_mask, false]
    t_batch = t_train[batch_mask, false]
    
    weight_init_types.keys.each{|key|
        grads = networks[key].gradient(x: x_batch, t: t_batch)
        optimizer.update(params: networks[key].params, grads: grads)
    
        loss = networks[key].loss(x: x_batch, t: t_batch)
        train_loss[key] << loss
    }
    if i % 100 == 0
        puts ("===========" + "iteration:" + "#{i}" + "===========")
        weight_init_types.keys.each{|key|
            loss = networks[key].loss(x: x_batch, t: t_batch)
            print(key.to_s + ":" + "#{loss}\n")
        }
	end
}
