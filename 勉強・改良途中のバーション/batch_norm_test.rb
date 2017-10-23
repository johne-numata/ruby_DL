require 'numo/narray'
require 'numo/gnuplot'
require './mnist'
require './multi_layer_net_extend'
require './optimizers'

x_train, t_train, x_test, t_test = load_mnist(normalize: true)

# 学習データを削減
@x_train = x_train[0..999, false]
@t_train = t_train[0..999, false]

@max_epochs = 20
@train_size = @x_train.shape[0]
@batch_size = 100
@learning_rate = 0.01


def train(weight_init_std:)
    bn_network = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100, 100, 100, 100], output_size: 10, 
                                    weight_init_std: weight_init_std, use_batchnorm: true)
    network = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100, 100, 100, 100], output_size: 10,
                                weight_init_std: weight_init_std)
    optimizer = SGD.new(lr: @learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = [@train_size / @batch_size, 1].max.to_i
    epoch_cnt = 0
    
    for i in 0..1000000000
        batch_mask = Numo::Int32.new(@batch_size).rand(0, @train_size - 1)
        x_batch = @x_train[batch_mask, false]
        t_batch = @t_train[batch_mask, false]
    
        [bn_network, network].each{|_network|
#        	p _network.use_batchnorm
            grads = _network.gradient(x: x_batch, t: t_batch)
#            p x_batch
#            p t_batch
#            puts "grads ok\n"
            optimizer.update(params: _network.params, grads: grads)
#            puts "optimizer OK\n"
    	}
#    	puts "okok\n"
        if i % iter_per_epoch == 0
            train_acc = network.accuracy(x: @x_train, t: @t_train)
            bn_train_acc = bn_network.accuracy(x: @x_train, t: @t_train)
            train_acc_list << train_acc
            bn_train_acc_list << bn_train_acc

            print("epoch:" + epoch_cnt.to_s + " | " + train_acc.to_s + " - " + bn_train_acc.to_s + "\n")

            epoch_cnt += 1
            if epoch_cnt >= @max_epochs
                break
            end
        end
    end

    return train_acc_list, bn_train_acc_list
end

# 3.グラフの描画==========
weight_scale_list = Numo::DFloat.new(16).store(0.step(-4.0, -4.0/15).map{|a| 10**a})
#weight_scale_list = Numo::DFloat.new(3).store(-1.step(-4.0, -3.0/2).map{|a| 10**a})

train_acc_list = []; bn_train_acc_list = []
weight_scale_list.to_a.each_index{|i|
    print( "\n============== " + (i+1).to_s + "/16" + " ==============\n")
    non_bn, bn = train(weight_init_std: weight_scale_list[i])
	train_acc_list << non_bn; bn_train_acc_list << bn
}

x = (0..@max_epochs-1).to_a
Numo.gnuplot do
  set multiplot:{layout: [4,4]}
  set yrange: "[0:1]"
  weight_scale_list.size.times{|i|
	set title: "weight_init_std = #{weight_scale_list[i]}"
    set size: [0.2,0.2]
    plot x, train_acc_list[i], { w: :lines, t: 'without BN', lc_rgb: 'blue', lw: 1 },
         x, bn_train_acc_list[i], { w: :lines, t: 'with BN', lc_rgb: 'green', lw: 1 }
  }
  sleep 15
end

