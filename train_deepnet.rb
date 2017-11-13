require 'numo/narray'
require 'numo/gnuplot'
require './mnist'
require './deep_convnet'
require './trainer'
require 'fileutils'

x_train, t_train, x_test, t_test = load_mnist(flatten:false)

# 処理に時間のかかる場合はデータを削減 
# x_train, t_train = x_train[0...5000, false], t_train[0...5000, false]
# x_test, t_test = x_test[0...1000, false], t_test[0...1000, false]

max_epochs = 1

network = DeepConvNet.new  
trainer = Trainer.new(network:network, x_train:x_train, t_train:t_train, x_test:x_test, t_test:t_test,
                  		epochs:max_epochs, mini_batch_size:100,	optimizer:'Adam', optimizer_param:{lr:0.001},
                  		evaluate_sample_num_per_epoch:1000)

network.load_params("deep_convnet_params.dump") if File.exist?("deep_convnet_params.dump")

trainer.train()

# パラメータの保存
network.save_params("deep_convnet_params.dump")
puts("Saved Network Parameters!")
