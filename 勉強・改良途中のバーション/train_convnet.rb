require 'numo/narray'
require 'numo/gnuplot'
require './mnist'
require './simple_convnet'
require './trainer'

# データの読み込み
x_train, t_train, x_test, t_test = load_mnist(flatten:false)

# 処理に時間のかかる場合はデータを削減 
x_train, t_train = x_train[0...5000, false], t_train[0...5000, false]
x_test, t_test = x_test[0...1000, false], t_test[0...1000, false]

max_epochs = 20

network = SimpleConvNet.new(input_dim:[1,28,28], 
                        conv_param:{'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size:100, output_size:10, weight_init_std:0.01)

trainer = Trainer.new(network:network, x_train:x_train, t_train:t_train, x_test:x_test, t_test:t_test,
                  epochs:max_epochs, mini_batch_size:100,
                  optimizer:'Adam', optimizer_param:{'lr': 0.001},
                  evaluate_sample_num_per_epoch:1000)
#          p x_train
#          p trainer
trainer.train()

# パラメータの保存
#network.save_params("params.pkl")
#print("Saved Network Parameters!")

# グラフの描画
=begin
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
=end
