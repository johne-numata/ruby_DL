require 'numo/narray'
require 'numo/gnuplot'
require './mnist'
require './multi_layer_net_extend'
require './optimizers'

x_train, t_train, x_test, t_test = load_mnist(normalize: true)

# 過学習を再現するために、学習データを削減
x_train = x_train[0..299, false]
t_train = t_train[0..299, false]

# weight decay（荷重減衰）の設定 =======================
#weight_decay_lambda = 0 # weight decayを使用しない場合
weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100, 100, 100, 100, 100], output_size: 10,
                       		 weight_decay_lambda: weight_decay_lambda)
optimizer = SGD.new(lr: 0.1)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = [train_size / batch_size, 1].max.to_i
epoch_cnt = 0

1000000000.times do |i|
    batch_mask = Numo::Int32.new(batch_size).rand(0, train_size - 1)
    x_batch = x_train[batch_mask, false]
    t_batch = t_train[batch_mask, false]

    grads = network.gradient(x: x_batch, t: t_batch)
    optimizer.update(params: network.params, grads: grads)

    if i % iter_per_epoch == 0
        train_acc = network.accuracy(x: x_train, t: t_train)
        test_acc = network.accuracy(x: x_test, t: t_test)
        train_acc_list << train_acc
        test_acc_list << test_acc

        print("epoch:" + epoch_cnt.to_s + ", train acc:" + train_acc.to_s + ", test acc:" + test_acc.to_s + "\n")

        epoch_cnt += 1
        if epoch_cnt >= max_epochs
            break
        end
	end
end

=begin
# 3.グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
=end
