require 'numo/narray'
require 'numo/gnuplot'
require './mnist'
require './multi_layer_net_extend'
require './optimizers'
require './trainer'

x_train, t_train, x_test, t_test = load_mnist(normalize: true)

# �ߊw�K���Č����邽�߂ɁA�w�K�f�[�^���팸
x_train = x_train[0..299, false]
t_train = t_train[0..299, false]

# Dropuout�̗L���A���荇���̐ݒ� ========================
use_dropout = true  # Dropout�Ȃ��̂Ƃ��̏ꍇ��False��
dropout_ratio = 0.2
# ====================================================

network = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100, 100, 100, 100, 100],
                              output_size: 10, use_dropout: use_dropout, dropout_ratio: dropout_ratio)
trainer = Trainer.new(network: network, x_train: x_train, t_train: t_train, x_test: x_test, t_test: t_test,
                  epochs: 301, mini_batch_size: 100,
                  optimizer: 'sgd', optimizer_param: {'lr': 0.01}, verbose: true)
trainer.train()

#train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

=begin
# �O���t�̕`��==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
=end