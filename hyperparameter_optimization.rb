require 'numo/narray'
require 'numo/gnuplot'
require './mnist'
require './multi_layer_net_extend'
require './trainer'

x_train, t_train, x_test, t_test = load_mnist(normalize: true)

# 高速化のため訓練データの削減
x_train = x_train[0..499, false]
t_train = t_train[0..499, false]

# 検証データの分離
validation_rate = 0.20
validation_num = x_train.shape[0] * validation_rate
shuffle_seq = Numo::Int32.new(x_train.shape[0]).seq.to_a.shuffle
x_train = x_train[shuffle_seq, false]
t_train = t_train[shuffle_seq, false]
@x_val = x_train[0..validation_num - 1, false]
@t_val = t_train[0..validation_num - 1, false]
@x_train = x_train[validation_num..-1, false]
@t_train = t_train[validation_num..-1, false]


def _train(lr:, weight_decay:, epocs: 50)
    network = MultiLayerNet.new(input_size: 784, hidden_size_list: [100, 100, 100, 100, 100, 100],
                            output_size: 10, weight_decay_lambda: weight_decay)
    trainer = Trainer.new(network: network, x_train: @x_train, t_train: @t_train, x_test: @x_val, t_test: @t_val,
                      epochs: epocs, mini_batch_size: 100,
                      optimizer: 'sgd', optimizer_param: {'lr': lr}, verbose: false)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list
end

# ハイパーパラメータのランダム探索======================================
optimization_trial = 100
results_val = {}
results_train = {}
optimization_trial.times{
    # 探索したハイパーパラメータの範囲を指定===============
    weight_decay = 10 ** rand(-8..-4.0)
    lr = 10 ** rand(-6..-2.0)
    # ================================================

    val_acc_list, train_acc_list = _train(lr: lr, weight_decay: weight_decay)
    puts("val acc:" + val_acc_list[-1].to_s + " | lr:" + lr.to_s + ", weight decay:" + weight_decay.to_s)
    key = "lr:" + lr.to_s + ", weight decay:" + weight_decay.to_s
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list
}
# グラフの描画========================================================
puts("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
#col_num = 5
#row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

# for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
results_val.sort{|(k1, v1), (k2, v2)| v1[-1] <=> v2[-1]}.reverse_each do |key, val_acc_list|
    puts("Best-" + (i+1).to_s + "(val acc:" + val_acc_list[-1].to_s + ") | " + key)

#    plt.subplot(row_num, col_num, i+1)
#    plt.title("Best-" + str(i+1))
#    plt.ylim(0.0, 1.0)
#    if i % 5: plt.yticks([])
#    plt.xticks([])
#    x = np.arange(len(val_acc_list))
#    plt.plot(x, val_acc_list)
#    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num
        break
	end
#plt.show()
end
