require 'numo/narray'
require './optimizers'

class Trainer
    # ニューラルネットの訓練を行うクラス
    attr_reader :train_acc_list, :test_acc_list
    def initialize(network: , x_train: , t_train: , x_test: , t_test:,
                 epochs: 20, mini_batch_size: 100,
                 optimizer: 'SGD', optimizer_param: {'lr':0.01}, 
                 evaluate_sample_num_per_epoch: nil, verbose: true)
        @network = network
        @verbose = verbose
        @x_train = x_train
        @t_train = t_train
        @x_test = x_test
        @t_test = t_test
        @epochs = epochs
        @batch_size = mini_batch_size
        @evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':"SGD", 'momentum':"Momentum", 'nesterov':"Nesterov",
                                'adagrad':"AdaGrad", 'rmsprpo':"RMSprop", 'adam':"Adam"}
        @optimizer = Object.const_get(optimizer_class_dict[optimizer.downcase.to_sym]).new(**optimizer_param)
        
        @train_size = x_train.shape[0]
        @iter_per_epoch = [@train_size / mini_batch_size, 1].max
        @max_iter = (epochs * @iter_per_epoch).to_i
        @current_iter = 0
        @current_epoch = 0
        
        @train_loss_list = []
        @train_acc_list = []
        @test_acc_list = []
	end

    def train_step()
        batch_mask = Numo::DFloat.new(@batch_size).rand(@train_size - 1)
        x_batch = @x_train[batch_mask, false]
        t_batch = @t_train[batch_mask, false]
        
        grads = @network.gradient(x: x_batch, t: t_batch)
        @optimizer.update(params: @network.params, grads: grads)
        
        loss = @network.loss(x: x_batch, t: t_batch)
        @train_loss_list << (loss)
        puts("train loss:" + loss.to_s) if @verbose
        
        if @current_iter % @iter_per_epoch == 0
            @current_epoch += 1
            
            x_train_sample, t_train_sample = @x_train, @t_train
            x_test_sample, t_test_sample = @x_test, @t_test
            if @evaluate_sample_num_per_epoch
                t = @evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = @x_train[0...t, false], @t_train[0...t, false]
                x_test_sample, t_test_sample = @x_test[0...t, false], @t_test[0...t, false]
            end
            train_acc = @network.accuracy(x: x_train_sample, t: t_train_sample)
            test_acc = @network.accuracy(x: x_test_sample, t: t_test_sample)
            @train_acc_list << train_acc
            @test_acc_list << test_acc

            puts("=== epoch:" + @current_epoch.to_s + ", train acc:" + train_acc.to_s + ", test acc:" + test_acc.to_s + " ===") if @verbose
        end
        @current_iter += 1
	end

    def train()
        @max_iter.times{ train_step() }
        test_acc = @network.accuracy(x: @x_test, t: @t_test)
        if @verbose
            puts("=============== Final Test Accuracy ===============")
            puts("test acc:" + test_acc.to_s)
        end
	end
end
