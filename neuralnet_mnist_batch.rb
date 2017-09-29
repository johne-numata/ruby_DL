#! ruby -Ks
require 'tk'
require 'tk/image'
require 'json'
require 'numo/narray'
require './mnist.rb'
require './plot_lib'
require './functions'

start_time = Time.now

def get_data()
    x_train, t_train, x_test, t_test = load_mnist(normalize:true, flatten:true, one_hot_label:false)
    return x_test, t_test
end

def init_network()
	nw = JSON.load(File.read('sample_weight.json'))
	network = {}
	nw.each do |k, v|
    	network[k.to_sym] = Numo::DFloat[*v]
	end
    return network
end

def predict(network, x)
    w1, w2, w3 = network[:w1], network[:w2], network[:w3]
    b1, b2, b3 = network[:b1], network[:b2], network[:b3]

	a1 = x.dot(w1) + b1
	z1 = sigmoid(a1)
	a2 = z1.dot(w2) + b2
	z2 = sigmoid(a2)
	a3 = z2.dot(w3) + b3
	return softmax(a3)
end

x, t = get_data()
network = init_network()
accuracy_cnt = 0
=begin
for i in 0..x.shape[0] - 1
    y = predict(network, x[i, true])
    k = y.max_index		 # 最も確率の高い要素のインデックスを取得
    accuracy_cnt += 1  if k == t[i]
end
=end

batch_size = 1000
for i in 0.step(x.shape[0] - 1, batch_size)
	y_batch = predict(network, x[i..i + batch_size - 1, true])
	k = y_batch.max_index(1) % y_batch[0, true].shape
  	accuracy_cnt += k.eq(t[i..i + batch_size - 1]).count_true
end

puts "Accuracy: #{accuracy_cnt.to_f / x.shape[0]}"
puts "処理時間　#{Time.now - start_time}"
