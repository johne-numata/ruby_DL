require 'numo/narray'
require 'numo/gnuplot'


def sigmoid(x)
    return 1 / (1 + Numo::DFloat::Math.exp(-x))    
end


def relu(x)
	y = x.copy
	y[y<0] = 0
    return y
end


def tanh(x)
    return Numo::NMath.tanh(x)
end


input_data = Numo::DFloat.new(1000, 100).rand_norm  # 1000個のデータ
node_num = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5  # 隠れ層が5層
activations = {}  # ここにアクティベーションの結果を格納する

x = input_data

(0..hidden_layer_size - 1).each{ |i|
    x = activations[i-1]  if i != 0

    # 初期値の値をいろいろ変えて実験しよう！
    # w = Numo::DFloat.new(node_num, node_num).rand_norm * 1.0
    # w = Numo::DFloat.new(node_num, node_num).rand_norm * 0.01
     w = Numo::DFloat.new(node_num, node_num).rand_norm * Math.sqrt(1.0 / node_num)
    # w = Numo::DFloat.new(node_num, node_num).rand_norm * Math.sqrt(2.0 / node_num)

    a = x.dot(w)

    # 活性化関数の種類も変えて実験しよう！
     z = sigmoid(a)
    # z = relu(a)
    # z = tanh(a)

    activations[i] = z
}

# ヒストグラムを描画
Numo.gnuplot do
  set multiplot:{layout: [1,5]}
  set xrange: "[0:10]"
  activations.each{|i, a|
	title = (i+1).to_s + "-layer"
	x = []
	(0..9).each{|i| x << Numo::Int16.cast(a.flatten * 10).eq(i).count_true }
	plot x, {s: "frequency", w: "boxes", t: title}
  }
  sleep 5
end
