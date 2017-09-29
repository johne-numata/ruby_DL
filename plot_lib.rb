#################################################
##    plot_liblary
##    2017-9-11
#################################################
require 'numo/narray'
require 'numo/gnuplot'
require 'randomext'

###################################################################
###    Randomクラスの拡張　
###    乱数の拡張は　Randomext gemで対応。以下の拡張ができる
###		normal (Gaussian), lognormal, Cauthy, levy, exponential, Laplace, Rayleigh
###		Weibull, Gumbel, gamma, beta, power, Chi-square, F, t, Wald (inverse Gaussian)
###		Pareto, logistic, von Mises, Non-Central Chi-Square, Non-Central t, Planck
###		Bernoulli, binomial, Poisson, geometric, negative binomial, log series
###		Zipf-Mandelbrot, zeta
###　　　アルゴリズムの出典は
###  	Almost all algorithms are based on: 四辻哲章, "計算機シミュレーションのための確率分布乱数生成法", プレアデス出版 (2010)
###   使用例は
###		require 'randomext'
###		random_numbers = Array.new(100){ Random::DEFAULT.normal(0.0, 2.0) }
###		or
###		random = Random.new
###		random.poisson(10)
###################################################################

#######################################################################
#####	 		以下 Plot ユーティリティ
#####			t には下記フォーマットのデーが入る
#####				2D [dd[0], dd[1], {w:'lines', t: @name}]
#####				3D [dd[0], dd[1], dd[2], {w:'pm3d at s', t: @name}]	
#######################################################################
def make_plot_2d(t, title: "title", x_label:"x_data", y_label: "y_label",
				 file_name: nil, key: "right", grid: nil, size: nil, xrange: nil,
				 yrange: nil, zrange: nil, xtics: nil, ytics: nil, ztics: nil,
				 logscale: nil)
	tt = "plot t[0]"
	t[1..-1].each_index{|i|  tt += ",t[#{i+1}]" }
    Numo.gnuplot do
   	   	set terminal: "gif"   if file_name
   	   	set output: file_name   if file_name
   	   	set key: key 	if key			# "right" or "left" or "outside" etc
   	   	set nokey: ""	if key.nil?
   	   	set grid: ""	if grid
   	   	set size: size	if size			# "square" or "ratio 0.5"
   	   	set "autoscale"
   	   	set xrange: xrange	if xrange	# "[min:max]"
   	   	set yrange: yrange	if yrange
   	   	set xtics: xtics	if xtics	# "start,incr,end" or "(0,1,2,4,8)" + rotate
   	   	set ytics: ytics	if ytics
   	   	if logscale
  	   		unset "logscale"
   	   		set "logscale x"	if logscale =~ /[xX]/
   	   		set "logscale y"	if logscale =~ /[yY]/
  		else
  	   		unset "logscale"
  	   	end
  	   	set title: title
		set xlabel: x_label
		set ylabel: y_label
		eval(tt)
	    sleep (5)
    end
end

def make_plot_3d(t, title: "title", x_label:"x_data", y_label: "y_label", z_label: "z_label",
				 file_name: nil, key: "right", grid: nil, size: nil, xrange: nil,
				 yrange: nil, zrange: nil, xtics: nil, ytics: nil, ztics: nil,
				 logscale: nil, contour: nil, map: nil)
	tt = "splot t[0]"
	t[1..-1].each_index{|i|  tt += ",t[#{i+1}]" }
    Numo.gnuplot do
   	   	set terminal: "gif"   if file_name
   	   	set output: file_name   if file_name
   	   	set key: key 	if key			# "right" or "left" or "outside" etc
   	   	set nokey: ""	if key.nil?
   	   	set grid: ""	if grid
   	   	set size: size	if size			# "square" or "ratio 0.5"
   	   	set "autoscale"
   	   	set xrange: xrange	if xrange	# "[min:max]"
   	   	set yrange: yrange	if yrange
   	   	set zrange: zrange	if zrange
   	   	set xtics: xtics	if xtics	# "start,incr,end" or "(0,1,2,4,8)" + rotate
   	   	set ytics: ytics	if ytics
   	   	set ztics: ztics	if ztics
   	   	if logscale
  	   		unset "logscale"
   	   		set "logscale x"	if logscale =~ /[xX]/
   	   		set "logscale y"	if logscale =~ /[yY]/
   	   		set "logscale z"	if logscale =~ /[zZ]/
  		else
  	   		unset "logscale"
  	   	end
  	   	set title: title
		set xlabel: x_label
		set ylabel: y_label
		set zlabel: z_label
		set "hidden3d"
		set "dgrid3d 100,100,4"
		if map then set "pm3d map" else unset "view" end
		if contour then set contour: contour  else unset "contour" end	# "base" or "surface" or "both"
		eval(tt)
	    sleep (5)
    end
end
