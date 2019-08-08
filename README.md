![tsne_plot](https://github.com/jeremy9959/tsne-demo/blob/master/png/tsne_plot.png)

To Do:

- use builtin torch primitives and automatic differentiation for optimization step. (seems to work)
- pass a parameter dictionary and set parameters "all at once":
	- perplexity tolerance
	- perplexity goal
	- PCA dims
	- device
	
	- initial l2 parameter
	- l2 decay factor
	- l2 decay period

	- learning rate (and perhaps learning rate decay)
	- momentum
	- max_iter
  
- port to a bokeh server app 
- change the legend structure so you can click to highlight particular values of the legend; move the legend otuside the plot body
- add widgets to report/adjust parameters 
- allow for other data sets
- deploy

Along the way, refactor the code to clean things up.

