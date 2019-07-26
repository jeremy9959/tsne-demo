# myapp.py
from functools import partial
from threading import Thread
import pandas as pd
import numpy as np
from random import random
import time
from numpy.random import normal
from bokeh.layouts import column
from bokeh.models import Button, Legend, LegendItem
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral10
from bokeh.events import ButtonClick
from tornado import gen
# load the data


Ysave = np.loadtxt("data/tsne_evolution.txt.gz",delimiter='\t')
labels = np.loadtxt("data/mnist2500_labels.txt.gz")
n=Ysave.shape[0]

# create a plot and style its properties
title = 'tSNE plot of 2500 MNIST digits'

doc = curdoc()

p = figure(height=600,width=600,x_range=(-150,150), y_range=(-150,150), toolbar_location=None,title=title)
D = pd.DataFrame(Ysave,columns=['x','y'])
colors = [Spectral10[int(i)] for i in labels]


initial_x = D['x'].iloc[-2500:]
initial_y = D['y'].iloc[-2500:]
initial_fill = colors

r = p.scatter(x=initial_x, y=initial_y, fill_color=initial_fill)
i = 0
ds = r.data_source

@gen.coroutine
def update(new_data,i):
    if i==0:
        ds.data==dict(x=[],y=[])
    else:
        ds.data = new_data

def advance():
    global i,D

    # BEST PRACTICE --- update .data in one step with a new dict
    while i<D.shape[0]:
        new_data = dict()
        new_data['x'] = D['x'].iloc[i:(i+2500)]
        new_data['y'] = D['y'].iloc[i:(i+2500)]
        new_data['fill_color']=colors
        #    new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
        #    new_data['text'] = ds.data['text'] + [str(i)]
        time.sleep(0.1)
        doc.add_next_tick_callback(partial(update,new_data,i))
        i += 2500
    
    

def go_thread():
    thread = Thread(target=advance)
    thread.start()
    
# add a button widget and configure with the call back

button = Button(label="Run the Animation")
button.on_click(go_thread)

# put the button and plot in a layout and add to the document
doc.add_root(column(p,button))

