import numpy as np
import torch
from threading import Thread
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from bokeh.document import without_document_lock
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Button, Label, Slider
from bokeh.models.ranges import Range1d
from bokeh.layouts import column, widgetbox, row, Spacer
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral10
from tornado import gen
from functools import partial
import high_dim
from util import dist_matrix, off_diag_mask

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

executor = ThreadPoolExecutor(max_workers=1)

labels = np.loadtxt("data/mnist2500_labels.txt")
X = np.loadtxt("data/mnist2500_X.txt")
X = torch.from_numpy(X).float().to(device)
max_iter = 1500
doc = curdoc()
doc.title = "tSNE Animation - Setting Up"

def setup_graph():
    global max_iter, device

    tsne_source = ColumnDataSource(pd.DataFrame(columns=["x", "y", "color"]))
    loss_source = ColumnDataSource(pd.DataFrame(columns=["iteration", "loss"]))

    TOOLTIPS = [("label", "@color")]

    title = "tSNE on 2500 MNIST digits using {} device".format(device)

    p = figure(
        width=600,
        height=600,
        match_aspect=True,
        tooltips=TOOLTIPS,
        name="tsne_graph",
        title=title,
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
    )
    p.title.align = "center"
    q = figure(
        width=600,
        height=200,
        name="loss_graph",
        x_range=(0, max_iter),
        y_range=(0, 15),
        title="Loss vs iterations",
        tools="save",
        title_location="below",
    )
    q.title.align = "center"

    r = p.circle(
        x="x",
        y="y",
        fill_color=linear_cmap("color", palette=Spectral10, low=0, high=9),
        source=tsne_source,
        legend="color",
        size=5,
        name="tsne_glyphs",
    )

    s = q.line(x="iteration", y="loss", source=loss_source, name="loss_glyphs", line_width=3, line_color="red")

    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"

    notice = Label(
        x=50,
        y=300,
        x_units="screen",
        y_units="screen",
        text="Thinking...",
        name="notice",
        text_color="red",
    )
    p.add_layout(notice)

    button = Button(
        label="Computing Distances in Feature Space",
        button_type="success",
        name="go_button",
        disabled=True,
        width=250
    )
    button.on_click(no_op)

    slider = Slider(
        start=500, end=5000, value=1500, step=100, title="Number of Iterations", name="iter_slider", width=250
    )

    slider.on_change("value", slider_callback)

    doc.add_root(column(p, row(Spacer(width=175), widgetbox(button,  slider),Spacer(width=175)), q))

    return 


def no_op():
    pass


def slider_callback(attr, old, new):
    global max_iter, doc

    max_iter = int(new)
    doc.get_model_by_name('loss_graph').x_range.end = max_iter
    
@gen.coroutine
def update_graph(Y, labels=None):
    global doc
    
    if labels is None:
        labels = np.zeros(Y.shape[0])

    data = Y.clone().detach().cpu().numpy()
    A = np.max(data)
    D = pd.DataFrame(
        {"x": data[:, 0] / A, "y": data[:, 1] / A, "color": labels}
    ).sort_values("color")
    doc.get_model_by_name("tsne_glyphs").data_source.stream(D, rollover=data.shape[0])


@gen.coroutine
def update_title(i,  loss):
    global doc, max_iter
    
    doc.get_model_by_name("go_button").label = "Iteration {}/{}     Loss {:.4f}".format(
        i, max_iter, loss
    )
    doc.get_model_by_name("loss_glyphs").data_source.stream(
        pd.DataFrame({"iteration": [i], "loss": [loss]})
    )


@gen.coroutine
def wrap():
    global doc
    
    doc.get_model_by_name("go_button").label = "Go!"
    doc.get_model_by_name("go_button").disabled = False
    doc.get_model_by_name("iter_slider").disabled = False
    doc.title = "tSNE Animation - Ready"

    
def KL_loss(P, Y,  l2=1):
    global device
    
    n = Y.shape[0]
    D = dist_matrix(Y)
    L2 = D.sum()
    num = 1.0 / (1.0 + D)
    mask = off_diag_mask(n, device)
    numU = torch.masked_select(num, mask)
    Q = numU / numU.sum()
    PU = torch.masked_select(P, mask)

    return (PU * (torch.log(PU / Q))).sum() + l2 * torch.log(L2)


def advance(P):
    global max_iter, doc, device

    Y = torch.randn(P.shape[0], 2, device=device, requires_grad=True)
    l2 = 2

    doc.add_next_tick_callback(partial(update_graph, Y, labels))

    optimizer = torch.optim.SGD([Y], lr=500, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 1)
    optimizer.zero_grad()

    for iter in range(max_iter):

        E = KL_loss(P, Y,  l2=l2)
        E.backward()

        doc.add_next_tick_callback(partial(update_graph, Y, labels))
        doc.add_next_tick_callback(partial(update_title, iter,  E.item()))
        optimizer.step()
        scheduler.step()

        if l2 > 0.01:
            l2 = l2 * 0.95
        else:
            l2 = 0

        optimizer.zero_grad()

    doc.add_next_tick_callback(wrap)


def go_thread(P):
    global max_iter, doc, device
    
    if not doc.get_model_by_name("go_button").disabled:
        doc.title = "tSNE Animation - Running"
        doc.get_model_by_name("notice").visible = False
        doc.get_model_by_name("loss_glyphs").data_source.stream(
            pd.DataFrame({"iteration": [np.nan], "loss": [np.nan]}), rollover=1
        )
        doc.get_model_by_name("go_button").label = "Starting..."
        doc.get_model_by_name("go_button").disabled = True
        doc.get_model_by_name("iter_slider").disabled = True
        thread = Thread(target=partial(advance, P))
        thread.start()


@gen.coroutine
def enable(P):
    global max_iter, device
    
    doc.get_model_by_name("go_button").on_click(
        partial(go_thread, P)
    )
    doc.get_model_by_name("go_button").disabled = False
    doc.get_model_by_name("iter_slider").disabled = False
    doc.get_model_by_name("go_button").label = "Go!"
    doc.get_model_by_name("notice").text = "Ready! Push the go button!"
    doc.title = 'tSNE Animation - Ready'

@gen.coroutine
@without_document_lock
def compute_feature_matrix(X):
    global max_iter, device
    
    P = yield executor.submit(
        partial(
            high_dim.pmatrix,
            X,
            pca_dims=50,
            tolerance=1e-5,
            perplexity=20,
            device=device,
        )
    )
    doc.add_next_tick_callback(partial(enable, P))


setup_graph()

# compute the distance matrix asynchronously so the server
# can draw the graph.  When this process finishes, the go button
# is enabled and the user can start the animation.

compute_feature_matrix(X)
