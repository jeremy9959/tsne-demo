import numpy as np
import torch
from threading import Thread
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Button, Label
from bokeh.layouts import column, row, gridplot, widgetbox
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral10
from tornado import gen
from functools import partial
import high_dim
from util import dist_matrix

go_flag = False


def setup_graph(max_iter):

    tsne_source = ColumnDataSource(pd.DataFrame(columns=['x', 'y', 'color']))
    loss_source = ColumnDataSource(pd.DataFrame(columns=['iteration', 'loss']))

    TOOLTIPS = [("label", "@color")]

    doc = curdoc()

    p = figure(width=600, height=600, match_aspect=True, tooltips=TOOLTIPS, name="tsne_graph")
    q = figure(width=600, height=200, name="loss_graph", x_range=(0, max_iter),  title="Loss vs iterations", tools="save")

    r = p.circle(
        x="x",
        y="y",
        fill_color=linear_cmap("color", palette=Spectral10, low=0, high=9),
        source=tsne_source,
        legend="color",
        size=5,
        name="tsne_glyphs",
    )

    s = q.line(x='iteration', y='loss', source=loss_source, name="loss_glyphs")

    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"

    notice = Label(x=50, y=300, x_units='screen', y_units='screen', render_mode='css', text="Press the Go button when ready", name='notice', text_color='green' )
    p.add_layout(notice)
    
    button = Button(label="Go!", button_type="success", name="go_button")
    button.on_click(lambda: print("not ready yet"))
    doc.add_root(column(button, p, q))

    return doc


@gen.coroutine
def update_graph(doc, Y, labels=None):

    if labels is None:
        labels = np.zeros(Y.shape[0])

    data = Y.clone().detach().cpu().numpy()
    D = pd.DataFrame({"x": data[:, 0], "y": data[:, 1], "color": labels}).sort_values(
        "color"
    )
    doc.get_model_by_name("tsne_glyphs").data_source.stream(D,rollover=data.shape[0])


@gen.coroutine
def update_title(doc, i, max, loss):
    doc.get_model_by_name("go_button").label = "Iteration {}/{}     Loss {:.4f}".format(
        i, max, loss
    )
    doc.get_model_by_name("loss_glyphs").data_source.stream(pd.DataFrame({'iteration': [i],'loss': [loss]}))

@gen.coroutine
def wrap(doc):
    global go_flag

    doc.get_model_by_name("go_button").label = "Go!"
    doc.get_model_by_name("go_button").disabled = False
    go_flag = False


def KL_loss(P, Y, mask, device="cpu", l2=1):
    (n, d) = P.shape

    D = dist_matrix(Y)
    L2 = D.sum()
    num = 1.0 / (1.0 + D)
    # diag set to zero
    numU = torch.masked_select(num, mask)

    Q = numU / numU.sum()
    PU = torch.masked_select(P, mask)
    # Q = torch.max(Q, torch.tensor([1e-12], device=device))
    return (PU * (torch.log(PU / Q))).sum() + l2 * torch.log(L2)


def advance(doc, max_iter, P, mask, device):

    Y = torch.randn(n, 2, device=device, requires_grad=True)
    l2 = 1

    doc.add_next_tick_callback(partial(update_graph, doc, Y, labels))
    
    optimizer = torch.optim.SGD([Y], lr=500, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 1)
    optimizer.zero_grad()

    for iter in range(max_iter):

        E = KL_loss(P, Y, mask, device, l2=l2)
        E.backward()

        doc.add_next_tick_callback(partial(update_graph, doc, Y, labels))
        doc.add_next_tick_callback(partial(update_title, doc, iter, max_iter, E.item()))
        optimizer.step()
        scheduler.step()

        if (iter % 100) == 0:
            if l2 > 0.01:
                l2 = l2 * 0.5
            else:
                l2 = 0

            print(
                "Loss after {} steps is {}; l2 parameter is {}".format(
                    iter, E.item(), l2
                )
            )
        optimizer.zero_grad()

    doc.add_next_tick_callback(partial(wrap, doc))


def go_thread(doc, max_iter, P, mask, device):
    global go_flag

    print("in go_thread")
    if not go_flag:
        go_flag = True
        doc.get_model_by_name("notice").visible = False
        doc.get_model_by_name("loss_glyphs").data_source.stream(pd.DataFrame({'iteration': [np.nan], 'loss': [np.nan]}), rollover=1)
        doc.get_model_by_name("go_button").label = "Starting..."
        doc.get_model_by_name("go_button").disabled = True
        thread = Thread(target=partial(advance, doc, max_iter, P, mask, device))
        thread.start()


if torch.cuda.is_available():
    device = "cuda"
    print("Using cuda\n ")
else:
    device = "cpu"
    print("Using cpu\n")

labels = np.loadtxt("data/mnist2500_labels.txt")
X = np.loadtxt("data/mnist2500_X.txt")
X = torch.from_numpy(X).float().to(device)
n = X.shape[0]

max_iter = 1500
L = [[(i != j) for i in range(n)] for j in range(n)]
mask = torch.ByteTensor(L).to(device)

print('at setup_graph call')

doc = setup_graph(max_iter)


doc.get_model_by_name("tsne_graph").title.text = "tSNE on 2500 MNIST digits"
print('setup done')

P = high_dim.pmatrix(X, pca_dims=50, tolerance=1e-5, perplexity=20, device=device)
doc.get_model_by_name("go_button").on_click(partial(go_thread, doc, max_iter, P, mask,  device))
