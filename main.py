import numpy as np
import torch
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Button
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral10
from tornado import gen
from functools import partial

go_flag = False
interrupt_flag = False

def Hbeta(D=torch.tensor([]), beta=None, device="cpu"):
    """
    Compute the perplexity and the P-row for a specific value of the
    precision of a Gaussian distribution.  Note that D is the i_th row
    of the pairwise distance matrix with the i_th entry deleted.
    """

    # Compute P-row and corresponding perplexity
    # at this point, P is the numerator of the conditional probabilities
    if beta is None:
        beta = torch.tensor([1.0], device=device)

    P = (-D * beta).exp()

    # P = np.exp(-D.copy() * beta)
    # sumP is the denominator, the normalizing factor
    sumP = torch.sum(P, dim=0, keepdim=True)

    # the entropy is the sum of p \log p which is P/sumP
    # Checking with the formula above, sumP = S_i and np.sum(D*P/sumP) is the dot
    # product of the distances with the probabilities

    H = sumP.log() + (D.dot(P) * beta) / sumP

    # now normalize P to be the actual probabilities and return them
    P = P / sumP

    return H, P


def dist_matrix(Y):
    """
    Return the matrix whose i,j entry is the euclidean distance between the rows of Y.
    """
    sum_Y = torch.sum(torch.mul(Y, Y), dim=1)
    num = -2.0 * (torch.mm(Y, Y.t()))
    num2 = torch.add(torch.add(num, sum_Y).t(), sum_Y)
    return num2


def x2p(X=torch.tensor([]), tol=1e-5, perplexity=30.0, device="cpu"):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    (n, d) = X.shape
    D = dist_matrix(X)
    P = torch.zeros(n, n, device=device)
    beta = torch.ones(n, 1, device=device)
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # prep for binary search on beta

        betamin = -np.inf
        betamax = np.inf

        # Compute the Gaussian kernel and entropy for the current precision
        # the first line drops the ith entry in row i from the pairwise distances
        # Hbeta in the second line expects this

        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
        (H, thisP) = Hbeta(Di, beta[i], device=device)

        # Evaluate whether the perplexity is within tolerance

        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision (via binary search)
            if Hdiff > 0:
                betamin = beta[i].clone().detach()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].clone().detach()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i], device=device)
            Hdiff = H - logU
            tries += 1

        # Set the final row of P, reinserting the missing spot as 0
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = thisP

    # Return final P-matrix

    return P


def pca(X=torch.tensor([]), no_dims=50):
    """
    Return the projection of X onto the no_dims largest eigenspaces
    """

    (n, d) = X.shape
    X = X - X.mean(0)
    e, v = torch.symeig(X.t().mm(X), eigenvectors=True)
    Y = X.mm(v[:, -no_dims:])

    return Y


def setup_graph(labels=None):

    if labels is None:
        labels = np.zeros([])

    D = pd.DataFrame(columns=["x", "y", "color"])
    D = D.sort_values("color")
    loss_graph_data = pd.DataFrame(columns=["iteration","loss"])

    TOOLTIPS = [("label", "@color")]

    source = ColumnDataSource(D)
    loss_source = ColumnDataSource(loss_graph_data)

    doc = curdoc()
    p = figure(width=600, height=600, match_aspect=True,  tooltips=TOOLTIPS, name="main")
    q = figure(width=600, height=600, name="loss_graph", x_range=(0,max_iter), title="Loss vs iterations")

    r = p.circle(
        x="x",
        y="y",
        fill_color=linear_cmap("color", palette=Spectral10, low=0, high=9),
        source=source,
        legend="color",
        size=5,
    )

    s = q.line(x='iteration', y='loss', source=loss_source)

    p.legend.location = "top_center"
    p.legend.orientation = "horizontal"

    button = Button(label="Go!", button_type="success", name="go_button")
    ds = r.data_source

    return doc, ds, p, button, q, loss_source


@gen.coroutine
def update_graph(ds, Y, labels=None):

    if labels is None:
        labels = np.zeros(Y.shape[0])
    data = Y.clone().detach().cpu().numpy()
    #    r = np.sqrt(np.max(data[:,0]**2 + data[:,1]**2))
    r = 1
    D = pd.DataFrame({"x": data[:, 0]/r, "y": data[:, 1]/r, "color": labels}).sort_values(
        "color"
    )
    ds.data = D


@gen.coroutine
def update_title(doc, loss_source, i, max, loss):
    doc.get_model_by_name("main").title.text = "Iteration {}/{}     Loss {:.4f}".format(
        i, max, loss
    )
    loss_source.stream(pd.DataFrame({'iteration':[i],'loss':[loss]}))
    
@gen.coroutine
def wrap(doc):
    global go_flag

    doc.get_model_by_name("go_button").visible = True
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


def advance(doc, ds, loss_source, max_iter, P, mask, device):


    
    Y = torch.randn(n, 2, device=device, requires_grad=True)
    l2 = 1
    doc.add_next_tick_callback(partial(update_graph, ds, Y, labels))
    optimizer = torch.optim.SGD([Y], lr=500, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 1)
    optimizer.zero_grad()

    for iter in range(max_iter):

        E = KL_loss(P, Y, mask, device, l2=l2)
        E.backward()

        doc.add_next_tick_callback(partial(update_graph, ds, Y, labels))
        doc.add_next_tick_callback(partial(update_title, doc, loss_source, iter, max_iter, E.item()))
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

    doc.add_next_tick_callback(partial(wrap,doc))



    
def go_thread(doc, ds, loss_source, max_iter, P, mask, device):
    global go_flag

    print("In go_thread with flag = {}".format(go_flag))
    if not go_flag:
        go_flag = True
        interrupt_flag = False
        doc.get_model_by_name("go_button").visible = False
        thread = Thread(target=partial(advance, doc, ds, loss_source,  max_iter, P, mask, device))
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
X = pca(X, 50)
print("calling x2p\n")
P = x2p(X, 1e-5, 20, device=device)
n = P.shape[0]
P = P + P.t()
# avoid zeros
P = torch.max(P, torch.tensor([1e-12], device=device))
P = P / P.sum()

max_iter = 1500
L = [[(i != j) for i in range(n)] for j in range(n)]
mask = torch.ByteTensor(L).to(device)
print("calling setup graph\n")
doc, ds, p, button, q, loss_source = setup_graph(max_iter)
p.title.text = "MNIST tsne example"
# Run iterations
button.on_click(partial(go_thread, doc, ds, loss_source, max_iter, P, mask, device))
doc.add_root(row(column(p, button), q))
