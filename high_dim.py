import numpy as np
import torch
from util import dist_matrix

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


def pca(X=torch.tensor([]), pca_dims=50):
    """
    Return the projection of X onto the no_dims largest eigenspaces
    """

    (n, d) = X.shape
    X = X - X.mean(0)
    e, v = torch.symeig(X.t().mm(X), eigenvectors=True)
    Y = X.mm(v[:, -pca_dims:])

    return Y


def pmatrix(X, pca_dims=50, tolerance=1e-5, perplexity=20, device='cpu'):
    X = pca(X, pca_dims)
    P = x2p(X, 1e-5, 20, device=device)
    P = P + P.t()
    P = torch.max(P, torch.tensor([1e-12], device=device))
    P = P / P.sum()
    return P
