import torch


def dist_matrix(Y):
    """
    Return the matrix whose i,j entry is the euclidean distance between the rows of Y.
    """
    sum_Y = torch.sum(torch.mul(Y, Y), dim=1)
    num = -2.0 * (torch.mm(Y, Y.t()))
    num2 = torch.add(torch.add(num, sum_Y).t(), sum_Y)
    return num2


