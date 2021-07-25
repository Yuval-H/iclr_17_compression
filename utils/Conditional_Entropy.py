import numpy as np


def compute_conditional_entropy(z1, z2, bins_for_hist=2):
    # Compute Uncertainty-Coefficient : according to "Numerical Recipes in C" p.635
    # U(z1,z2) =
    #
    hist2d = np.histogram2d(z1, z2, bins=bins_for_hist)

    # Get row and column totals:
    sum_i = np.sum(hist2d[0], axis=1)
    sum_j = np.sum(hist2d[0], axis=0)
    sum_hist = np.sum(hist2d[0])

    # Calc entropy of z1 and z2
    p1 = sum_i / sum_hist
    p2 = sum_j / sum_hist
    p1 = p1[p1 != 0]
    p2 = p2[p2 != 0]
    h_z1 = np.sum(-p1*np.log2(p1))
    h_z2 = np.sum(-p2*np.log2(p2))

    # Calc total entropy
    p_12 = hist2d[0] / sum_hist
    p_12 = p_12[p_12 != 0]
    h_total = np.sum(-p_12*np.log2(p_12))

    # Compute the (avg) conditional using : H(x,y) = H(x) + H(y|x)
    mean_conditional_entropy = h_total - 0.5*(h_z1 + h_z2)
    # Compute the Uncertainty-Coefficient
    U_z1_z2 = 2 * (h_z1 + h_z2 - h_total) / (h_z1 + h_z2)

    return U_z1_z2, mean_conditional_entropy
