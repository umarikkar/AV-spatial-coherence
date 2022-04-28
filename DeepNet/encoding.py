
import numpy as np
import matplotlib.pyplot as plt

a = np.zeros((14,14))

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

num_cams = 22

plt.figure()

for idx in range(num_cams):
    k = 10
    arr = np.copy(a)
    
    idx_1d = (k**2) * idx // num_cams

    i = (idx_1d // k) + 2
    j = (idx_1d % k) + 2

    # arr[i-1:i+1, j-1:j+1] = 

    arr[i-1:i+2, j-1:j+2] = gkern(3)


    plt.subplot(4,6,idx+1)
    plt.axis('off')
    plt.imshow(arr)

plt.show()




