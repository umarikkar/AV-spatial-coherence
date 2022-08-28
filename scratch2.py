import os
import numpy as np
import matplotlib.pyplot as plt


a = np.eye(16)
b = 1 - np.eye(16)

plt.figure(figsize=(3,2.5))
# plt.subplot(121)
plt.imshow(a)
plt.xlabel('visual representation')
plt.ylabel('audio representation')
# plt.title('(a) positive scores')
plt.colorbar()
# plt.subplot(122)
# plt.imshow(b)
# plt.title('(b) negative scores')
# plt.colorbar()
plt.show()

# plt.savefig('diag.pdf')

