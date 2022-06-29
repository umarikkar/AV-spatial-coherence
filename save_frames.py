import os
from core.helper_fns import SaveVideoFrames

base_path = os.getcwd()
data_path = os.path.join(base_path, 'data', 'RJDataset')


SaveVideoFrames(data_path)