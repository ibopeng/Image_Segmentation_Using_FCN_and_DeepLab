import numpy as np
import os
import scipy.io as spio

flist = os.listdir('./test/SegmentationClass_1D_255')

if not os.path.exists('./test/SegmentationClass_1D_255_npy'):
    os.makedirs('./test/SegmentationClass_1D_255_npy')

for fl in flist:
    mat = spio.loadmat('./test/SegmentationClass_1D_255/' + fl)
    im = mat['im']
    np.save('./test/SegmentationClass_1D_255_npy/' + fl[:-4], im)
    print(np.unique(im))