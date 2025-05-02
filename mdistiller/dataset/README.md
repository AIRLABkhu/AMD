## Preparing Datasets

### NYUd V2

1. Download the dataset from:
    - [Labeled data](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat),
    - [Split info](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat).
    ```bash
    mkdir nyud_v2
    cd nyud_v2
    wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    wget http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    ```

2. Install dependency for python setup code
    ```bash
    pip install mat73
    ```

3. Run the setup code:
    ```python
    import mat73
    import scipy.io
    import h5py
    import numpy as np

    dataroot = '.'
    mat = mat73.loadmat(dataroot + 'nyu_depth_v2_labeled.mat')
    splits = scipy.io.loadmat(dataroot + 'splits.mat')
    
    with h5py.File('./nyud_v2.hdf5', 'w') as file:
        for split in ('train', 'test'):
            dset = file.create_group(split)
            image = mat['images'][..., splits['trainNdxs'].reshape(-1) - 1].transpose(3, 0, 1, 2)
            depth = mat['depths'][..., splits['trainNdxs'].reshape(-1) - 1].transpose(2, 0, 1)
            depth = np.clip(depth, 0, 10) / 10.0
            train.create_dataset('images', data=image)
            train.create_dataset('depths', data=depth)
    ```

4. Check MD5 sum (`b81c4d8db95d62487356fdc73ccd3728`)
