# Malis Loss

## What is this?
This program MALIS loss described in for 2D and 3D data:

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). *Maximin learning of image segmentation*. _Advances in Neural Information Processing Systems (NIPS) 2009_.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

## Performance of malis loss:
![image](https://github.com/HelmholtzAI-Consultants-Munich/Malis-Loss/blob/master/README_files/result.png)

This figure compares the performance of malis loss and cross entropy loss for segmenting mitochondrion. The two networks (UNet, please check /example/keras_example.py) were trained totally same except for the loss. The evaluation criteria (values shown on the figure) for segmentation is based on counting the number of correctly segmented mitochondria (Dice coefficient > 70 %), divided by the average of total number of ground-truth mitochondria and that of automatic segmented mitochondria (similar to the definition of the Dice coefficient, but number-based rather than pixel-based). The final average scores for these three cases are summarized in this table:


  | cross entropy loss  | malis loss |
-------| ------------- | ------------- |
Image 1 | 0.57  | 0.71  |
Image 2 | 0.68  | 0.80  |
Image 3 | 0.39  | 0.77  |

It can be shown that for this criteria, malis loss can achieved better performance than cross entropy loss. Malis loss performs better in separating nearby mitochondria and generating more clear boundaries, especially when the input data has many adjacent mitochondria (eg. Image 3).


## Installation:
```
./make.sh            (Building c++ extension only: run inside directory)
pip install .        (Installation as python package: run inside directory)
```


### Installation example in anaconda:
```
conda create -n malis python=3.7
conda install cython
conda install numpy
conda install gxx_linux-64
conda install -c anaconda boost
./make.sh
pip install .
```

## Example Usage:
See keras_example and pytorch_example in example folder for further detailed training(with UNet) examples
### Using Keras/Tensorflow (channel last):

#### 2D usage
```
import malis as m
from malis.malis_keras import malis_loss2d

model = ... (set the channel of output layer as 2)
model.compile(optimizer, loss = malis_loss2d)
```

#### 3D usage (please use batch size as 1)
```
import malis as m
from malis.malis_keras import malis_loss3d

model = ... (set the channel of output layer as 3)
model.compile(optimizer, loss = malis_loss3d)
```

### Using Pytorch: 
#### 2D usage
```
import malis
from malis.malis_torch import malis_loss2d
    
loss = malis_loss2d(seg_gt, pred_aff)
```
#### 3D usage (please use batch size as 1)
```
import malis
from malis.malis_torch import malis_loss3d
    
loss = malis_loss3d(seg_gt, pred_aff)
```

### Useful Functions of malis loss in python:
```
import malis as m
nhood = m.mknhood3d(): Makes neighbourhood structures
aff = m.seg_to_affgraph(seg_gt,nhood): Construct an affinity graph from a segmentation
seg = m.affgraph_to_seg(affinity,nhood): Obtain a segentation graph from an affinity graph
```
### Postprocessing:
The output of network should be affinity graphs, to obtain final segmentation graphs, threshold should be manully selected and than apply affgraph_to_seg functions. An example is like below:
```
import malis as m
import numpy as np

aff = .... # predicted affinity graph from trained model
aff = np.where(aff<threshold,0,1)
nhood = malis.mknhood3d(1)[:-1]  # or malis.mknhood3d(1) for 3d data prediction
seg = m.affgraph_to_seg(aff,nhood)
```
