# 3D-CNN-resnet-keras
## Residual version of the 3DCNN net.
- Cre_model is simple version
- To deeper the net uncomment bottlneck_Block and replace identity_Block to is

# Overview of resnet
- 1. In order to solve the problem of gradient degradation when training a very deep network, Kaiming He proposed the Resnet structure. Due to the use of the method of residual learning, the number of layers of the network has been greatly improved.
- 2. ResNet uses a shortcut to change the identity mapping of the unknown function H(x) that needs to be learned to become a function that approximates F(x)=H(x)-x.The author believes that the two expressions have the same effect, but the difficulty of optimization is not the same. The author assumes that the optimization of F(x) is much simpler than H(x). 
This idea is also derived from the residual vector coding in image processing. Through a reformulation, a problem is decomposed into multiple scale direct residual problems, which can well optimize the training effect.
- 3. ResNet proposes a BottleNeck structure for a deeper network with a number of layers greater than or equal to 50. This structure can reduce the time complexity of the operation.
- 4. There are two types of shortcuts in the network structure, Identity shortcut & Projection shortcut. 
Identity shortcut uses zero padding to ensure that its latitude is unchanged, while Projection shortcut has the following form y=F(x, Wi)+Wsx to match the dimension transformation.

## We used 3D version to achieve good results in the classification of 2D timing matrix.
