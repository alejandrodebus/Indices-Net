# Indices-Net

Paper: Direct Multitype Cardiac Indices Estimation via Joint Representation and Regression Learning.
https://arxiv.org/abs/1705.09307

Own implementation of the Indices-Net model proposed in [Xue et al, 2017]. This code was written as a baseline to compare with my own model presented in STACOM 2018 [Debus and Ferrante, 2018]. If you found this code useful in your publication, please cite both works as follows:

[Debus and Ferrante, 2018] " Left ventricle quantification through spatio-temporal CNNs" Alejandro Debus and Enzo Ferrante. MICCAI STACOM 2018. https://arxiv.org/abs/1808.07967

[Xue et al, 2017] "Direct Multitype Cardiac Indices Estimation via Joint Representation and Regression Learning." Wufeng Xue, Ali Islam, Mousumi Bhaduri, Shuo Li. IEEE TMI 2017. https://arxiv.org/abs/1705.09307


## General architecture of Indices-Net ##

Joint learning of representation and regression model for multitype cardiac indices estimation by Indices-Net. Two tightly coupled networks are included: DCAE for image representation and CNN for multiple indices regression. The two parts are learned with iterated forward propagation (solid arrows) and backward propagation (dashed arrows) to maximally benefit each other.

![alt text](https://github.com/alejandrodebus/IndicesNet/blob/master/imgs_architecture/indices_net.png)



## DCAE (Deep Convolutional AutoEncoder) ##

Architecture of DCAE, which constitutes two mirrored subparts: the discriminitive convolution layers and the generative deconvolution layers. With both of them, a mapping between the input and the output of DCAE is built.

![alt text](https://github.com/alejandrodebus/IndicesNet/blob/master/imgs_architecture/dcae.png)



## Index-specific feature extraction and regression ##

Index-specific feature extraction (first two layers) and regression (third layer) for multiple cardiac indices estimation.

![alt text](https://github.com/alejandrodebus/IndicesNet/blob/master/imgs_architecture/conv_reg.png)

## Configuration of the Network ##

The following table shows the configuration of the network

![alt text](https://github.com/alejandrodebus/IndicesNet/blob/master/imgs_architecture/network_config.png)
