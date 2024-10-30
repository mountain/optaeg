# optaeg: combine neural networks and AEG theory

We tried different ways to combine neural networks and AEG theory and test them on different datasets.
Several models showed very good results with extremely fewer parameters:
- OptAEGV3 complex version with only 645 parameters to reach 98.2% accuracy on MNIST: [code](https://github.com/mountain/optaeg/blob/main/mnist_cmplx.py)
- OptAEGV3 complex version 14k parameters to reach 92.2% accuracy on FashionMNIST: [code](https://github.com/mountain/optaeg/blob/main/fashion_mnist_cmplx.py)

AEG stands for arithmetical expression geometry, which is a new theory studying the geometry of arithmetical expressions.
It opens a new optimization space for neural networks, and can be used to construct a new kind of neural network.
For the details of AEG theory, please refer to the draft paper:
* https://github.com/mountain/aeg-paper : Can arithmetical expressions form a geometry space?
* https://github.com/mountain/aeg-invitation : an invitation to AEG theory
* https://github.com/mountain/aeg-invitation/blob/main/slides2/aeg.pdf : introductory slides

