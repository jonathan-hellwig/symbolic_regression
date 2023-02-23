# Symbolic regression

In this project, building on the ideas from Martius et al.[^1] my friend [@SilenceInTheBox](https://github.com/SilenceInTheBox) and I implemented a neural network that is capable of learning symbolic expression. In `layer.py` we implement symbolic and linear layers that compute the forward pass using the [SciPy](https://scipy.org/) library. This allows us to obtain analytical expressions from the trained network and conveniently simply them.
In `symbolic_regression.ipynb`, we train a simple network of two symbolic and three linear layers to obtain the polynomial $x^4- 2 x^2 + 1$. The output of the trained neural network simplifies to $0.986237x^4−0.00102282x^3−1.95504x^2+0.00101377x+0.992526$.

[^1]: [Extrapolation and learning equations](https://arxiv.org/abs/1610.02995v1)
