# ml-core
ML algorithms from scratch. From gradients to transformers, learn the math with interactive visualizations and implement research papers

## Numpy Functions

### simple functions

```python
# Convert normal list to numpy array
a = np.array(a)
# Dot product
c = a @ b
# find length
print(a.shape[1], a.shape[0]) # here i represents items num
# Transpose
return a.T 
np.transpose(a)
# Exponential
np.exp(val)
```

## Activation Function

### Sigmoid Function

Formula
$ f(x) = \frac{1}{1 + e^{-x}} $


## Linear Algebra

### Eigen vector and Eigen values

Eigenvalue: The factor by which a matrix stretches an eigenvector without changing its direction.

$ Av=λv $

* v = eigenvector
* λ (lambda) = eigenvalue

* Meaning: The matrix scales the vector without changing its direction.
* Intuition: Think of pressing a rubber sheet.
    - Most arrows on the sheet: rotate, change direction
    - But some arrows: stay pointing the same way, just stretch or shrink

* Those arrows are eigenvectors.
* The stretch amount is the eigenvalue.

Ex application:
* PCA (find directions of maximum variance), helps analyze exploding / vanishing gradients and Eigenfaces (face recognition) uses eigenvectors. 

## Optimization

```python

# Calculating gradients in logistic regression
dw = (X.T @ errors ) / n_samples # how much each feature contribute to erros
db = np.sum(errors) / n_samples # average error as bias is global for all features
# updating weights and bias
w -= lr * dw
b -= lr * db

```

