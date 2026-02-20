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
return a.T # check with matrix[0] size == matrix2 size
np.transpose(a)
# Exponential
np.exp(val)
# Aggregate function
np_array.mean(axis = 1)
# Empty np array
 np.full((N, L), pad_value)
```

## Activation Function
### Sigmoid Function

Formula
### Logistic (Sigmoid) Function

[
f(x) = \frac{1}{1 + e^{-x}}
]

## Linear Algebra


## Optimization

```python

# Calculating gradients in logistic regression
dw = (X.T @ errors ) / n_samples # how much each feature contribute to erros
db = np.sum(errors) / n_samples # average error as bias is global for all features
# updating weights and bias
w -= lr * dw
b -= lr * db

```

