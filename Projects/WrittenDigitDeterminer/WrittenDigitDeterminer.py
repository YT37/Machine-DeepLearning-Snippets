from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

digits = load_digits()
data = scale(digits.data)
y = digits.target

samples, features = data.shape