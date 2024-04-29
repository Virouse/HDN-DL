import numpy as np

class_probs = np.bincount([0, 1, 1, 2, 5], weights=[1, 1, 1, 1, 1])
y_predict = np.argmax(class_probs)
