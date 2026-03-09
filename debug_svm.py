import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Generatin dummy data...")
X = np.random.rand(100, 64*64*3).astype('float32') # Flattened
y = np.random.randint(0, 3, 100)

print("Initializing SVM...")
model = SVC(kernel='linear')

print("Training SVM...")
start = time.time()
model.fit(X, y)
print(f"Training done in {time.time() - start:.4f}s")

print("Predicting...")
preds = model.predict(X)
print("Done.")
