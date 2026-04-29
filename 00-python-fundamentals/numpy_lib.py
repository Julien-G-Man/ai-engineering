"""
Why NumPy instead of plain Python lists? 

All ML algorithms work on matrices (2D arrays of numbers). A plain Python list is flexible but slow — it 
stores pointers to objects. 
NumPy arrays store raw numbers in contiguous memory blocks, like C arrays. Operations on NumPy 
arrays are 100x faster than Python loops. 
scikit-learn, PyTorch, and TensorFlow all work with NumPy arrays (or are compatible with them). You 
must be comfortable with NumPy shape and slicing. 
"""

import numpy as np 


def creating_arrays():
	print("\n[Creating Arrays]")
	a = np.array([1, 2, 3, 4, 5])            # 1D array from list
	b = np.array([[1, 2, 3], [4, 5, 6]])     # 2D array (matrix)
	zeros = np.zeros((3, 4))                  # 3x4 matrix of zeros
	ones = np.ones((2, 3))                    # 2x3 matrix of ones
	eye = np.eye(3)                           # 3x3 identity matrix
	rand = np.random.randn(5, 3)              # 5x3 matrix, standard normal
	rng = np.random.default_rng(42)           # Reproducible random generator
	data = rng.uniform(0, 100, size=(100,))   # 100 uniform random numbers

	print("a:", a)
	print("b shape:", b.shape)
	print("zeros shape:", zeros.shape)
	print("ones shape:", ones.shape)
	print("eye shape:", eye.shape)
	print("rand shape:", rand.shape)
	print("data sample:", data[:5])


def shape_and_dimensions():
	print("\n[Shape And Dimensions]")
	x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	print(x.shape)    # (3, 3)
	print(x.ndim)     # 2
	print(x.size)     # 9
	print(x.dtype)    # int64 (platform dependent)


def reshaping():
	print("\n[Reshaping]")
	flat = np.array([1, 2, 3, 4, 5, 6])
	matrix = flat.reshape(2, 3)               # 2 rows, 3 cols
	row_vec = flat.reshape(1, -1)             # (1, 6)
	col_vec = flat.reshape(-1, 1)             # (6, 1)

	print("flat:", flat)
	print("matrix shape:", matrix.shape)
	print("row_vec shape:", row_vec.shape)
	print("col_vec shape:", col_vec.shape)


def indexing_and_slicing():
	print("\n[Indexing And Slicing]")
	x = np.random.randn(100, 5)  # 100 samples, 5 features

	print("first row:", x[0])
	print("row 0 col 2:", x[0, 2])
	print("first 10 of col 0:", x[:10, 0])
	print("rows 10-19 shape:", x[10:20, :].shape)
	print("first 5 rows of cols 0 and 2:\n", x[:5, [0, 2]])

	y = np.random.randint(0, 2, size=100)     # Labels aligned with x rows
	x_positive = x[y == 1]                    # Only rows where label is 1
	print("positive subset shape:", x_positive.shape)


def operations_and_aggregations():
	print("\n[Operations And Aggregations]")
	a = np.array([1, 2, 3, 4, 5])
	b = np.array([10, 20, 30, 40, 50])

	print(a + b)          # element-wise
	print(a * b)          # element-wise
	print(a + 100)        # broadcasting
	print(np.dot(a, b))   # dot product

	print(a.mean())       # mean
	print(a.std())        # standard deviation
	print(np.sum(b))      # sum
	print(np.argmax(b))   # index of max value


if __name__ == "__main__":
	creating_arrays()
	shape_and_dimensions()
	reshaping()
	indexing_and_slicing()
	operations_and_aggregations()