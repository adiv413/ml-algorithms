import numpy as np
import csv

def most(x):
	most = x[0]
	pos = 0
	for i in range(len(x)):
		if(x[i] > most):
			most = x[i]
			pos = i

	return pos

def l_relu(x):
	return max(0.01 * x, x)

def l_relu_dx(x):
	return 1 if x > 0 else 0.01

def forward(val, w1, w2, b1, b2): 
	x = np.matmul(w1, val) + b1
	for i in range(len(x)):
		x[i][0] = l_relu(x[i][0])
	y = np.matmul(w2, x) + b2

	for i in range(len(y)):
		y[i][0] = l_relu(y[i][0])
	z = softmax(y)  
	return [most(z), x, y] 

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def backprop(actual, x0, x1, x2, w1, w2, b1, b2, alpha):
	y = np.array([[0.0] for i in range(10)])
	y[actual] = 1
	r = np.matmul(w2, x1)
	for i in range(len(r)):
		r[i][0] = l_relu_dx(r[i][0])
	d = np.subtract(x2, y) * r
	w2_ret = np.subtract(w2, alpha * np.matmul(d, x1.T))
	b2_ret = np.subtract(b2, alpha * d)

	r = np.matmul(w1, np.array([x0]).T)
	for i in range(len(r)):
		r[i][0] = l_relu_dx(r[i][0])
	d = np.matmul(w2_ret.T, d) * r
	w1_ret = np.subtract(w1, alpha * np.matmul(d, np.array([x0])))
	b1_ret = np.subtract(b1, alpha * d)

	return [w1_ret, w2_ret, b1_ret, b2_ret]


 
def main():
	train = []
	test = []
	pos = 0
	alpha = 0.01
	n = 50 #number of neurons in the hidden layer

	with open("train.csv", newline = '') as trainingData:

		csvreader = csv.reader(trainingData)
		train = [list(map(int, x)) for x in csvreader] 

		for i in train:
			for j in range(1, len(i)):
				i[j] = i[j]/255

	with open("test.csv", newline = '') as testingData:
		next(testingData)

		csvreader2 = csv.reader(testingData)
		test = [list(map(int, x)) for x in csvreader2]

		for i in test:
			for j in range(1, len(i)):
				i[j] = i[j]/255

	w1 = np.array([[np.random.randn() * np.sqrt(2/(len(train[0]) - 1)) for i in range(len(train[0]) - 1)] for i in range(n)])
	w2 = np.array([[np.random.randn() * np.sqrt(2/n) for i in range(n)] for i in range(10)])
	b1 = np.array([[0] for i in range(n)])
	b2 = np.array([[0] for i in range(10)])

	pos = 0

	while pos < 11800:

		ans, first, second = 0, 0, 0

		for k in range(5):

			ret = forward(np.array([[i] for i in train[pos][1:]]), w1, w2, b1, b2)
			ans = ret[0]
			first = ret[1]
			second = ret[2]
			pos += 1

		b_ret = backprop(train[pos][0], np.array(train[pos][1:]), first, second, w1, w2, b1, b2, alpha)
		w1 = b_ret[0]
		w2 = b_ret[1]
		b1 = b_ret[2]
		b2 = b_ret[3]

	final = [forward(np.array([[i] for i in test[x][1:]]), w1, w2, b1, b2)[0] for x in range(len(test))]

	f = open("solution.txt", "w")
	for i in final:
		f.write(str(i) + "\n")
	f.close()


if __name__ == "__main__":
	main()