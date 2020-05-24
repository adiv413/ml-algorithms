from decision_tree import DecisionTree
import csv
from random import randint, sample

def main():

	training = []
	testing = []
	trees = []
	final = []
	n = 20000 # number of decision trees to be trained

	with open("train.csv", newline = '') as trainingData:
			next(trainingData)

			csvreader = csv.reader(trainingData)
			training = [list(map(float, x)) for x in csvreader] 

	with open("test.csv", newline = '') as testingData:
		next(testingData)

		csvreader2 = csv.reader(testingData)
		testing = [list(map(float, x)) for x in csvreader2]

	for i in range(n): #training n decision trees
		data = []

		'''repeat x times where x is sample size:
		select and store the 3 features
		select a row
		in that row, append those 3 features and the outcome'''

		features = sample(range(7), 6)
		features.append(7)

		for i in range(10): # sample size

			row = randint(0, len(training) - 1)
			point = []

			for i in range(8):
				if i not in features:
					point.append(-1)
				else:
					point.append(training[row][i])
			data.append(point)

		dt = DecisionTree(data) 
		dt.train()
		trees.append(dt)

	#voting

	count = 0
	x = 0
	for i in testing:
		for j in range(len(trees)):
			x += 1
			if trees[j].fit([i])[0] == 1:
				count += 1
		x = 0
		if count / n > 0.5:
			final.append(1)
			count = 0
		else:
			final.append(0)
			count = 0

	f = open("solution.txt", "w")
	for i in final:
		f.write(str(i) + "\n")
	f.close()


if __name__ == "__main__":
	main()