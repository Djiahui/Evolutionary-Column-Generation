import copy
import math

class velocity(object):
	def __init__(self):
		self.edges = []

class pop(object):
	def __init__(self):
		self.position = []
		self.route = None

class edge(object):
	def __init__(self):
		self.arc = None
		self.probability = 0

def problem_read(path):
	customers = {}
	_, capcity, customer_num = path.split('.')[0].split('_')
	flag = False

	with open('../data/' + path) as p:
		for line in p:
			if not flag:
				flag = True
				pass

			else:
				temp = line.split(',')
				length = len(customers)
				customers[length] = {}
				customers[length]['loc'] = [float(temp[1]), float(temp[2])]
				customers[length]['demand'] = int(float(temp[3]))
				customers[length]['start'] = int(float(temp[4]))
				customers[length]['end'] = int(float(temp[5]))
				customers[length]['service'] = int(float(temp[6]))

				if length == int(customer_num):
					length = len(customers)
					customers[length] = copy.deepcopy(customers[0])
					break

	return customers, int(capcity), int(customer_num)


def distance_calculate(customers, customer_num):
	dis = {}
	for i in range(customer_num + 2):
		for j in range(customer_num + 2):
			if i == j:
				dis[(i, j)] = 0
				continue
			if i == 0 and j == customer_num + 1:
				dis[(i, j)] = 0
			if i == customer_num + 1 and j == 0:
				dis[(i, j)] = 0
			temp = [customers[i]['loc'][0] - customers[j]['loc'][0],
					customers[i]['loc'][1] - customers[j]['loc'][1]]
			dis[(i, j)] = round(math.sqrt(temp[0] * temp[0] + temp[1] * temp[1]), 2)
	return dis


def main(path):
	customers, capacity, customer_num = problem_read(path)
	dis = distance_calculate(customers, customer_num)

	pops = []




if __name__ == '__main__':
	main('C101_200_100.csv')
	exit()
