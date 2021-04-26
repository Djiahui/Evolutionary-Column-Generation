import numpy as np


class Population(object):
	def __init__(self, size, customer_number, eva):
		self.size = size
		self.customer_number = customer_number
		self.population = []
		self.eva = eva

	def pop_generate(self):
		# random_generate
		for i in range(self.size):
			temp = Individual(self.customer_number)
			temp.cost = self.eva.evaluate(temp.x)

	def update(self):
		pass


class Individual(object):
	def __init__(self, variable_size):
		self.variable_size = variable_size
		self.x = np.random.rand(variable_size)
		self.cost = None


class Evaluator(object):
	def __init__(self, dis, customers, capacity, customer_number):
		self.pi = None
		self.dis = dis
		self.customers = customers
		self.customer_number = customer_number
		self.capacity = capacity

	def evaluate(self, x):
		# greedy_decoder
		index = x.argsort()
		index += 1

		terminal = False

		while not terminal:
			current = index.pop()



if __name__ == '__main__':
	pass
