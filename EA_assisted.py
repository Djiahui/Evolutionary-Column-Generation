import copy

import numpy as np


class Population(object):
	def __init__(self, size, customers, eva):
		self.size = size
		self.customer_number = len(customers)
		self.population = []
		self.eva = eva
		self.customers = customers
		self.global_best = None

	def pop_generate(self):
		# random_generate

		temp_best_cost = 1e6

		for i in range(self.size):
			temp = Individual(self.customer_number)
			temp.cost = self.eva.evaluate(temp.x)
			temp.pbest_cost = temp.cost

			if temp.cost<temp_best_cost:
				self.global_best = copy.deepcopy(temp)
				temp_best_cost = temp.cost

	def update(self):
		pass


class Individual(object):
	def __init__(self, customer_number):
		self.customer_number = customer_number
		self.x = np.zeros(customer_number+2,customer_number+2)
		self.cost = None
		self.pbest = None
		self.gbest = None



class Evaluator(object):
	def __init__(self, dis, customers, capacity, customer_number):
		self.pi = None
		self.dis = dis
		self.customers = customers
		self.customer_number = customer_number
		self.capacity = capacity

		self.customer_list = set(range(1,customer_number+2))

		self.updated = [False]*(customer_number+2)

	def evaluate(self, x):
		# MCTS_decoder
		root = Node(0,self.customers[0])

		while True:
			root.pre_sample(self.customer_list,x,self.customers)








class Node(object):
	def __init__(self,index,customer):
		self.current = index
		self.current_customer = customer
		self.tabu = self.current_customer['tabu']
		self.children = []
		self.father = None

		self.quality = 0
		self.visited_times = 0
		self.max_children = 5

		self.max_reward = 0
		self.min_reward = 0

		self.score = 1e6


	def pre_sample(self,customer_list,matrix,customers):
		temp_reachable = list(customer_list-self.tabu)
		if len(temp_reachable)>self.max_children:
			p = softmax(matrix[self.current,temp_reachable])
			reachable = np.random.choice(temp_reachable,size = self.max_children,replace=False,p=p)
		else:
			reachable = temp_reachable



		for target in reachable:
			temp_node = Node(target,customers[target])
			temp_node.father = self
			temp_node.tabu.update(self.tabu)
			temp_node.rollout()
			self.children.append(temp_node)




	def expand(self):
		pass

	def select(self):
		pass

	def backup(self):
		pass

	def rollout(self):
		pass

	def iteration(self):
		pass







def softmax(x):
	return np.exp(x)/np.sum(np.exp(x),axis=0)


if __name__ == '__main__':
	pass
