import copy

import numpy as np
import math


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

			if temp.cost < temp_best_cost:
				self.global_best = copy.deepcopy(temp)
				temp_best_cost = temp.cost

	def update(self):
		pass


class Individual(object):
	def __init__(self, customer_number):
		self.customer_number = customer_number
		self.x = np.zeros(customer_number + 2, customer_number + 2)
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

		self.customer_list = set(range(1, customer_number + 2))

		self.updated = [False] * (customer_number + 2)

		self.iteration = 10

	def evaluate(self, pop):
		# MCTS_decoder
		root = Node(0)
		root.customers = self.customers
		root.path.append(0)

		for _ in range(self.iteration):
			pass




class Node(object):
	def __init__(self, index):
		self.current = index
		self.customers = None
		self.tabu = self.customers[index]['tabu']
		self.selected = set()
		self.children = []
		self.father = None
		self.c = 1
		self.state = None

		self.quality = 0
		self.visited_times = 0
		self.max_children = 5

		self.max_quality = 0
		self.min_quality = 0

		self.score = 1e6

		self.path = []

	# def pre_sample(self, customer_list, matrix, customers):
	# 	temp_reachable = list(customer_list - self.tabu)
	# 	if len(temp_reachable) > self.max_children:
	# 		p = softmax(matrix[self.current, temp_reachable])
	# 		reachable = np.random.choice(temp_reachable, size=self.max_children, replace=False, p=p)
	# 	else:
	# 		reachable = temp_reachable
	#
	# 	for target in reachable:
	# 		temp_node = Node(target, customers[target])
	# 		temp_node.father = self
	# 		temp_node.tabu.update(self.tabu)
	# 		temp_node.rollout()
	# 		self.children.append(temp_node)

	def expand(self,customer_list,matrix):
		temp_reachable = list(customer_list-self.tabu-self.selected)
		#Todo the evolutionary process of particle is need to be added in this function
		p = softmax(matrix[self.current, temp_reachable])
		reachable = np.random.choice(temp_reachable, size=1, replace=False, p=p)
		self.selected.add(reachable)

		#generate a new child
		new_child = Node(reachable)
		new_child.customers = self.customers
		new_child.father = self
		new_child.tabu.update(self.tabu)


		if new_child.current == customer_list[-1]:
			new_child.state = 'terminal'
			new_child.backup()
		else:
			new_child.rollout()


		return new_child

	def select(self, customer_list, matrix):
		if len(self.children) < self.max_children and len(customer_list - self.tabu - self.selected) > 0:
			self.expand(customer_list)
		else:
			selected_index = np.argmax(map(lambda x: x.real_score(matrix), self.children))
			if self.children[selected_index].state=='terminal':
				self.children[selected_index].backup()
			else:
				self.children[selected_index].select()




	def backup(self):
		pass

	def rollout(self):
		self.backup()
		pass

	def iteration(self):
		pass

	def real_score(self, matrix):

		return (self.quality - self.father.min_quality) / (self.father.max_quality - self.father.min_quality) + matrix[
			self.father.current, self.current] + self.c * math.sqrt((math.log(self.father.visited_times)/self.visited_times))


def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
	pass
