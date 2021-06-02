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
		self.x = np.random.rand(customer_number + 2, customer_number + 2)
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
		root = Node(0,self.customers)

		root.path.append(0)
		root.pi = self.pi
		root.dis = self.dis

		for _ in range(self.iteration):
			root.select(self.customer_list,pop.x)




class Node(object):
	def __init__(self, index,customers):
		self.current = index
		self.current_dis = 0
		self.current_time = 0

		self.customers = customers
		self.dis = None
		self.dual = None
		self.tabu = self.customers[index]['tabu']
		self.selected = set()

		self.children = []
		self.father = None
		self.max_children = 5

		self.c = 1

		self.state = None

		self.quality = 0
		self.max_quality = -1e6
		self.min_quality = 1e6
		self.visited_times = 0

		self.path = []
		self.best_quality_route = None

	def expand(self,customer_list,matrix):
		temp_reachable = list(customer_list-self.tabu-self.selected)
		#Todo the evolutionary process of particle is need to be added in this function
		p = self.softmax(matrix[self.current, temp_reachable])
		reachable = int(np.random.choice(temp_reachable, size=1, replace=False, p=p)[-1])
		self.selected.add(reachable)

		#generate a new child
		new_child = Node(reachable,self.customers)
		new_child.father = self
		new_child.dual = self.dual
		new_child.dis = self.dis

		new_child.tabu.update(self.tabu)
		new_child.path = self.path+[reachable]
		new_child.current_dis = self.dis[self.current,reachable]+self.current_dis-self.dual[self.current]
		new_child.current_time = self.current_time+self.dis[self.current,reachable]


		if new_child.current == len(self.customers)-1:
			new_child.quality = new_child.dis
			new_child.best_quality_route = new_child.path
			new_child.state = 'terminal'
			new_child.backup()
		else:
			new_child.rollout()


		return new_child

	def select(self, customer_list, matrix):
		if len(self.children) < self.max_children and len(customer_list - self.tabu - self.selected) > 0:
			self.expand(customer_list,matrix)
		else:
			selected_index = np.argmax(map(lambda x: x.real_score(matrix), self.children))
			if self.children[selected_index].state=='terminal':
				self.children[selected_index].backup()
			else:
				self.children[selected_index].select()




	def backup(self):
		cur = self
		while cur.father:
			cur.father.min_quality = min(cur.father.min_quality, cur.quality)
			cur.father.max_quality = max(cur.father.max_quality, cur.quality)

			cur.father.visited_times += 1

			if cur.quality < cur.father.quality:
				cur.father.quality = cur.quality
				cur.father.quality.best_quality_route = cur.best_quality_route

	def rollout(self):
		## generate a route
		dis = np.random.uniform(-100,-50)
		self.quality = dis
		self.visited_times += 1
		self.best_quality_route = [1,2,3,4,5,21,99]
		self.backup()

	def iteration(self):
		pass

	def real_score(self, matrix):

		return (self.quality - self.father.min_quality) / (self.father.max_quality - self.father.min_quality) + matrix[
			self.father.current, self.current] + self.c * math.sqrt((math.log(self.father.visited_times)/self.visited_times))


	def softmax(self,x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

def t(dual, dis, customers, capacity, customer_number):
	eva = Evaluator(dis,customers,capacity,customer_number)
	print(id(customers))
	eva.pi = [0]+dual+[0]
	pop = Individual(customer_number)
	eva.evaluate(pop)
	exit()
if __name__ == '__main__':
	pass
