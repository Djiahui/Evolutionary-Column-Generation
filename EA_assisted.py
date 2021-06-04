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

		self.customer_list = set(range(1, customer_number + 2))


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
		root.dual= self.pi
		root.dis = self.dis
		root.pop = pop

		for _ in range(self.iteration):
			root.select()




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
		self.pop = None

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

	def expand(self):
		temp_reachable = list(self.pop.customer_list-self.tabu-self.selected)
		#Todo the evolutionary process of particle is need to be added in this function
		p = self.softmax(self.pop.x[self.current, temp_reachable])
		reachable = int(np.random.choice(temp_reachable, size=1, replace=False, p=p)[-1])
		self.selected.add(reachable)

		#generate a new child
		new_child = Node(reachable,self.customers)
		new_child.father = self
		new_child.dual = self.dual
		new_child.dis = self.dis
		new_child.pop = self.pop

		new_child.tabu.update(self.tabu)
		new_child.path = self.path+[reachable]
		new_child.current_dis = self.dis[self.current,reachable]+self.current_dis-self.dual[self.current]
		new_child.current_time = self.current_time+self.dis[self.current,reachable]+self.customers[new_child.current]['service']

		self.children.append(new_child)


		if new_child.current == len(self.customers)-1:
			new_child.quality = new_child.current_dis
			new_child.best_quality_route = new_child.path
			new_child.state = 'terminal'
			new_child.backup()
		else:
			new_child.rollout()


	def select(self):
		if len(self.children) < self.max_children and len(self.pop.customer_list - self.tabu - self.selected) > 0:
			self.expand()
		else:
			selected_index = np.argmax(list(map(lambda x: x.real_score(), self.children)))
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
				cur.father.best_quality_route = cur.best_quality_route

			cur = cur.father

	def rollout(self):
		## generate a route
		rollout_set = set()
		rollout_path = self.path[:]
		current_customer = self.current

		rollout_dis = self.current_dis
		rollout_time = self.current_time
		while current_customer!=len(self.customers)-1:
			candidates = list(self.pop.customer_list-self.customers[current_customer]['tabu']-rollout_set)
			next_customer = list(candidates)[np.argmax(self.pop.x[current_customer,candidates])]
			rollout_path.append(next_customer)
			rollout_set.add(next_customer)

			rollout_dis += self.dis[current_customer,next_customer]-self.dual[current_customer]
			rollout_time += self.dis[current_customer,next_customer]+self.customers[current_customer]['service']

			current_customer = next_customer

		self.quality = rollout_dis
		self.visited_times += 1
		self.best_quality_route = rollout_path[:]
		self.backup()

	def iteration(self):
		pass

	def real_score(self):

		return (self.quality - self.father.min_quality) / (self.father.max_quality - self.father.min_quality) + self.pop.x[
			self.father.current, self.current] + self.c * math.sqrt((math.log(self.father.visited_times)/self.visited_times))


	def softmax(self,x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

def t(dual, dis, customers, capacity, customer_number):
	eva = Evaluator(dis,customers,capacity,customer_number)
	eva.pi = [0]+dual+[0]
	pop = Individual(customer_number)
	eva.evaluate(pop)
	exit()

if __name__ == '__main__':
	pass
