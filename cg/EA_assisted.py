import copy

import numpy as np
import math


class Population(object):
	def __init__(self, size, customers,dis,capacity,customer_number,dual):
		self.size = size
		self.max_iter = 5
		self.customer_number = customer_number
		self.pops = []
		self.eva = Evaluator(dis,customers,capacity,customer_number)
		self.customers = customers
		self.global_best = None

		self.dual = [0]+dual+[0]
		self.eva.dual = [0]+dual+[0]

	def gbest_update(self,pop):
		pop.gbest = self.global_best
		return pop

	def pop_generate(self):
		# random_generate

		temp_best_cost = 1e6

		for i in range(self.size):
			temp = Individual(self.customer_number)
			self.eva.evaluate(temp)
			temp.p_best = copy.deepcopy(temp)
			if temp.cost < temp_best_cost:
				self.global_best = copy.deepcopy(temp)
				temp_best_cost = temp.cost

			self.pops.append(temp)

		self.pops = list(map(lambda x:self.gbest_update(x),self.pops))


	def evolution(self,dual):

		self.update(dual)
		for _ in range(self.max_iter):
			self.iteration()

	def update(self,dual):
		self.dual = [0]+dual+[0]
		self.eva.dual = [0]+dual+[0]

	def iteration(self):
		for pop in self.pops:
			self.eva.evaluate(pop)

			pass






class Individual(object):
	def __init__(self, customer_number):
		self.customer_number = customer_number
		self.x = np.random.rand(customer_number + 2, customer_number + 2)
		self.velocity = np.random.rand(customer_number + 2, customer_number + 2)
		self.cost = None
		self.path = None

		self.pbest = None

		self.gbest = None

		self.customer_list = set(range(1, customer_number + 2))

		self.query = [True] * (customer_number + 1)

	def update(self, index,reachable_index):
		self.velocity[index][reachable_index] = self.velocity[index][reachable_index] + np.random.rand() * self.gbest.velocity[index][reachable_index]
		self.query[index] = True


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
		self.dual = None

	def evaluate(self, pop):
		# MCTS_decoder
		root = Node(0, self.customers)

		root.path.append(0)
		root.dual = self.dual
		root.dis = self.dis
		root.pop = pop

		for _ in range(self.iteration):
			root.select()
		pop.cost = root.quality
		pop.path = root.best_quality_route[:]

		pop.query = [False]*(pop.customer_number+1)


class Node(object):
	def __init__(self, index, customers):
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

		self.quality = 1e6
		self.max_quality = -1e6
		self.min_quality = 1e6
		self.visited_times = 0

		self.path = []
		self.best_quality_route = None

	def expand(self):
		temp_reachable = list(self.pop.customer_list - self.tabu - self.selected)

		if not self.pop.query[self.current]:
			self.pop.update(self.current,list(self.pop.customer_list-self.tabu))
		# Todo the evolutionary process of particle is need to be added in this function
		p = self.softmax(self.pop.x[self.current, temp_reachable])
		reachable = int(np.random.choice(temp_reachable, size=1, replace=False, p=p)[-1])
		self.selected.add(reachable)

		# generate a new child
		new_child = Node(reachable, self.customers)
		new_child.father = self
		new_child.dual = self.dual
		new_child.dis = self.dis
		new_child.pop = self.pop

		new_child.tabu.update(self.tabu)
		new_child.selected.update()
		new_child.path = self.path + [reachable]
		new_child.current_dis = self.dis[self.current, reachable] + self.current_dis - self.dual[self.current]
		new_child.current_time = self.current_time + self.dis[self.current, reachable] + \
								 self.customers[new_child.current]['service']

		self.children.append(new_child)

		if new_child.current == len(self.customers) - 1:
			new_child.quality = new_child.current_dis
			new_child.best_quality_route = new_child.path
			new_child.state = 'terminal'
			new_child.visited_times += 1
			new_child.backup()
		else:
			new_child.rollout()

	def select(self):
		if len(self.children) < self.max_children and len(self.pop.customer_list - self.tabu - self.selected) > 0:
			self.expand()
		else:
			selected_index = np.argmax(list(map(lambda x: x.get_score(), self.children)))
			if self.children[selected_index].state == 'terminal':
				self.children[selected_index].backup()
			else:
				self.children[selected_index].select()


	def backup(self):
		cur = self
		while cur.father:
			#Todo there is a problem, the min/max quality
			cur.father.min_quality = min(cur.father.min_quality, cur.quality)
			cur.father.max_quality = max(cur.father.max_quality, cur.quality)

			cur.father.visited_times += 1

			if cur.quality < cur.father.quality:
				cur.father.quality = cur.quality
				cur.father.best_quality_route = cur.best_quality_route

			cur = cur.father

	def rollout(self):
		## generate a route

		rollout_path = self.path[:]
		rollout_set = set(rollout_path)
		current_customer = self.current

		rollout_dis = self.current_dis
		rollout_time = self.current_time
		while current_customer != len(self.customers) - 1:
			candidates = list(self.pop.customer_list - self.customers[current_customer]['tabu'] - rollout_set)
			next_customer = list(candidates)[np.argmax(self.pop.x[current_customer, candidates])]
			rollout_path.append(next_customer)
			rollout_set.add(next_customer)

			rollout_dis += self.dis[current_customer, next_customer] - self.dual[current_customer]
			rollout_time += self.dis[current_customer, next_customer] + self.customers[current_customer]['service']

			current_customer = next_customer

		self.quality = rollout_dis
		self.visited_times += 1
		self.best_quality_route = rollout_path[:]
		self.backup()

	def iteration(self):
		pass

	def get_score(self):
		if not self.visited_times:
			a = 0

		return (self.quality - self.father.min_quality) / (self.father.max_quality - self.father.min_quality) + \
			   self.pop.x[self.father.current, self.current] + self.c * math.sqrt((math.log(self.father.visited_times) / self.visited_times))

	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)


def t(dual, dis, customers, capacity, customer_number):
	# Todo maintain a global archive in slover, the sarchive can be regarded as a input augrment to EA approach.More specifilty, the population should be maintained for all optimization process
	eva = Evaluator(dis, customers, capacity, customer_number)
	eva.pi = [0] + dual + [0]
	pop = Individual(customer_number)
	eva.evaluate(pop)
	exit()


if __name__ == '__main__':
	import pickle
	dual = [30.46, 36.0, 44.72, 50.0, 41.24, 22.36, 42.42, 52.5, 64.04, 51.0, 67.08, 30.0, 22.36, 64.04, 60.82, 58.3, 60.82, 31.62, 64.04, 63.24, 36.06, 53.86, 72.12, 60.0, -9.77000000000001, 22.36, 10.0, 12.64, 59.66, 51.0, 34.92, 68.0, 49.52, 72.12, 74.97, 82.8, 42.42, 84.86, 67.94, 22.36, 57.72, 51.0, 68.36, 63.78, 58.3, 39.94, 68.42, -11.430000000000007, 68.82, -7.75, 53.86, 22.62, 8.94, 28.900000000000006, -8.009999999999991, 40.97, 46.38, 17.369999999999997, 35.6, -23.93, 51.0, 51.0, 69.86, 93.04, 99.86, 19.909999999999997, 87.72, -10.100000000000009, 24.34, -146.32000000000002, 0.030000000000001137, 44.94, 40.24, 23.940000000000012, 55.56, 31.3, -37.219999999999985, 54.92, 5.079999999999995, -0.6599999999999966, 33.35000000000001, 46.64, 42.2, 48.66, 24.72, 35.730000000000004, 12.739999999999995, 38.48, -28.500000000000007, -62.43000000000001, 51.22, 36.76, -73.82, -26.92, 29.74, -68.12, -66.98999999999998, 42.52, -57.730000000000004, -135.7]
	capacity = 200
	customer_number = 100
	with open('../dis.pkl', 'rb') as pkl:
		dis = pickle.load(pkl)
	with open('../customers.pkl', 'rb') as pkl2:
		customers = pickle.load(pkl2)


	population = Population(5,customers,dis,capacity,customer_number,dual)
	population.pop_generate()

