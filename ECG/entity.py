import gurobipy as gp
from gurobipy import GRB

import math
import copy
import numpy as np
import pickle


# 0 is not appear in path
class Individual(object):
	def __init__(self, path, dis):
		self.path = path
		self.dis = dis

		self.cost = 0

	def evaluate_under_dual(self, dual):
		for customer in self.path[:-1]:
			self.cost += dual[customer]

		self.cost += self.dis


class Population(object):
	def __init__(self, customer_num, customers, dis, capacity):
		self.pops = []

		self.customer_num = customer_num
		self.customers = customers
		self.dis = dis
		self.capacity = capacity

		self.initial_routes_generates()

	def initial_routes_generates(self):
		customer_list = [i for i in range(1, self.customer_num + 1)]
		to_visit = customer_list[:]
		routes = []
		distances = []
		route = [0]
		temp_load = 0
		temp_time = 0
		temp_dis = 0

		# 从头遍历判断一个顾客顾客是否满足情况，如果满足的话就扣减，如果不符合情况就跳过（先判断是不是最后一个如果是最后一个认为一条路径完结）
		while customer_list:
			for customer in customer_list:
				if self.customers[customer]['demand'] + temp_load < self.capacity:
					temp = temp_time + self.dis[route[-1], customer]
					if temp <= self.customers[customer]['end']:
						temp_time = max(temp, self.customers[customer]['start']) + self.customers[customer]['service']
						temp_dis += self.dis[route[-1], customer]
						temp_load = temp_load + self.customers[customer]['demand']
						route.append(customer)
						to_visit.remove(customer)
					else:
						if customer == customer_list[-1]:
							route.append(self.customer_num + 1)
							temp_dis += self.dis[route[-1], self.customer_num + 1]
							routes.append(route[:])
							distances.append(temp_dis)
							route = [0]
							temp_dis = 0
							temp_load = 0
							temp_time = 0
				else:
					if customer == customer_list[-1]:
						route.append(self.customer_num + 1)
						temp_dis += self.dis[route[-1], self.customer_num + 1]
						routes.append(route[:])
						distances.append(temp_dis)
						route = [0]
						temp_load = 0
						temp_time = 0

			customer_list = to_visit[:]

		if len(route) > 1:
			route.append(self.customer_num + 1)
			temp_dis += self.dis[route[-1], self.customer_num + 1]
			distances.append(temp_dis)
			routes.append(route)

		for dis, path in zip(distances, routes):
			self.pops.append(Individual(path, dis))

	def evaluate(self, dual):
		for pop in self.pops:
			pop.evaluate_under_dual(dual)


class MCTS(object):
	def __init__(self, dis, customers, capacity, customer_number):
		self.pi = None
		self.dis = dis
		self.customers = customers
		self.customer_number = customer_number
		self.capacity = capacity

		self.iteration = 50

		self.matrix_init()

	def matrix_init(self):
		self.rel_matrix = np.random.rand(customer_number + 2, customer_number + 2)
		for customer, information in self.customers.items():
			self.rel_matrix[customer, list(information['tabu'])] = 0

	def find_path(self, dual):
		dual = [0] + dual + [0]
		# MCTS_decoder
		root = Node(0, self.customers, self.rel_matrix, dual, self.dis, self.capacity)

		root.path.append(0)
		root.dual = dual
		root.dis = self.dis
		root.demand = 0
		root.capacity = self.capacity

		for _ in range(self.iteration):
			root.select()

		exit()


class Node(object):
	def __init__(self, index, customers, matrix, dual, dis, capacity):
		self.current = index
		self.current_dis = 0
		self.current_time = 0
		self.current_cost = 0

		self.customers = customers
		self.customer_list = set([i for i in range(1, len(customers))])
		self.dis = dis
		self.dual = dual
		self.demand = None
		self.capacity = capacity
		self.tabu = copy.deepcopy(self.customers[index]['tabu'])
		self.selected = set()

		self.children = []
		self.father = None
		self.max_children = 20

		self.c = 1

		self.state = None

		self.quality = 1e6
		self.max_quality = -1e6
		self.min_quality = 1e6
		self.visited_times = 0

		self.path = []
		self.best_quality_route = None

		self.rel_matrix = matrix

	def evaluate(self, path):
		cur = 0
		dis_eva = 0
		cost_eva = 0
		time_eva = 0
		for cus in path:
			if time_eva + self.dis[cus, cur] > self.customers[cus]['end']:
				print('wrong' + str(cur))
				return
			else:
				time_eva = max(time_eva + self.dis[cus, cur], self.customers[cus]['start']) + self.customers[cus][
					'service']
				dis_eva += self.dis[cur, cus]
				cost_eva += (self.dis[cur, cus] - self.dual[cur])
			cur = cus
		demand = sum([customers[x]['demand'] for x in path[:-1]])
		if demand > self.capacity:
			print('wrong capacity')

		print(dis_eva, cost_eva)

	def expand(self, reachable_customers):

		p = self.softmax(self.rel_matrix[self.current, list(reachable_customers)])
		reachable = int(np.random.choice(list(reachable_customers), size=1, replace=False, p=p)[-1])
		self.selected.add(reachable)

		# generate a new child
		new_child = Node(reachable, self.customers, self.rel_matrix, self.dual, self.dis, self.capacity)
		new_child.father = self
		new_child.dual = self.dual
		new_child.dis = self.dis
		new_child.rel_matrix = self.rel_matrix

		new_child.tabu.update(self.tabu)
		new_child.path = self.path + [reachable]
		new_child.current_cost = self.dis[self.current, reachable] + self.current_cost - self.dual[self.current]
		new_child.current_dis = self.dis[self.current, reachable] + self.current_dis
		new_child.current_time = max(self.current_time + self.dis[self.current, reachable],
									 self.customers[reachable]['start']) + self.customers[reachable]['service']
		new_child.demand = self.demand + self.customers[reachable]['demand']
		new_child.capacity = self.capacity
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
		# customers_list - already visited customers - cannot visit (time window) - this node selected
		reachable_customers = self.customer_list - self.tabu - self.selected
		# - cannot visit capacity
		reachable_customers = list(filter(
			lambda x: self.customers[x]['demand'] + self.demand <= self.capacity and self.current_time + self.dis[
				x, self.current] <= self.customers[x]['end'], reachable_customers))
		# the return depot can be selected at any time
		if len(self.children) < self.max_children and len(reachable_customers) > 0:
			self.expand(reachable_customers)
		else:
			selected_index = np.argmax(list(map(lambda x: x.get_score(), self.children)))
			if self.children[selected_index].state == 'terminal':
				self.children[selected_index].backup()
			else:
				self.children[selected_index].select()

	def backup(self):
		cur = self
		while cur.father:
			# Todo there is a problem, the min/max quality
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
		rollout_cost = self.current_cost
		demand = self.demand
		rollout_tabu = copy.deepcopy(self.tabu)
		while current_customer != len(self.customers) - 1:
			# Todo current rollout policy is just choosing the maximum value of relationship matrix
			candidates = list(self.customer_list - rollout_tabu - rollout_set)
			candidates = list(filter(
				lambda x: customers[x]['demand'] + demand <= self.capacity and rollout_time + self.dis[
					current_customer, x] < self.customers[x]['end'], candidates))
			next_customer = list(candidates)[np.argmax(self.rel_matrix[current_customer, candidates])]
			rollout_path.append(next_customer)
			rollout_set.add(next_customer)

			rollout_cost += (self.dis[current_customer, next_customer] - self.dual[current_customer])
			rollout_dis += self.dis[current_customer, next_customer]
			rollout_time = max(rollout_time + self.dis[current_customer, next_customer],
							   self.customers[next_customer]['start']) + self.customers[next_customer]['service']
			rollout_tabu.update(self.customers[next_customer]['tabu'])
			demand += self.customers[next_customer]['demand']

			current_customer = next_customer

		self.quality = rollout_cost
		self.visited_times += 1
		self.best_quality_route = rollout_path[:]
		self.backup()

	def iteration(self):
		pass

	def get_score(self):
		if not self.visited_times:
			a = 0
		if self.father.min_quality == self.father.max_quality:
			# only one children node for father thus only one choice
			return 1

		return -(self.quality - self.father.min_quality) / (self.father.max_quality - self.father.min_quality) + \
			   self.rel_matrix[self.father.current, self.current] + self.c * math.sqrt(
			(math.log(self.father.visited_times) / self.visited_times))

	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)


class Solver(object):
	def __init__(self, path, num, capacity=200):
		self.customers = {}
		self.customer_num = num
		self.path = path
		self.dis = {}
		self.routes = {}
		self.rmp = None
		self.customer_list = set(range(1, num + 2))

		# self.capacity = int(path.split('.')[0].split('_')[-1])
		self.capacity = capacity
		self.problem_csv()
		self.pre_press()
		self.set_cover()

		self.population = Population(self.customer_num, self.customers, self.dis, self.capacity)
		self.mcts = MCTS(self.dis, self.customers, self.capacity, self.customer_num)

	def problem_csv(self):
		flag = False

		with open(self.path) as p:
			for line in p:
				if not flag:
					flag = True
					pass

				else:
					temp = line.split(',')
					length = len(self.customers)
					self.customers[length] = {}
					self.customers[length]['loc'] = [float(temp[1]), float(temp[2])]
					self.customers[length]['demand'] = int(float(temp[3]))
					self.customers[length]['start'] = int(float(temp[4]))
					self.customers[length]['end'] = int(float(temp[5]))
					self.customers[length]['service'] = int(float(temp[6]))

					if length == self.customer_num:
						length = len(self.customers)
						self.customers[length] = copy.deepcopy(self.customers[0])
						break

	def dis_calcul(self):
		for i in range(self.customer_num + 2):
			for j in range(self.customer_num + 2):
				if i == j:
					self.dis[(i, j)] = 0
					continue
				if i == 0 and j == self.customer_num + 1:
					self.dis[(i, j)] = 0
				if i == self.customer_num + 1 and j == 0:
					self.dis[(i, j)] = 0
				temp = [self.customers[i]['loc'][0] - self.customers[j]['loc'][0],
						self.customers[i]['loc'][1] - self.customers[j]['loc'][1]]
				self.dis[(i, j)] = round(math.sqrt(temp[0] * temp[0] + temp[1] * temp[1]), 2)

	def pre_press(self):
		self.dis_calcul()
		for start, customer in self.customers.items():
			customer['tabu'] = set()
			customer['tabu'].add(start)
			if start == self.customer_num + 1:
				return
			for target in range(1, self.customer_num + 2):
				if customer['start'] + customer['service'] + self.dis[start, target] > self.customers[target]['end']:
					# print(customer['start'],customer['service'],dis[start,target],customer['start']+customer['service']+dis[start,target],customers[target]['end'])
					customer['tabu'].add(target)

	def set_cover(self):
		self.rmp = gp.Model('rmp')
		self.rmp.Params.logtoconsole = 0

		for i in range(self.customer_num):
			index = i + 1
			fea = self.path_eva_vrptw([index, self.customer_num + 1])
			if not fea:
				print('unfeasible', [index, self.customer_num])
			self.routes[index]['var'] = self.rmp.addVar(ub=1, lb=0, obj=self.routes[index]['distance'], name='x')

		cons = self.rmp.addConstrs(self.routes[index]['var'] == 1 for index in range(1, self.customer_num + 1))

		self.rmp.update()

	def path_eva_vrptw(self, path):
		cost = 0
		pre = 0
		load = 0
		time = 0
		fea = True

		for cus in path:
			cost = cost + self.dis[pre, cus]
			time = time + self.dis[pre, cus]
			load = load + self.customers[cus]['demand']

			if cus == path[-1]:
				continue

			if time > self.customers[cus]['end']:
				fea = False
				return fea
			if load > self.capacity:
				fea = False
				return fea
			time = max(time, self.customers[cus]['start']) + self.customers[cus]['service']
			pre = cus

		if fea:
			column = [1 if i in path else 0 for i in range(1, len(self.customers) - 1)]
			n = len(self.routes) + 1
			self.routes[n] = {}
			self.routes[n]['demand'] = load
			self.routes[n]['distance'] = cost
			self.routes[n]['column'] = column
			self.routes[n]['route'] = path

		return fea

	def linear_relaxition(self):
		self.rmp.optimize()
		dual = self.rmp.getAttr(GRB.Attr.Pi, self.rmp.getConstrs())
		return dual

	def add_column(self, routes):
		for route in routes:
			fea = self.path_eva_vrptw(route)
			if not fea:
				print('unfeasibile', route[1:])
				continue
			temp_length = len(self.routes)
			added_column = gp.Column(self.routes[temp_length]['column'], self.rmp.getConstrs())
			self.routes[temp_length]['var'] = self.rmp.addVar(column=added_column,
															  obj=self.routes[temp_length]['distance'])

	def paths_generate(self, dual):
		self.population.evaluate(dual)

	# two ways to generate the path
	# 1.evolutionary operator 2.MCTS
	# MCTS consider two factors: the customer number in current population, the dual, the negative information from population

	def solve(self):

		dual = self.rmp.step()

		best_reduced_cost = 1e6
		while best_reduced_cost > -(1e-1):
			paths = self.paths_generate(dual)


if __name__ == '__main__':
	solver = Solver('../data/C101_200.csv', 100, 200)
	exit()
	# test for mcts
	dual = [30.46, 36.0, 44.72, 50.0, 41.24, 22.36, 42.42, 52.5, 64.04, 51.0, 67.08, 30.0, 22.36, 64.04, 60.82, 58.3,
			60.82, 31.62, 64.04, 63.24, 36.06, 53.86, 72.12, 60.0, -9.77000000000001, 22.36, 10.0, 12.64, 59.66, 51.0,
			34.92, 68.0, 49.52, 72.12, 74.97, 82.8, 42.42, 84.86, 67.94, 22.36, 57.72, 51.0, 68.36, 63.78, 58.3, 39.94,
			68.42, -11.430000000000007, 68.82, -7.75, 53.86, 22.62, 8.94, 28.900000000000006, -8.009999999999991, 40.97,
			46.38, 17.369999999999997, 35.6, -23.93, 51.0, 51.0, 69.86, 93.04, 99.86, 19.909999999999997, 87.72,
			-10.100000000000009, 24.34, -146.32000000000002, 0.030000000000001137, 44.94, 40.24, 23.940000000000012,
			55.56, 31.3, -37.219999999999985, 54.92, 5.079999999999995, -0.6599999999999966, 33.35000000000001, 46.64,
			42.2, 48.66, 24.72, 35.730000000000004, 12.739999999999995, 38.48, -28.500000000000007, -62.43000000000001,
			51.22, 36.76, -73.82, -26.92, 29.74, -68.12, -66.98999999999998, 42.52, -57.730000000000004, -135.7]
	capacity = 200
	customer_number = 100
	with open('../dis.pkl', 'rb') as pkl:
		dis = pickle.load(pkl)
	with open('../customers.pkl', 'rb') as pkl2:
		customers = pickle.load(pkl2)
	for customer, info in customers.items():
		info['tabu'].add(customer)

	mcts = MCTS(dis, customers, capacity, customer_number)
	mcts.find_path(dual)
