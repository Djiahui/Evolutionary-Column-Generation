import gurobipy as gp
from gurobipy import GRB

import math
import copy
import numpy as np

# 0 is not appear in path
class Individual(object):
	def __init__(self):
		pass


class Population(object):
	def __init__(self, num):
		pass

class MCTS(object):
	def __init__(self, dis, customers, capacity, customer_number):
		self.pi = None
		self.dis = dis
		self.customers = customers
		self.customer_number = customer_number
		self.capacity = capacity

		self.customer_list = set(range(1, customer_number + 2))

		self.updated = [False] * (customer_number + 2)

		self.rel_matrix = np.random.rand(customer_number+1,customer_number+1)

		self.iteration = 10
		self.dual = None

	def find_path(self, pop):
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
			# Todo a feasibility test can be added to test the performance of pre_process
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


class Solver(object):
	def __init__(self, path, num):
		self.customers = {}
		self.num = num
		self.path = path
		self.dis = {}
		self.routes = {}
		self.rmp = None

		self.capacity = int(path.split('.')[0].split('_')[-1])

		self.population = Population(num)

		self.problem_csv()
		self.pre_press()
		self.set_cover()
		self.initial_routes_generates()

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

					if length == self.num:
						length = len(self.customers)
						self.customers[length] = copy.deepcopy(self.customers[0])
						break

	def dis_calcul(self):
		for i in range(self.num + 2):
			for j in range(self.num + 2):
				if i == j:
					self.dis[(i, j)] = 0
					continue
				if i == 0 and j == self.num + 1:
					self.dis[(i, j)] = 0
				if i == self.num + 1 and j == 0:
					self.dis[(i, j)] = 0
				temp = [self.customers[i]['loc'][0] - self.customers[j]['loc'][0],
						self.customers[i]['loc'][1] - self.customers[j]['loc'][1]]
				self.dis[(i, j)] = round(math.sqrt(temp[0] * temp[0] + temp[1] * temp[1]), 2)

	def pre_press(self):
		self.dis_calcul()
		for start, customer in self.customers.items():
			customer['tabu'] = set()
			if start == self.num + 1:
				a = 0
				return
			for target in range(1, self.num + 2):
				if customer['start'] + customer['service'] + self.dis[start, target] > self.customers[target]['end']:
					# print(customer['start'],customer['service'],dis[start,target],customer['start']+customer['service']+dis[start,target],customers[target]['end'])
					customer['tabu'].add(target)

	def set_cover(self):
		self.rmp = gp.Model('rmp')
		self.rmp.Params.logtoconsole = 0

		for i in range(self.num):
			index = i + 1
			fea = self.path_eva_vrptw([index, self.num + 1])
			if not fea:
				print('unfeasible', [index, self.num])
			self.routes[index]['var'] = self.rmp.addVar(ub=1, lb=0, obj=self.routes[index]['distance'], name='x')

		cons = self.rmp.addConstrs(self.routes[index]['var'] == 1 for index in range(1, self.num + 1))

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

	def initial_routes_generates(self):
		customer_list = [i for i in range(1, self.num + 1)]
		to_visit = customer_list[:]
		routes = []
		route = [0]
		temp_load = 0
		temp_time = 0

		while customer_list:
			for customer in customer_list:
				if self.customers[customer]['demand'] + temp_load < self.capacity:
					temp = temp_time + self.dis[route[-1], customer]
					if temp <= self.customers[customer]['end']:
						temp_time = max(temp, self.customers[customer]['start']) + self.customers[customer]['service']
						temp_load = temp_load + self.customers[customer]['demand']
						route.append(customer)
						to_visit.remove(customer)
					else:
						if customer == customer_list[-1]:
							route.append(self.num + 1)
							routes.append(route[:])
							route = [0]
							temp_load = 0
							temp_time = 0
				else:
					if customer == customer_list[-1]:
						route.append(self.num + 1)
						routes.append(route[:])
						route = [0]
						temp_load = 0
						temp_time = 0

			customer_list = to_visit[:]

		if len(route) > 1:
			route.append(self.num + 1)
			routes.append(route)

		self.add_column(routes)


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
			self.routes[temp_length]['var'] = self.rmp.addVar(column=added_column, obj=self.routes[temp_length]['distance'])

	def solve(self):
		population = Population()
		mcts = MCTS()
		dual = self.step()


		best_reduced_cost = 1e6
		while best_reduced_cost> -(1e-1):
			pass




