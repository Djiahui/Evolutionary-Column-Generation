import gurobipy as gp
from gurobipy import GRB

import math

class Solver(object):
	def __init__(self,path,num):
		self.customers = {}
		self.num = num
		self.path = path
		self.dis = {}
		self.routes = {}
		self.rmp = None

		self.capacity = int(path.split('.')[0].split('_')[-1])




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

	def path_eva_vrptw(self,path):
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

	def start(self):
		self.problem_csv()
		self.pre_press()
		self.set_cover()
		self.initial_routes_generates()

	def step(self):
		self.rmp.optimize()
		dual = self.rmp.getAttr(GRB.Attr.Pi, self.rmp.getConstrs())
		return dual

	def add_column(self,routes):
		for route in routes:
			fea = self.path_eva_vrptw(route)
			if not fea:
				print('unfeasibile', route[1:])
				continue
			temp_length = len(self.routes)
			added_column = gp.Column(self.routes[temp_length]['column'], self.rmp.getConstrs())
			self.routes[temp_length]['var'] = self.rmp.addVar(column=added_column, obj=self.routes[temp_length]['distance'])

class Individual(object):
	def __init__(self):
		pass
