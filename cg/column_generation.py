import time

import gurobipy as gp
from gurobipy import GRB

import math
import matplotlib.pyplot as plt

import labeling_Algoithm_vrptw
import pickle

import copy


class Solver(object):
	def __init__(self,path,num):
		self.customers = {}
		self.num = num
		self.customer_num = num
		self.path = path
		self.dis = {}
		self.routes = {}
		self.rmp = None

		self.capacity = 200




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
			self.routes[index]['var'] = self.rmp.addVar(ub=1, lb=0, obj=self.routes[index]['distance'])

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
		customer_list = [i for i in range(1, self.customer_num + 1)]
		to_visit = customer_list[:]
		routes = []
		distances = []
		demands = []
		route = [0]
		arrive_times_vectors = []
		arrive_time_vector = [0]
		temp_load = 0
		departure_time = 0
		temp_dis = 0

		# 从头遍历判断一个顾客顾客是否满足情况，如果满足的话就扣减，如果不符合情况就跳过（先判断是不是最后一个如果是最后一个认为一条路径完结）
		while customer_list:
			for customer in customer_list:
				arrive_time = departure_time + self.dis[route[-1], customer]
				if self.customers[customer]['demand'] + temp_load < self.capacity and arrive_time <= \
						self.customers[customer]['end']:
					arrive_time_vector.append(arrive_time)
					departure_time = max(arrive_time, self.customers[customer]['start']) + self.customers[customer][
						'service']
					temp_dis += self.dis[route[-1], customer]
					temp_load = temp_load + self.customers[customer]['demand']
					route.append(customer)
					to_visit.remove(customer)
				elif customer == customer_list[-1]:
					arrive_time_vector.append(departure_time + self.dis[route[-1], self.customer_num + 1])
					temp_dis += self.dis[route[-1], self.customer_num + 1]
					route.append(self.customer_num + 1)
					routes.append(route[:])
					arrive_times_vectors.append(arrive_time_vector[:])
					distances.append(temp_dis)
					demands.append(temp_load)
					route = [0]
					arrive_time_vector = [0]
					temp_dis = 0
					temp_load = 0
					departure_time = 0

			customer_list = to_visit[:]

		if len(route) > 1:
			arrive_time_vector.append(departure_time + self.dis[route[-1], self.customer_num + 1])
			temp_dis += self.dis[route[-1], self.customer_num + 1]
			route.append(self.customer_num + 1)
			distances.append(temp_dis)
			demands.append(temp_load)
			routes.append(route)
			arrive_times_vectors.append(arrive_time_vector[:])

		self.add_column(routes)

	def start(self):
		self.problem_csv()
		self.pre_press()
		self.set_cover()
		# self.initial_routes_generates()

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
			self.routes[temp_length]['var'] = self.rmp.addVar(column=added_column, obj=self.routes[temp_length]['distance'],ub = 1,lb = 0)


def problem_read(path):
	customers = {}
	customer_number = 0
	capacity = 0
	best_know = 100000

	for line in open(path):
		temp = line.split(' ')
		if len(temp) == 2:
			if not customer_number:

				customer_number = int(temp[0])
				best_know = float(temp[1])
			else:
				customers[0] = {}
				customers[0]['loc'] = [float(temp[0]), float(temp[1])]
				customers[0]['demand'] = 0
		elif len(temp) == 1:
			capacity = int(temp[0])

		elif len(temp) == 4:
			l = len(customers)
			customers[l] = {}
			customers[l]['loc'] = [float(temp[1]), float(temp[2])]
			customers[l]['demand'] = int(temp[3])

	# virtual depot for return
	l = len(customers)
	customers[l] = {}
	customers[l]['loc'] = customers[0]['loc']
	customers[l]['demand'] = customers[0]['demand']

	return customers, capacity, customer_number, best_know





def history_routes_load(rmp, routes):
	# hwo to store the route generated
	with open('../problem1_route.pkl', 'rb') as pkl2:
		new_routes = pickle.load(pkl2)

	n = len(new_routes)

	for i in range(51, n):
		v = new_routes[i]
		m = len(routes) + 1
		routes[m] = {}

		routes[m]['demand'] = v['demand']
		routes[m]['column'] = v['column']
		routes[m]['distance'] = v['distance']
		routes[m]['route'] = v['route']

		added_column = gp.Column(routes[m]['column'], rmp.getConstrs())

		routes[m]['var'] = rmp.addVar(column=added_column, obj=routes[m]['distance'])

	rmp.update()

	return rmp, routes


def plot(path, customers):
	pre = 0
	for cus in path:
		plt.plot((customers[pre]['loc'][0], customers[cus]['loc'][0]),
				 (customers[pre]['loc'][1], customers[cus]['loc'][1]))
		plt.text(customers[pre]['loc'][0], customers[pre]['loc'][1], str(pre), verticalalignment='bottom')
		pre = cus
	plt.show()








def main(path,num):
	solver = Solver(path,num)
	solver.start()
	dual = solver.step()

	t = time.time()

	objs, paths = labeling_Algoithm_vrptw.labeling_algorithm(dual, solver.dis, solver.customers, solver.capacity, solver.num)
	print(time.time()-t)
	exit()

	# plot(paths[0],solver.customers)
	while objs[0] < -(1e-1):
		solver.add_column(paths)
		dual = solver.step()
		objs, paths = labeling_Algoithm_vrptw.labeling_algorithm(dual, solver.dis, solver.customers, solver.capacity, solver.num)
		print(objs[0])


	for key in solver.routes.keys():
		solver.routes[key]['var'].vtype = GRB.BINARY
	for con in solver.rmp.getConstrs():
		con.sense = '='

	solver.rmp.update()
	solver.rmp.optimize()
	temp = []
	for key in solver.routes.keys():
		if solver.routes[key]['var'].x > 0:
			print(solver.routes[key]['route'])
			temp += solver.routes[key]['route'][:-1]
	temp.sort()
	print(temp)
	print(solver.rmp.objval)


if __name__ == '__main__':
	main('../data/R101_200.csv', 100)
