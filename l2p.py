import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import SPP
import labeling_Algoithm
import pickle


class du(object):
	def __init__(self, n):
		self.father = [i for i in range(n)]

	def find(self, x):
		if x == self.father[x]:
			return self.father[x]
		self.father[x] = self.find(self.father[x])
		return self.father[x]

	def union(self, x, y):
		if self.find(x) == self.find(y):
			return
		self.father[self.find(x)] = self.find(y)


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


def dis_calcul(customers, number):
	dis = {}
	graph = []
	for i in range(number + 2):
		for j in range(number + 2):
			if i == j:
				continue
			if i == 0 and j == number + 1:
				dis[(i, j)] = 1e6
				continue
			if i == number + 1 and j == 0:
				dis[(i, j)] = 1e6
			temp = [customers[i]['loc'][0] - customers[j]['loc'][0], customers[i]['loc'][1] - customers[j]['loc'][1]]
			dis[(i, j)] = math.sqrt(temp[0] * temp[0] + temp[1] * temp[1])
	return dis


def path_eva(path, customers, dis):
	total_demand = 0
	distance = 0
	pre = 0
	for customer in path:
		distance += dis[(pre, customer)]
		pre = customer
		total_demand += customers[customer]['demand']

	column = [1 if i in path else 0 for i in range(1, len(customers) - 1)]

	return column, total_demand, distance


def set_cover(customers, capacity, number, dis):
	rmp = gp.Model('rmp')
	routes = {}
	for i in range(number):
		index = i + 1
		column, total_demand, distance = path_eva([index, customer_number + 1], customers, dis)
		routes[index] = {}
		routes[index]['column'] = column
		routes[index]['demand'] = total_demand
		routes[index]['distance'] = distance

		routes[index]['var'] = rmp.addVar(ub=1, lb=0, obj=distance, name='x')

	cons = rmp.addConstrs(routes[index]['var'] == 1 for index in range(1, number + 1))

	rmp.update()

	return rmp, routes


def sub_cycle(model, where):
	if where == GRB.Callback.MIPSOL:
		edges = []
		vals = model.cbGetSolution(model._vars)
		num = 0
		for (i, j) in vals.keys():
			if i > num:
				num = i
			if vals[i, j] > 0.5:
				edges.append([i, j])

		cycles = find_cycle(edges, num - 1)

		for cycle in cycles:
			model.cbLazy(
				gp.quicksum(model._vars[i, j] for i, j in cycle) <= len(cycle) - 1)


def find_cycle(edges, customer_number):
	d = du(customer_number + 2)
	for edge in edges:
		d.union(edge[0], edge[1])

	dic = {}

	for edge in edges:
		for x in edge:
			if d.find(x) in dic:
				if x not in dic[d.find(x)]:
					dic[d.find(x)].append(x)
			else:
				dic[d.find(x)] = [x]

	cycles = []

	for father, childs in dic.items():
		edges_in_cycles = []
		if not 0 in childs:
			for edge in edges:
				if edge[0] in childs:
					edges_in_cycles.append(edge)
		if edges_in_cycles:
			cycles.append(edges_in_cycles)
	return cycles


def price_problem(pi, dis, customers, capacity, customer_number):
	# pi   (1,cus_num)

	## 如果有负权就有回环
	new_pi = [0] + pi
	sub_model = gp.Model('price_model')
	sub_model.Params.logtoconsole = 0
	vars = sub_model.addVars(customer_number + 2, customer_number + 2, vtype=GRB.BINARY, name='x')

	obj = gp.quicksum(
		(dis[i, j] - new_pi[i]) * vars[i, j] for i in range(customer_number + 1) for j in
		range(customer_number + 2) if i != j)

	# 应该是要考虑0，0位置上的对偶为0

	sub_model.setObjective(obj)

	sub_model.addConstr(
		gp.quicksum(customers[i]['demand'] * vars.sum(i, '*') for i in range(1, customer_number + 1)) <= capacity)
	sub_model.addConstr(gp.quicksum(vars[0, i] for i in range(1, customer_number + 1)) == 1)
	sub_model.addConstr(gp.quicksum(vars[i, customer_number + 1] for i in range(1, customer_number + 1)) == 1)
	sub_model.addConstrs(vars.sum(h, '*') == vars.sum('*', h) for h in range(1, customer_number + 1))
	sub_model.addConstrs(vars.sum(h, '*') <= 1 for h in range(1, customer_number + 1))
	sub_model.addConstr(vars.sum('*', 0) == 0)
	sub_model.addConstr(vars.sum(customer_number + 1, '*') == 0)
	sub_model.update()

	sub_model._vars = vars
	sub_model.Params.lazyConstraints = 1

	sub_model.optimize(sub_cycle)
	sub_model.write('temp.lp')
	path = []
	obj = sub_model.objval
	dic = {}
	for i in range(customer_number + 2):
		for j in range(customer_number + 2):
			if vars[i, j].x > 0.5:
				dic[i] = j
				plt.plot((customers[i]['loc'][0], customers[j]['loc'][0]),
						 (customers[i]['loc'][1], customers[j]['loc'][1]))
				plt.text(customers[i]['loc'][0], customers[i]['loc'][1], str(i), verticalalignment='bottom')
	plt.show()

	def dfs(x):
		if x not in dic:
			return
		path.append(dic[x])
		dfs(dic[x])

	dfs(0)

	return obj, path


def floyd(graph):
	length = len(graph)
	path = {}

	for i in range(length):
		path.setdefault(i, {})
		for j in range(length):
			if i == j:
				continue

			path[i].setdefault(j, [i, j])
			new_node = None

			for k in range(length):
				if k == j:
					continue

				new_len = graph[i][k] + graph[k][j]
				if graph[i][j] > new_len:
					graph[i][j] = new_len
					new_node = k
			if new_node:
				path[i][j].insert(-1, new_node)

	return graph, path


def history_routes_load(rmp, routes):
	with open('problem1_route.pkl', 'rb') as pkl2:
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


def main(customers, capacity, customer_number, dis):
	rmp, routes = set_cover(customers, capacity, customer_number, dis)
	rmp, routes = history_routes_load(rmp, routes)
	print(len(routes))
	routes_store = {}
	rmp.optimize()

	dual = rmp.getAttr(GRB.Attr.Pi, rmp.getConstrs())

	sub_obj = []

	# obj,path = price_problem(dual, dis, customers, capacity, customer_number)
	final_label = labeling_Algoithm.labeling_algorithm(dual, dis, customers, capacity, customer_number)
	obj, path = SPP.spp(dual, dis, customers, capacity, customer_number)
	while obj < 0:
		column, total_demand, distance = path_eva(path, customers, dis)
		print(obj, column)
		sub_obj.append(obj)
		if len(sub_obj) > 2:
			if sub_obj[-1] == sub_obj[-2]:
				print('strange')
				break
		l = len(routes) + 1  # 这里没有从0开始索引
		routes[l] = {}
		routes[l]['column'] = column
		routes[l]['demand'] = total_demand
		routes[l]['distance'] = distance
		routes[l]['route'] = path

		added_column = gp.Column(column, rmp.getConstrs())

		routes[l]['var'] = rmp.addVar(column=added_column, obj=distance)
		rmp.optimize()
		dual = rmp.getAttr(GRB.Attr.Pi, rmp.getConstrs())
		print([routes[i]['var'].x for i in range(1, l + 1)])
		obj, path = SPP.spp(dual, dis, customers, capacity, customer_number)

	for key in routes.keys():
		routes[key]['var'].vtype = GRB.BINARY
	for con in rmp.getConstrs():
		con.sense = '='

	rmp.update()
	rmp.optimize()
	for key in routes.keys():
		if routes[key]['var'].x > 0:
			print(routes[key]['route'])
	print(rmp.objval)


if __name__ == '__main__':
	start = time.time()
	customers, capacity, customer_number, best_know = problem_read('problem.txt')
	dis = dis_calcul(customers, customer_number)
	main(customers, capacity, customer_number, dis)
	print(time.time() - start)
