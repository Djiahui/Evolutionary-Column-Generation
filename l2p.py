import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import SPP
import labeling_Algoithm
import pickle


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


def path_eva(path, customers, dis,routes):
	total_demand = 0
	distance = 0
	pre = 0
	for customer in path:
		distance += dis[(pre, customer)]
		pre = customer
		total_demand += customers[customer]['demand']

	column = [1 if i in path else 0 for i in range(1, len(customers) - 1)]

	n = len(routes)+1
	routes[n] = {}
	routes[n]['demand'] = total_demand
	routes[n]['distance'] = distance
	routes[n]['cloumn'] = column
	routes[n]['route'] = path

	return column, total_demand, distance,routes


def set_cover(customers, capacity, number, dis):
	rmp = gp.Model('rmp')
	rmp.Params.logtoconsole = 0
	routes = {}
	for i in range(number):
		index = i+1
		column, total_demand, distance,routes = path_eva([index, customer_number + 1], customers, dis,routes)
		routes[len(routes)]['var'] = rmp.addVar(ub=1, lb=0, obj=distance, name='x')

	cons = rmp.addConstrs(routes[index]['var'] == 1 for index in range(1, number + 1))

	rmp.update()

	return rmp, routes


def history_routes_load(rmp, routes):
	# hwo to store the route generated
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
	# rmp, routes = history_routes_load(rmp, routes)
	print(len(routes))
	rmp.optimize()

	dual = rmp.getAttr(GRB.Attr.Pi, rmp.getConstrs())

	sub_obj = []

	# obj,path = SPP.price_problem(dual, dis, customers, capacity, customer_number)
	obj,path = labeling_Algoithm.labeling_algorithm(dual, dis, customers, capacity, customer_number)
	print(obj,path)
	obj,path = SPP.spp(dual, dis, customers, capacity, customer_number)
	print(obj, path)
	while obj < 0:
		column, total_demand, distance,routes = path_eva(path, customers, dis,routes)
		print(obj, column)
		sub_obj.append(obj)
		if len(sub_obj) > 2:
			if sub_obj[-1] == sub_obj[-2]:
				print('strange')
				del routes[len(routes)]
				break
		added_column = gp.Column(column, rmp.getConstrs())

		routes[len(routes)]['var'] = rmp.addVar(column=added_column, obj=distance)
		rmp.optimize()
		dual = rmp.getAttr(GRB.Attr.Pi, rmp.getConstrs())
		print([routes[i]['var'].x for i in range(1, len(routes) + 1)])
		obj,path = SPP.spp(dual, dis, customers, capacity, customer_number)

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
