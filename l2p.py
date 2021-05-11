import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import SPP
import labeling_Algoithm_vrptw
import pickle
import pandas as pd
import labeling_approach

def problem_csv(num):
	flag = False
	customers = {}


	with open('data/C101_200.csv') as p:
		for line in p:
			if not flag:
				flag = True
				pass

			else:
				temp = line.split(',')
				length = len(customers)
				customers[length] = {}
				customers[length]['loc'] = [float(temp[1]),float(temp[2])]
				customers[length]['demand'] = int(float(temp[3]))
				customers[length]['start'] = int(float(temp[4]))
				customers[length]['end'] = int(float(temp[5]))
				customers[length]['service'] = int(float(temp[6]))


				if length==num:
					length = len(customers)
					customers[length] = customers[0]
					break

	return customers, 200, num


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
			dis[(i, j)] = round(math.sqrt(temp[0] * temp[0] + temp[1] * temp[1]),2)
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
	routes[n]['column'] = column
	routes[n]['route'] = path

	return column, total_demand, distance,routes


def set_cover(customers, capacity, number, dis):
	rmp = gp.Model('rmp')
	rmp.Params.logtoconsole = 0
	routes = {}
	for i in range(number):
		index = i+1
		fea,routes = path_eva_vrptw([index, customer_number + 1],customers,capacity,dis,routes)
		if not fea:
			print('unfeasible',[index,customer_number])
		routes[index]['var'] = rmp.addVar(ub=1, lb=0, obj=routes[index]['distance'], name='x')

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

def plot(path,customers):
	pre = 0
	for cus in path:
		plt.plot((customers[pre]['loc'][0], customers[cus]['loc'][0]),
				 (customers[pre]['loc'][1], customers[cus]['loc'][1]))
		plt.text(customers[pre]['loc'][0], customers[pre]['loc'][1], str(pre), verticalalignment='bottom')
		pre = cus
	plt.show()

def path_eva_vrptw(path,customers,capacity,dis,routes):
	cost = 0
	pre = 0
	load = 0
	time = 0
	fea = True

	for cus in path:
		cost = cost+dis[pre,cus]
		time = time+dis[pre,cus]
		load = load+customers[cus]['demand']

		if cus==path[-1]:
			continue

		if time>customers[cus]['end']:
			fea = False
			return fea,routes
		if load >capacity:
			fea = False
			return fea,routes
		time = max(time,customers[cus]['start'])+customers[cus]['service']
		pre = cus


	if fea:
		column = [1 if i in path else 0 for i in range(1,len(customers)-1)]
		n = len(routes)+1
		routes[n] = {}
		routes[n]['demand'] = load
		routes[n]['distance'] = cost
		routes[n]['column'] = column
		routes[n]['route'] = path


	return fea,routes


def initial_routes_generates(customers, capacity, customer_number, dis):
	customer_list = [i for i in range(1,customer_number+1)]
	to_visit = customer_list[:]
	# customer_list = [10, 11, 14, 26, 66]
	# to_visit = [10, 11, 14, 26, 66]
	routes = []
	route = [0]
	temp_load = 0
	temp_time = 0

	while customer_list:
		for customer in customer_list:
			if customers[customer]['demand']+temp_load<capacity:
				temp = temp_time+dis[route[-1],customer]
				if temp<=customers[customer]['end']:
					temp_time = max(temp,customers[customer]['start'])+customers[customer]['service']
					temp_load = temp_load+customers[customer]['demand']
					route.append(customer)
					to_visit.remove(customer)
				else:
					if customer==customer_list[-1]:
						route.append(customer_number+1)
						routes.append(route[:])
						route = [0]
						temp_load = 0
						temp_time = 0
			else:
				if customer == customer_list[-1]:
					route.append(customer_number + 1)
					routes.append(route[:])
					route = [0]
					temp_load = 0
					temp_time = 0


		customer_list = to_visit[:]

	if len(route)>1:
		route.append(customer_number+1)
		routes.append(route)

	return routes



def main(customers, capacity, customer_number, dis):
	generated_routes = initial_routes_generates(customers,capacity,customer_number,dis)
	rmp, routes = set_cover(customers, capacity, customer_number, dis)

	for route in generated_routes:
		fea, routes = path_eva_vrptw(route[1:], customers, capacity, dis, routes)
		if not fea:
			print('unfeasibile',route[1:])
			continue
		temp_length = len(routes)
		added_column = gp.Column(routes[temp_length]['column'], rmp.getConstrs())
		routes[temp_length]['var'] = rmp.addVar(column=added_column, obj=routes[temp_length]['distance'])

	# rmp, routes = history_routes_load(rmp, routes)
	print(len(routes))
	rmp.optimize()

	dual = rmp.getAttr(GRB.Attr.Pi, rmp.getConstrs())

	sub_obj = []

	# obj,path = SPP.price_problem(dual, dis, customers, capacity, customer_number)
	# obj,path = SPP.spp(dual, dis, customers, capacity, customer_number)


	#obj,path = labeling_Algoithm_vrptw.labeling_algorithm(dual, dis, customers, capacity, customer_number)
	labeling_approach.t(dual,dis,customers,capacity,customer_number)
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
	customers,capacity,customer_number = problem_csv(50)
	dis = dis_calcul(customers, customer_number)
	main(customers, capacity, customer_number, dis)

	start = time.time()
	customers, capacity, customer_number, best_know = problem_read('problem.txt')
	dis = dis_calcul(customers, customer_number)
	main(customers, capacity, customer_number, dis)
	print(time.time() - start)
