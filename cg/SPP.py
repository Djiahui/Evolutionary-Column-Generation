import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


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

def spp(pi, dis, customers, capacity, customer_number):
	# pi   (1,cus_num)

	## 如果有负权就有回环
	new_pi = [0] + pi
	sub_model = gp.Model('price_model')
	sub_model.Params.logtoconsole = 0
	vars = sub_model.addVars(customer_number + 2, customer_number + 2, vtype=GRB.BINARY, name='x')
	mu = sub_model.addVars(customer_number+2,vtype=GRB.CONTINUOUS,lb=0,name='mu')

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

	sub_model.addConstrs(mu[i]-mu[j]+customer_number*vars[i,j] <= (customer_number-1) for i in range(customer_number+1) for j in range(1,customer_number+2))
	sub_model.update()

	# sub_model._vars = vars
	# sub_model.Params.lazyConstraints = 1

	sub_model.optimize()
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

	assert len(path)==len(dic)

	return obj, path

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



if __name__ == '__main__':
	pass
