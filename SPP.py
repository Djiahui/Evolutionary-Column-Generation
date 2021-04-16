import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


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






if __name__ == '__main__':
	pass
