import gurobipy as gp
from gurobipy import GRB

def tree_search(customers,dis,target_set, ub,cur_path,capacity):
	print(cur_path)
	if len(cur_path)>10:
		a = 10
	tree = Tree(customers,dis,ub,target_set,cur_path,capacity)
	tree.start()

	if tree.best_path[0] == tree.best_path[1] == 0:
		path1 = tree.best_path[1:]
		path2 = None
	elif tree.best_path[-1] == len(customers)-1 and tree.best_path[-1] ==0:
		path1 = tree.best_path[0:-2] + [tree.best_path[-1]]
		path2 = None
	else:
		n = len(tree.best_path)
		for index in range(1,n):
			if tree.best_path[index] == 0:
				break
		path1 = tree.best_path[0:index] + [len(customers)-1]
		path2 = tree.best_path[index:]
	print(path1,path2)
	print('improvement:%f',ub-tree.ub)
	return path1,path2


class Node(object):
	def __init__(self):
		self.path = []
		self.time = 0
		self.load = 0
		self.total_dis = 0
		self.flag = False
		self.rest_cus = set()
		self.unreachable = set()

	def expand(self,cus,cus_info,arrive_time,temp_dis):
		new_node = Node()
		new_node.path = self.path + [cus]
		if cus == 0:
			new_node.flag = True
		else:
			new_node.load = self.load + cus_info['demand']
			new_node.time = max(cus_info['start'], arrive_time)
			new_node.flag = self.flag

		new_node.total_dis = self.total_dis + temp_dis
		new_node.rest_cus.update(self.rest_cus)
		new_node.rest_cus.remove(cus)
		return new_node

class Tree(object):
	def __init__(self,customers, dis,ub,target_set,cur_path,capacity):
		self.dis = dis
		self.customers = customers
		self.ub = ub
		self.target_set = target_set
		self.capacity = capacity
		self.best_path = cur_path


	def search(self,node):
		temp_list = [(cus,node.time + self.customers[node.path[-1]]['service'] + self.dis[node.path[-1],cus],node.load + self.customers[cus]['demand']) for cus in node.rest_cus]
		un_reach = [(cus,arrive_time,load) for cus,arrive_time,load in temp_list if arrive_time>self.customers[cus]['end'] or load>self.capacity]
		reach_able = [(cus,arrive_time,load) for cus,arrive_time,load in temp_list if arrive_time<=self.customers[cus]['end'] or load<=self.capacity]
		if node.flag and un_reach:
			return
		temp = self.target_set - set(node.path[1:] + [len(self.customers)-1])
		# obj,path  = self.bound(node.path[-1],len(self.customers)-1,temp)
		#
		# if node.total_dis+obj>self.ub:
		# 	return

		for cus,arrive_time,load in reach_able:
			new_node = node.expand(cus, self.customers[cus], arrive_time, self.dis[node.path[-1], cus])
			if new_node.path[-1] == len(self.customers) - 1:
				if not new_node.rest_cus:
					if new_node.total_dis < self.ub:
						self.ub = new_node.total_dis
						self.best_path = new_node.path
			else:
				self.search(new_node)

	def bound(self,start,end,target):
		model = gp.Model()
		model.Params.logtoconsole = 0
		n = len(target)
		points = [start] + list(target) + [end]
		vars = model.addVars(n+2,n+2,vtype=GRB.BINARY)
		obj = gp.quicksum([self.dis[points[i],points[j]]*vars[i,j] for i in range(n+2) for j in range(n+2)])
		model.setObjective(obj,GRB.MINIMIZE)
		mu = model.addVars(n+2)


		model.addConstrs(gp.quicksum([vars[i,j] for j in range(1,n+2) if i!=j] )==1 for i in range(0,n+1))
		model.addConstrs(gp.quicksum([vars[i,j] for i in range(0,n+1) if i!=j])==1 for j in range(1,n+2))
		model.addConstrs(mu[i]-mu[j]+(n+3)*vars[i,j]<=n+2 for i in range(n+2) for j in range(n+2))
		model.update()
		model.optimize()
		path = [start]
		cur = 0
		while True:
			for j in range(n+2):
				if vars[cur,j].x>0:
					cur = j
					path.append(points[j])
			if path[-1] == end:
				break
		return model.objVal,path




	def start(self):
		root = Node()
		root.path = [0]
		root.rest_cus.update(self.target_set)

		self.search(root)

	def bounding_procedure(self):
		pass


if __name__ == '__main__':
	pass
