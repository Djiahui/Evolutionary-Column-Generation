import time

import gurobipy as gp
from gurobipy import GRB

import math
import copy
import numpy as np
import pickle
import random

# 0 is not appear in path
class Individual(object):
	def __init__(self, path, dis):
		# deterministic
		self.path = path
		self.arrive_time_vector = []
		self.dis = dis

		# non-deterministic
		self.cost = 0
		self.demand = 0

		# for column generation
		self.init_route = False
		self.is_selected = False


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
		self.customers_set = set([i for i in range(len(customers))])
		self.initial_routes_generates()

		self.dual = None

		self.crossover_num = 100
		self.iteration_num = 10
		self.max_num_childres = customer_num//2

	def iteration(self,dual):
		temp_archive = []
		for pop in self.pops:
			mu_pop = self.mutation_operator(pop,dual)
			in_pop = self.insert_operator(pop,dual)
			if mu_pop:
				temp_archive.append(mu_pop)
			if in_pop:
				temp_archive.append(in_pop)
		for _ in range(self.crossover_num):
			p_pop1,p_pop2 = random.choices(self.pops,k=2)
			c_pop1, c_pop2 = self.crossover(p_pop1,p_pop2,dual)
			if c_pop1:
				temp_archive.append(c_pop1)
			if c_pop2:
				temp_archive.append(c_pop2)

		return temp_archive

	def evolute(self, dual):
		for iter in range(self.iteration_num):
			archive = self.iteration(dual)
			self.pops_update(archive)


	def pops_update(self, archive):
		self.pops = self.pops+archive
		self.pops.sort(key = lambda x:x.cost)
		self.pops = self.pops[:self.max_num_childres]




	def pt(self, path, dual):
		cur = 0
		dis_eva = 0
		cost_eva = 0
		time_eva = 0
		arrive_time = [0]
		for cus in path:
			arrive = time_eva+self.dis[cus,cur]
			if arrive > self.customers[cus]['end']:
				print('wrong' + str(cur))
				return
			else:
				time_eva = max(arrive, self.customers[cus]['start']) + self.customers[cus][
					'service']
				arrive_time.append(arrive)
				dis_eva += self.dis[cur, cus]
				cost_eva += (self.dis[cur, cus] - dual[cur])
			cur = cus
		demand = sum([customers[x]['demand'] for x in path[:-1]])
		if demand > self.capacity:
			print('wrong capacity')

		print(dis_eva, cost_eva,arrive_time)

	def path_eva(self, path, dual):
		cur = 0
		dis_eva = 0
		cost_eva = 0
		time_eva = 0
		arrive_time = [0]
		for cus in path:
			arrive = time_eva+self.dis[cus,cur]
			if arrive > self.customers[cus]['end']:
				return None,None,None
			else:
				time_eva = max(arrive, self.customers[cus]['start']) + self.customers[cus][
					'service']
				arrive_time.append(arrive)
				dis_eva += self.dis[cur, cus]
				cost_eva += (self.dis[cur, cus] - dual[cur])
			cur = cus

		return dis_eva,cost_eva,arrive_time

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
				if self.customers[customer]['demand'] + temp_load < self.capacity and arrive_time <= self.customers[customer]['end']:
					arrive_time_vector.append(arrive_time)
					departure_time = max(arrive_time, self.customers[customer]['start']) + self.customers[customer]['service']
					temp_dis += self.dis[route[-1], customer]
					temp_load = temp_load + self.customers[customer]['demand']
					route.append(customer)
					to_visit.remove(customer)
				elif customer == customer_list[-1]:
					arrive_time_vector.append(departure_time+self.dis[route[-1], self.customer_num + 1])
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

		for dis, path,arrive_time_vector,demand in zip(distances, routes,arrive_times_vectors,demands):
			self.pops.append(Individual(path, dis))
			self.pops[-1].arrive_time_vector = arrive_time_vector
			self.pops[-1].demand = demand

	def evaluate(self, dual):
		for pop in self.pops:
			pop.evaluate_under_dual(dual)
			pop.is_selected = True

	def mutation_operator(self,pop,dual):
		"""
		:type pop:ECG.entity.Individual
		:param pop: Individual
		:return:
		"""
		n = len(pop.path)

		index = random.randint(1,n-2)
		index_customer = pop.path[index]

		before_index, before_customer = index-1, pop.path[index-1]
		after_index,after_customer = index+1, pop.path[index+1]
		total_demand = pop.demand-self.customers[index_customer]['demand']

		start_service_time = max(pop.arrive_time_vector[before_index],self.customers[before_customer]['start'])
		departure_time = start_service_time + self.customers[before_customer]['service']
		selected_customers = self.feasible_customers_search(pop,before_customer,after_customer,total_demand,departure_time,after_index)
		if not selected_customers:
			return

		final_select = None
		final_improvement = 0
		basic_cost = self.dis[before_customer,index_customer]+self.dis[index_customer,after_customer]-dual[index_customer]
		for x in selected_customers:
			if basic_cost - (self.dis[before_customer,x]+self.dis[x,after_customer]-dual[x])>final_improvement:
				final_improvement = basic_cost - (self.dis[before_customer,x]+self.dis[x,after_customer]-dual[x])
				final_select = x

		if final_select:
			new_path = pop.path[:index]+[final_select]+pop.path[index+1:]
			new_cost = pop.cost-final_improvement
			new_demand = pop.demand + (self.customers[final_select]['demand']-self.customers[index_customer]['demand'])
			new_dis = pop.dis+ (self.dis[before_customer,final_select]+self.dis[final_select,after_customer]-self.dis[before_customer,index_customer]-self.dis[index_customer,after_customer])

			new_pop = Individual(new_path,new_dis)
			new_pop.arrive_time_vector = self.arrive_time_update(departure_time, new_path[index:], pop.arrive_time_vector[:index], before_customer)
			new_pop.demand = new_demand
			new_pop.cost = new_cost
			return new_pop
		else:
			return None

	def feasible_customers_search(self,pop,before_customer,after_customer,total_demand,departure_time,after_index):
			candidates = self.customers_set-self.customers[before_customer]['tabu']-set(pop.path)
			new_candidates = set([x for x in candidates if (total_demand+self.customers[x]['demand']<self.capacity) and (departure_time+self.dis[before_customer,x]<self.customers[x]['end'])])
			if not new_candidates:
				return


			threshold = min([self.customers[x]['end']-arrivetime for x,arrivetime in zip(pop.path[after_index:],pop.arrive_time_vector[after_index:])])

			def fea_deter(x):
				return max(departure_time+self.dis[before_customer,x],self.customers[x]['start'])+self.customers[x]['service']+self.dis[x,after_customer]<pop.arrive_time_vector[after_index]+threshold
			selected_customers = [x for x in new_candidates if fea_deter(x)]
			return selected_customers

	def arrive_time_update(self,departure,rest_customers,pre_arrivetime,pre_customer):
		for cus in rest_customers:
			temp_arrive = departure+self.dis[pre_customer,cus]
			pre_arrivetime.append(temp_arrive)
			departure = max(temp_arrive,self.customers[cus]['start']) + self.customers[cus]['service']
			pre_customer = cus
		return pre_arrivetime


	def insert_operator(self, pop, dual,index=-1):
		n = len(pop.path)
		if index == -1:
			index = random.randint(1,n-2)

		before_index,after_index = index,index+1
		before_customer,after_customer = pop.path[before_index], pop.path[after_index]
		total_demand = pop.demand

		start_service_time = max(pop.arrive_time_vector[before_index], self.customers[before_customer]['start'])
		departure_time = start_service_time + self.customers[before_customer]['service']
		selected_customers = self.feasible_customers_search(pop,before_customer,after_customer,total_demand,departure_time,after_index)
		if not selected_customers:
			return

		final_selected = None
		final_improvement = 0
		basic = self.dis[before_customer,after_customer]
		for x in selected_customers:
			if basic - (self.dis[before_customer,x]+self.dis[x,after_customer]-dual[x])>final_improvement:
				final_improvement = basic - (self.dis[before_customer,x]+self.dis[x,after_customer]-dual[x])
				final_selected = x

		if final_selected:
			new_path = pop.path[:index+1] + [final_selected] + pop.path[after_index:]
			new_cost = pop.cost - final_improvement
			new_demand = pop.demand + self.customers[final_selected]['demand']
			new_dis = pop.dis + (self.dis[before_customer,final_selected]+self.dis[final_selected,after_customer]-self.dis[before_customer,after_customer])

			new_pop = Individual(new_path,new_dis)
			new_pop.cost = new_cost
			new_pop.demand = new_demand
			new_pop.arrive_time_vector = self.arrive_time_update(departure_time,new_path[after_index:],pop.arrive_time_vector[:before_index+1],before_customer)

	def crossover(self,pop1,pop2,dual):
		"""
		:type pop1: Individual
		:type pop2: Individual
		:param pop1:
		:param pop2:
		:return:
		"""
		min_diff_1, demand_1 = self.data_pre(pop1)
		min_diff_2, demand_2 = self.data_pre(pop2)
		new_pop1 = self.crossover_operator(pop1,pop2,min_diff_2,demand_2,dual)
		new_pop2 = self.crossover_operator(pop2,pop1,min_diff_1,demand_1,dual)

		return new_pop1, new_pop2

	def data_pre(self, pop):
		"""
		:type pop Individual
		"""
		n = len(pop.path)
		min_diff = [0 for _ in range(n)]
		demand = [0 for _ in range(n)]
		for i in range(n-1,-1,-1):
			min_diff[i] =  (self.customers[pop.path[i]]['end']-pop.arrive_time_vector[i]) if i == n-1 else (min(min_diff[i+1],self.customers[pop.path[i]]['end']-pop.arrive_time_vector[i]))
			demand[i] = self.customers[pop.path[i]]['demand'] if i==n-1 else self.customers[pop.path[i]]['demand']+demand[i+1]
		return min_diff,demand

	def crossover_operator(self,pop1,pop2,min_diff_2,demand_2,dual,index = -1):
		"""
		:type pop1: Individual
		:type pop2: Individual
		"""
		path_len = len(pop1.path)
		if index == -1:
			index = random.randint(1,path_len-2)
		customer = pop1.path[index]
		departure = max(pop1.arrive_time_vector[index],self.customers[customer]['start'])+self.customers[customer]['service']
		total_demand = sum([self.customers[x]['demand'] for x in pop1.path[:index+1]])

		n = len(pop2.path)
		best_cost = 1e6
		best_dis = None
		best_arrive_time = None
		best_path = None
		best_demand = None
		for after_index in range(n-2,0,-1):
			if pop2.path[after_index] in pop1.path:
				break
			if total_demand+demand_2[after_index]<=self.capacity and departure + self.dis[customer,pop2.path[after_index]]<pop2.arrive_time_vector[after_index]+min_diff_2[after_index]:
				new_dis,new_cost,new_arrive_time = self.path_eva(pop1.path[:index+1]+pop2.path[after_index:],dual)
				if not new_arrive_time:
					# review
					print('wrong')
				if new_cost<best_cost:
					best_cost = new_cost
					best_demand = total_demand + demand_2[after_index]
					best_path = pop1.path[:index+1] + pop2.path[after_index:]
					best_arrive_time = new_arrive_time
					best_dis = new_dis
			else:
				# if this customer is infeasible, then customers before it are all infeasible
				break


		if best_path:
			new_pop = Individual(best_path,best_dis)
			new_pop.demand = best_demand
			new_pop.arrive_time_vector = best_arrive_time
			new_pop.cost = best_cost

			return new_pop
		else:
			return None


class MCTS(object):
	def __init__(self, dis, customers, capacity, customer_number):
		self.pi = None
		self.dis = dis
		self.customers = customers
		self.customer_number = customer_number
		self.capacity = capacity

		self.iteration = 100

	def path_eva(self, path, dual):
		cur = 0
		dis_eva = 0
		cost_eva = 0
		time_eva = 0
		arrive_time = [0]
		for cus in path:
			arrive = time_eva+self.dis[cus,cur]
			if arrive > self.customers[cus]['end']:
				return None,None,None
			else:
				time_eva = max(arrive, self.customers[cus]['start']) + self.customers[cus][
					'service']
				arrive_time.append(arrive)
				dis_eva += self.dis[cur, cus]
				cost_eva += (self.dis[cur, cus] - dual[cur])
			cur = cus

		return dis_eva,cost_eva,arrive_time


	def matrix_init(self,dual):
		self.rel_matrix = np.zeros((self.customer_number + 2, self.customer_number + 2))
		customer_set = set([ i for i in range(1,self.customer_number+2)])
		for customer, information in self.customers.items():
			candidates = customer_set-information['tabu']
			if customer == 0 :
				candidates -= {self.customer_number+1}
			dis_vec = [dual[customer]-self.dis[customer,to] for to in candidates]
			max_dis = max(dis_vec)
			min_dis = min(dis_vec)

			dis_vec = [(x-min_dis)/(max_dis-min_dis) if max_dis>min_dis else 1 for x in dis_vec]

			self.rel_matrix[customer,list(candidates)] = dis_vec

			self.rel_matrix[customer, list(information['tabu'])] = 0

	def find_path(self, dual):
		# MCTS_decoder
		root = Node(0, self.customers, self.rel_matrix, dual, self.dis, self.capacity)

		root.path.append(0)
		root.dual = dual
		root.dis = self.dis
		root.demand = 0
		root.capacity = self.capacity
		root.max_children=100

		dic = {}

		for _ in range(self.iteration):
			root.select()

		dis_eva,cost_eva,arrive_time_eva = self.path_eva(root.best_quality_route[1:],dual)
		new_ind = Individual(root.best_quality_route,dis_eva)
		new_ind.cost = cost_eva
		new_ind.arrive_time_vector = arrive_time_eva
		return new_ind




class Node(object):
	def __init__(self, index, customers, matrix, dual, dis, capacity):
		self.current = index
		self.current_dis = 0
		self.current_time = 0
		self.current_cost = 0

		self.customers = customers
		self.customer_list = set([i for i in range(len(customers))])
		self.dis = dis
		self.dual = dual
		self.demand = None
		self.capacity = capacity
		self.tabu = copy.deepcopy(self.customers[index]['tabu'])
		self.selected = set()

		self.children = []
		self.father = None
		self.max_children = 100

		self.c = 2

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

	def select(self):
		reachable_customers = self.candidate_get(self.current,self.selected,self.tabu,self.demand,self.current_time)
		if len(self.children) < self.max_children and len(reachable_customers) > 0:
			self.expand(reachable_customers)
		else:
			selected_index = np.argmax(list(map(lambda x: x.get_score(), self.children)))
			if self.children[selected_index].state == 'terminal':
				self.children[selected_index].backup()
			else:
				self.children[selected_index].select()

	def expand(self, reachable_customers):

		p = self.softmax(self.rel_matrix[self.current, list(reachable_customers)])
		reachable = int(np.random.choice(list(reachable_customers), size=1, replace=False, p=p)[-1])
		self.selected.add(reachable)

		# generate a new child
		new_child = Node(reachable, self.customers, self.rel_matrix, self.dual, self.dis, self.capacity)
		new_child.father = self

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

			# new_child.rollout_bfs()

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

	def rollout_bfs(self):
		rollout_path = self.path[:]
		rollout_set = set(rollout_path)
		rollout_customer = self.current

		rollout_dis = self.current_dis
		rollout_time = self.current_time
		rollout_cost = self.current_cost
		demand = self.demand
		rollout_tabu = copy.deepcopy(self.tabu)

		best_cost = 1e6
		best_path = None

		queue = [(rollout_path, rollout_set, rollout_customer,rollout_dis,rollout_time,rollout_cost,demand, rollout_tabu)]
		while queue:
			path, path_set, current_customer,current_dis,current_time,current_cost,current_demand,current_tabu = queue.pop(0)
			candidates = self.candidate_get(current_customer,path_set,current_tabu,current_demand,current_time)
			for candidate in candidates:
				can_path = path+[candidate]
				can_set = set(can_path)
				can_customer = candidate
				can_dis = current_dis + self.dis[current_customer,can_customer]
				can_time = max(current_time+self.dis[current_customer,can_customer],self.customers[can_customer]['start'])+self.customers[can_customer]['service']
				can_cost = current_cost + self.dis[current_customer,can_customer] -self.dual[current_customer]
				can_tabu = set()
				can_tabu.update(current_tabu)
				can_tabu.update(self.customers[can_customer]['tabu'])
				can_demand = current_demand + self.customers[can_customer]['demand']
				if candidate == len(self.customers) - 1 and can_cost< best_cost:
					best_path = can_path
					best_cost = can_cost
				else:
					queue.append((can_path,can_set,can_customer,can_dis,can_time,can_cost,can_demand,can_tabu))

		self.quality = best_cost
		self.visited_times += 1
		self.best_quality_route = best_path
		self.backup()

	def rollout(self):
		rollout_path = self.path[:]
		rollout_set = set(rollout_path)
		current_customer = self.current

		rollout_dis = self.current_dis
		rollout_time = self.current_time
		rollout_cost = self.current_cost
		demand = self.demand
		rollout_tabu = copy.deepcopy(self.tabu)
		while current_customer != len(self.customers) - 1:
			candidates = self.candidate_get(current_customer,rollout_set,rollout_tabu,demand,rollout_time)
			p = self.softmax(self.rel_matrix[current_customer, list(candidates)])
			next_customer = int(np.random.choice(list(candidates), size=1, replace=False, p=p)[-1])

			# next_customer = list(candidates)[np.argmax(self.rel_matrix[current_customer, candidates])]
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

	def candidate_get(self,in_customer,in_selected_set,in_tabu,in_demand,in_time):
		candidates = self.customer_list - in_tabu - in_selected_set
		new_tabu = set([x for x in candidates if (self.customers[x]['demand'] + in_demand > self.capacity) or (in_time + self.dis[in_customer, x] > self.customers[x]['end'])])
		candidates -= new_tabu
		in_tabu.update(new_tabu)
		return candidates

	def get_score(self):
		if not self.visited_times:
			a = 0
		if self.father.min_quality == self.father.max_quality:
			# only one children node for father thus only one choice
			return 1

		return math.sqrt(
			(math.log(self.father.visited_times) / self.visited_times))

		# return -(self.quality - self.father.min_quality) / (self.father.max_quality - self.father.min_quality) + \
		# 	   self.rel_matrix[self.father.current, self.current] + self.c * math.sqrt(
		# 	(math.log(self.father.visited_times) / self.visited_times))

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
		mat = self.initial_routes_generates()
		n = len(self.routes_archive)
		self.x = []
		for i in range(n):
			name = 'x' + str(i)
			self.x.append(self.rmp.addVar(ub=1,lb=0,obj = self.routes_archive[i].dis, name=name))
			self.routes_archive[i].init_route = True

		self.rmp.addConstrs(gp.quicksum(mat[i, j]*self.x[j] for j in range(n))==1 for i in range(self.customer_num))
		self.rmp.update()
		# self.rmp.write('test.lp')


		# for i in range(self.customer_num):
		# 	index = i + 1
		# 	fea = self.path_eva_vrptw([index, self.customer_num + 1])
		# 	if not fea:
		# 		print('unfeasible', [index, self.customer_num])
		# 	self.routes[index]['var'] = self.rmp.addVar(ub=1, lb=0, obj=self.routes[index]['distance'], name='x')
		#
		# cons = self.rmp.addConstrs(self.routes[index]['var'] == 1 for index in range(1, self.customer_num + 1))
		#
		# self.rmp.update()

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
				if self.customers[customer]['demand'] + temp_load < self.capacity and arrive_time <= self.customers[customer]['end']:
					arrive_time_vector.append(arrive_time)
					departure_time = max(arrive_time, self.customers[customer]['start']) + self.customers[customer]['service']
					temp_dis += self.dis[route[-1], customer]
					temp_load = temp_load + self.customers[customer]['demand']
					route.append(customer)
					to_visit.remove(customer)
				elif customer == customer_list[-1]:
					arrive_time_vector.append(departure_time+self.dis[route[-1], self.customer_num + 1])
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

		n = len(routes)
		mat = np.zeros((self.customer_num, n))
		self.routes_archive = []
		for i in range(n):
			self.routes_archive.append(Individual(routes[i], distances[i]))
			self.routes_archive[-1].arrive_time_vector = arrive_times_vectors[i]
			self.routes_archive[-1].demand = demands[i]
			for cus in routes[i][1:-1]:
				mat[cus - 1,i] = 1

		return mat

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

	def linear_relaxition_solve(self):
		self.rmp.optimize()
		dual = self.rmp.getAttr(GRB.Attr.Pi, self.rmp.getConstrs())
		return dual

	def add_column(self):
		for ind in self.new_added_column:
			column = [0 for _ in range(self.customer_num)]
			for cus in ind.path[1:-1]:
				column[cus-1] = 1
			gp_column = gp.Column(column,self.rmp.getConstrs)
			self.rmp.addVar(column = gp_column,obj = ind.dis,ub = 1,lb = 0)


		# for route in routes:
		# 	fea = self.path_eva_vrptw(route)
		# 	if not fea:
		# 		print('unfeasibile', route[1:])
		# 		continue
		# 	temp_length = len(self.routes)
		# 	added_column = gp.Column(self.routes[temp_length]['column'], self.rmp.getConstrs())
		# 	self.routes[temp_length]['var'] = self.rmp.addVar(column=added_column,
		# 													  obj=self.routes[temp_length]['distance'])

	def paths_generate(self, dual):
		self.population.evaluate(dual)
		self.mcts.matrix_init(dual)
		new_ind = self.mcts.find_path(dual)
		self.population.evolute(dual)

		self.new_added_column += self.population.pops
		self.new_added_column.append(new_ind)

		return None

	def solve(self):

		dual_cur = self.linear_relaxition_solve()
		dual_cur = [0] + dual_cur + [0]
		best_reduced_cost = 1e6

		self.new_added_column = []
		while best_reduced_cost > -(1e-1):
			vars = self.rmp.getVars()
			n = len(vars)
			m = len(self.routes_archive)
			self.population.pops = [self.routes_archive[i] for i in range(m) if vars[i].x > 1e-6]
			self.new_added_column = [self.new_added_column[j] for j in range(n) if vars[j+m].x>1e-6]
			self.population.pops += self.new_added_column

			self.paths_generate(dual_cur)
			self.add_column()

			dual_cur = self.linear_relaxition_solve()
		exit()





			# Now, the basic function is finished
			# 1. domination rule in MCTS is needed 2. hwo many routes should be generated in MCTS 3. the detail of adding column


if __name__ == '__main__':
	solver = Solver('../data/C101_200.csv', 100, 200)
	solver.solve()
	solver.paths_generate()
	exit()

	# # test for mcts
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
	dual = [round(x,2) for x in dual]
	t = time.time()
	pops = Population(customer_number,customers,dis,capacity)
	archive = []
	for pop in pops.pops:
		mu_pop = pops.mutation_operator(pop, dual)
		in_pop = pops.insert_operator(pop, dual)
		if mu_pop:
			archive.append(mu_pop)
		if in_pop:
			archive.append(in_pop)
	for _ in range(100):
		#[0, 7, 13, 19, 6, 0]
		#[0, 12, 22, 24, 77, 101]
		s_pop1,s_pop2 = random.choices(pops.pops,k=2)
		new_pop1,new_pop2 = pops.crossover(s_pop1,s_pop2,dual)
		if new_pop1:
			archive.append(new_pop1)
		if new_pop2:
			archive.append(new_pop2)
	print(time.time()-t)
	exit()


	t = time.time()
	mcts = MCTS(dis, customers, capacity, customer_number)
	mcts.matrix_init(dual)
	obj = mcts.find_path(dual)
	print(obj)
	print(time.time()-t)

	exit()
