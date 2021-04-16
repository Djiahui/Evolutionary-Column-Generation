
class Label(object):
	def __init__(self):
		self.path = []
		self.demand = []
		self.dis = []


def dominate():
	pass

def labeling_algorithm(pi,dis,customers,capacity,customer_number):
	customer_list = [i for i in range(1,customer_number+1)]
	label = Label()
	label.path = [0]
	label.dis = 0
	label.demand = 0

	queue = [label]

	while len(queue)>0:
		current = queue.pop()

		last_node = current.path[-1]
		for customer in customer_list:
			if customer in current.path:
				continue


			load = current.demand + customers[customer]['demand']
			if load<=capacity:
				new_label = Label()
				new_label.path = current.path[:].append(customer)
				new_label.demand = current.demand+customers[customer]['demand']
				label
