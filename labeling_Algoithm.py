class Label(object):
	def __init__(self):
		self.path = []
		self.demand = []
		self.dis = []
		self.dominated = False


def dominate(new_label, label_list, queue):
	for label in label_list:
		if label.dis <= new_label.dis and label.demand < new_label.demand:
			new_label.dominated = True
			break
		else:
			if new_label.dis <= label.dis and new_label.demand <= label.demand:
				label.dominated = True
	if not new_label.dominated:
		queue.append(new_label)
		label_list.append(new_label)

		queue = list(filter(lambda x: not x.dominated, queue))
		label_list = list(filter(lambda x: not x.dominated, label_list))
	else:
		return queue,label_list

	return queue, label_list


def labeling_algorithm(dis, customers, capacity, customer_number):
	customer_list = [i for i in range(customer_number + 2)]
	label = Label()
	label.path = [0]
	label.dis = 0
	label.demand = 0

	queue = [label]

	path_dic = {}

	while len(queue) > 0:
		current = queue.pop()

		last_node = current.path[-1]
		if last_node == customer_number + 1:
			continue
		for customer in customer_list:
			if customer in current.path:
				continue

			load = current.demand + customers[customer]['demand']
			if load <= capacity:
				new_label = Label()
				new_label.path = current.path[:] + [customer]
				new_label.demand = current.demand + customers[customer]['demand']
				new_label.dis = current.dis + dis[last_node, customer]

				if customer in path_dic:
					print('?')
					queue, path_dic[customer] = dominate(new_label, path_dic[customer], queue)


				else:
					# print('??')
					# exit()
					path_dic[customer] = [new_label]
					# print('???')
					# exit()
					queue.append(new_label)


	final_labels = path_dic[customer_number + 1]
	min_cost = 100000
	best_label = None
	for label in final_labels:
		if label.dis < min_cost:
			min_cost = label.dis
			best_label = label

	return best_label