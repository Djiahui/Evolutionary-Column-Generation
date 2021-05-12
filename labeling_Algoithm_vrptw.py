import time
class Label(object):
    def __init__(self):
        self.path = []
        self.demand = []
        self.dis = []
        self.dominated = False
        self.length = 0
        self.time = 0
        self.unreachable_cus = set()

    def expand(self,customers,customer,dis,last_node,new_pi):
        new_label = Label()
        new_label.path = self.path[:] + [customer]
        new_label.demand = self.demand + customers[customer]['demand']
        new_label.dis = self.dis + dis[last_node, customer] - new_pi[last_node]
        new_label.time = max(self.time + dis[last_node, customer], customers[customer]['start'])+customers[customer]['service']
        new_label.length = self.length + 1
        new_label.unreachable_cus.add(customer)


        return new_label


def dominate(new_label, label_list):
    for label in label_list:
        if new_label.length <= label.length:
            if label.dis <= new_label.dis and label.demand <= new_label.demand and label.time<=new_label.time:
                if new_label.unreachable_cus.issubset(label.unreachable_cus):
                    new_label.dominated = True
                    break
        elif new_label.length >= label.length:
            if new_label.dis <= label.dis and new_label.demand <= label.demand and new_label.time<= label.time:
                if label.unreachable_cus.issubset(new_label.unreachable_cus):
                    label.dominated = True
    if not new_label.dominated:
        label_list.append(new_label)
        label_list = list(filter(lambda x: not x.dominated, label_list))

    return label_list


def labeling_algorithm(pi, dis, customers, capacity, customer_number):
    t0 = time.time()
    customer_list = [i for i in range(customer_number + 2)]
    new_pi = [0] + pi + [0]
    label = Label()
    label.path = [0]
    label.dis = 0
    label.demand = 0
    label.length = 1

    queue = [label]
    queue[-1].unreachable_cus.add(0)

    path_dic = {}
    path_dic[customer_number+1] = []

    while len(queue) > 0:
        current = queue.pop(0)
        if current.dominated:
            continue



        last_node = current.path[-1]
        if last_node == customer_number + 1:
            continue

        temp_labels = []
        for customer in customer_list:
            if customer in current.unreachable_cus:
                continue
            if current.demand + customers[customer]['demand']<=capacity and current.time+dis[last_node,customer]<=customers[customer]['end']:
                temp_labels.append(current.expand(customers,customer,dis,last_node,new_pi))
            else:
                current.unreachable_cus.add(customer)

        for new_label in temp_labels:
            if new_label.path[-1]==customer_number+1:
                path_dic[customer_number+1].append(new_label)
                continue
            else:
                new_label.unreachable_cus.update(current.unreachable_cus)
                if new_label.path[-1] in path_dic:
                    path_dic[new_label.path[-1]] = dominate(new_label,path_dic[new_label.path[-1]])
                else:
                    path_dic[new_label.path[-1]] = [new_label]

            if not new_label.dominated:
                queue.append(new_label)

    final_labels = path_dic[customer_number + 1]
    min_cost = 100000
    best_label = None
    for label in final_labels:
        if label.dis < min_cost:
            min_cost = label.dis
            best_label = label

    print(time.time()-t0)
    exit()

    return best_label.dis, best_label.path[1:]
