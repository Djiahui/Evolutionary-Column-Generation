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

    def dominated_determine(self,other):
        if self.dis>=other.dis and self.demand>=other.demand and self.time>=other.time:
            return True and other.unreachable_cus.issubset(self.unreachable_cus)
        else:
            return False


def dominate(new_label, label_list):
    for label in label_list:
        if new_label.dominated_determine(label):
            new_label.dominated = True
            return label_list


    res = []
    for label in label_list:
        if not label.dominated_determine(new_label):
            res.append(label)
        else:
            label.dominated = False

    res.append(new_label)

    return res



def labeling_algorithm(pi, dis, customers, capacity, customer_number):
    customer_list = set([i for i in range(customer_number + 2)])
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
        for customer in (customer_list-current.unreachable_cus):
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
                # the set of un_reachable_cus for current label is varying during the last iteration.
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


    return [best_label.dis], [best_label.path[1:]]
