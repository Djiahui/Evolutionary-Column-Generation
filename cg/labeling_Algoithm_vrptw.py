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
    count = 0
    it = 0
    while len(queue) > 0:

        current = queue.pop(0)
        if current.dominated:
            print('hh')
            continue
        count += 1
        it += 1
        if not it%500:
            print(len(queue))
        # if not count%100:
        #     print(count)




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
                    # path_dic[new_label.path[-1]].append(new_label)
                    path_dic[new_label.path[-1]] = dominate(new_label,path_dic[new_label.path[-1]])
                else:
                    path_dic[new_label.path[-1]] = [new_label]

            if not new_label.dominated:
                queue.append(new_label)


    final_labels = path_dic[customer_number + 1]
    min_cost = 100000
    best_label = None
    final_labels.sort(key = lambda x:x.dis)

    for label in final_labels:
        if label.dis < min_cost:
            min_cost = label.dis
            best_label = label

    return [x.dis for x in final_labels[:50]],[x.path for x in final_labels[:50]]

    # return [best_label.dis], [best_label.path[1:]]

if __name__ == '__main__':
    import pickle
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
    dual = [round(x, 2) for x in dual]
    costs,paths = labeling_algorithm(dual,dis,customers,capacity,customer_number)
    print(paths[0])
    print(costs[0])
    exit(0)
