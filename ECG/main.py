import entity
import csv
import os
from multiprocessing import Pool
def main(path,mode):
    temp = path.split('.')[0].split('_')
    cap = int(temp[1])
    num = int(temp[-1])
    name = 'result_' + str(mode) +'.csv'
    if os.path.exists(name):
        f = open(name,'a+',newline='')
    else:
        f = open(name,'w',newline='')
    wrt = csv.writer(f)
    objs = []
    times = []
    for _ in range(20):
        slover = entity.Solver('../data/'+path,num,cap)
        obj,time_use = slover.solve(mode)
        print(path + '---' + str(_) +'th---'+str(obj)+'---'+str(time_use))
        objs.append(obj)
        times.append(time_use)

    wrt.writerow([path,'obj']+objs)
    wrt.writerow([path,'time']+times)
    print(path+'-----done')

def multi_process_fun(path):
    temp = path.split('.')[0].split('_')
    cap = int(temp[1])
    num = int(temp[-1])
    objs = []
    times = []
    for _ in range(30):
        slover = entity.Solver('../data/' + path, num, cap)
        obj, time = slover.solve(False)
        objs.append(obj)
        times.append(time)
    print(path+'-----done')
    return path,objs,times




if __name__ == '__main__':
    mode = True

    for problem in os.listdir('../data'):
        if problem[0] != 'l' and problem[-1] == 'v':
            main(problem,mode)
    exit(0)


    # pool = Pool(10)
    # process_result = []
    # for problem in os.listdir('../data'):
    #     if problem[0] == 'R':
    #         print(problem)
    #         process_result.append(pool.apply_async(multi_process_fun,(problem,)))
    # pool.close()
    # pool.join()
    # result = []
    # for r in process_result:
    #     result.append(r.get())
    #
    # if os.path.exists('result_false_nolocal.csv'):
    #     f = open('result_false_nolocal.csv', 'a+', newline='')
    # else:
    #     f = open('result_false_nolocal.csv', 'w', newline='')
    #
    # wrt = csv.writer(f)
    # for name, objs, times in result:
    #     wrt.writerow([name, 'obj'] + objs)
    #     wrt.writerow([name, 'time'] + times)


