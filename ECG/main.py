import entity
import csv
import os
from multiprocessing import Pool
def main(path):
    temp = path.split('.')[0].split('_')
    cap = int(temp[1])
    num = int(temp[-1])
    if os.path.exists('result.csv'):
        f = open('result.csv','a+',newline='')
    else:
        f = open('result.csv','w',newline='')
    wrt = csv.writer(f)
    objs = []
    times = []
    for _ in range(5):
        slover = entity.Solver('../data/'+path,num,cap)
        obj,time = slover.solve(False)
        print(path + '---' + str(_) +'th---'+str(obj)+'---'+str(time))
        objs.append(obj)
        times.append(time)

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
    for problem in os.listdir('../data'):
        if problem[:2] == 'R2':
            main(problem)
    exit(0)


    pool = Pool(10)
    process_result = []
    for problem in os.listdir('../data'):
        if problem[0] == 'R':
            print(problem)
            process_result.append(pool.apply_async(multi_process_fun,(problem,)))
    pool.close()
    pool.join()
    result = []
    for r in process_result:
        result.append(r.get())

    if os.path.exists('result.csv'):
        f = open('result.csv','a+',newline='')
    else:
        f = open('result.csv','w',newline='')

    wrt = csv.writer(f)
    for name, objs, times in result:
        wrt.writerow([name, 'obj'] + objs)
        wrt.writerow([name, 'time'] + times)


