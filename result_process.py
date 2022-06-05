import matplotlib.pyplot as plt
import random
import csv
import pickle
import numpy as np
def data_read():
	f_false = csv.reader(open('ECG/result_False.csv','r'))
	f_true = csv.reader(open('ECG/result_True.csv','r'))

	result = {}

	for line in f_false:
		name = line[0].split('.')[0].split('_')[0]
		item = line[1]
		if not name in result:
			result[name] = {}
		result[name][item] = {}
		result[name][item]['ECG-WOEL'] = []
		result[name][item]['ECG-WOE'] = []

		for temp in line[2:]:
			nums = temp.split(',')
			no_local = float(nums[0][1:])
			local = float(nums[1][:-1])
			result[name][item]['ECG-WOEL'].append(no_local)
			result[name][item]['ECG-WOE'].append(local)

	for line in f_true:
		name = line[0].split('.')[0].split('_')[0]
		item = line[1]
		print(name)
		result[name][item]['ECG-WOL'] = []
		result[name][item]['ECG'] = []

		for temp in line[2:]:
			nums = temp.split(',')
			no_local = float(nums[0][1:])
			local = float(nums[1][:-1])
			result[name][item]['ECG-WOL'].append(no_local)
			result[name][item]['ECG'].append(local)
	return result





def mean_time_obj(result):
	temp_time_dics = {}
	temp_obj_dics = {}
	problem_classes = ['R1','R2','C1','C2','RC1','RC2']
	for key in result.keys():
		obj_dics = result[key]['obj']
		time_dics = result[key]['time']
		if key[:2] in problem_classes:
			temp_name = key[:2]
		else:
			temp_name = key[:3]
		if temp_name not in temp_obj_dics:
			temp_obj_dics[temp_name] = {}
			temp_time_dics[temp_name]= {}
		temp_obj_dics[temp_name][key] = {}
		temp_time_dics[temp_name][key] = {}
		for alg in obj_dics.keys():
			print(key,alg)
			mean_obj = sum(obj_dics[alg])/len(obj_dics[alg])
			mean_time = sum(time_dics[alg])/len(time_dics[alg])
			temp_obj_dics[temp_name][key][alg] = mean_obj
			temp_time_dics[temp_name][key][alg] = mean_time


	algs = ['ECG-WOEL','ECG-WOE','ECG-WOL','ECG']
	algs_true = ['ECG-WOCL', 'ECG-WOC', 'ECG-WOL', 'ECG']
	for pro_class,datas in temp_obj_dics.items():
		max_value = 0
		min_value = 1e6
		for problem,data in datas.items():
			temp_data = []
			for alg in algs:
				temp_data.append(data[alg])
				max_value = max(data[alg],max_value)
				min_value = min(data[alg],min_value)
			temp_data2 = [x/max(temp_data) for x in temp_data]
			plt.plot(algs_true,temp_data2,marker = '*',label=problem)
		plt.legend()

		plt.savefig('ECG/'+pro_class+'.png')
		plt.show()

	for problem_classes, datas in temp_time_dics.items():

		for pro in datas.keys():
			temp_list = []
			for alg in algs:
				temp_list.append(datas[pro][alg])
			print([pro] + temp_list)


def improvement(result):
	dic = {}
	problem_classes = ['R1', 'R2', 'C1', 'C2', 'RC1', 'RC2']
	for key in result.keys():
		obj_dics = result[key]['obj']
		if key[:2] in problem_classes:
			temp_name = key[:2]
		else:
			temp_name = key[:3]
		if temp_name not in dic:
			dic[temp_name] = {}
		dic[temp_name][key] = {}

		improve_counter = 0
		for nel,ne in zip(obj_dics['ECG-WOEL'],obj_dics['ECG-WOE']):
			if nel-ne>1e-2:
				improve_counter += 1
		dic[temp_name][key]['ne'] = improve_counter/len(obj_dics['ECG-WOEL'])

		improve_counter = 0
		for el, e in zip(obj_dics['ECG-WOL'], obj_dics['ECG']):
			if el - e > 1e-2:
				improve_counter += 1
		dic[temp_name][key]['e'] = improve_counter / len(obj_dics['ECG'])


	for problem_classes in dic.keys():
		problem_list = []
		ne_list = []
		e_list = []


		plt.rcParams['axes.unicode_minus'] = False
		for problem,datas in dic[problem_classes].items():
			problem_list.append(problem)
			ne_list.append(datas['ne'])
			e_list.append(datas['e'])

		if problem_classes =='R1':
			a = 10

		bar_width = 0.35
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.bar(np.arange(len(problem_list)), ne_list, label='ECG-WOCL V.S. ECG-WOC', width=bar_width,ec='c', ls='-.', lw=1,hatch='/',color='skyblue')
		ax.bar( np.arange(len(problem_list))+ bar_width, e_list, label='ECG-WOL V.S. ECG', width=bar_width,ec='c', ls='-', lw=1,hatch='\\',color='pink')
			# 添加轴标签
		ax.set_xlabel('The '+problem_classes+' benchmark problems')
		ax.set_ylabel('The improvement rate')
		# 添加标题
		# ax.set_title('亿万财富家庭数Top5城市分布', fontsize=16)
		# 添加刻度标签
		plt.xticks(np.arange(len(problem_list)) + bar_width, problem_list)
		# 设置Y轴的刻度范围
		# plt.ylim([-0.1, 1.2])

		# # 为每个条形图添加数值标签
		# for x2016, y2016 in enumerate(Y2016):
		# 	plt.text(x2016, y2016 + 200, y2016, ha='center', fontsize=16)
		#
		# for x2017, y2017 in enumerate(Y2017):
		# 	plt.text(x2017 + 0.35, y2017 + 200, y2017, ha='center', fontsize=16)


		ax.legend()
		plt.savefig('ECG/'+problem_classes+'_improvement.png')

		plt.show()


def once_result_process(path):
	result = csv.reader(open('ECG/once.csv','r'))
	dic = {}

	for line in result:
		name = line[0].split('.')[0].split('_')[0]
		ress = [float(x) for x in line[1:]]
		if name[:2] in ['RC']:
			cls = 'RC'+name[2]
		else:
			cls = name[:2]
		if not cls in dic:
			dic[cls] = {}

		dic[cls][name] = ress

	for cls in ['RC1','RC2','R2','R1','C1','C2']:
		temp_res = dic[cls]
		for key,item in temp_res.items():
			plt.plot(range(len(item)), item, marker='*', label=key+str(sum(item)/len(item)))
		plt.legend()
		plt.title(cls)
		plt.show()

def once_result_compare(path1,path2):
	result1 = csv.reader(open('ECG/once.csv','r'))
	result2 = csv.reader(open('ECG/once2.csv', 'r'))
	dic1 = {}
	dic2 = {}

	for line in result1:
		name = line[0].split('.')[0].split('_')[0]
		ress = [float(x) for x in line[1:]]
		if name[:2] in ['RC']:
			cls = 'RC'+name[2]
		else:
			cls = name[:2]
		if not cls in dic1:
			dic1[cls] = {}

		dic1[cls][name] = ress

	for line in result2:
		name = line[0].split('.')[0].split('_')[0]
		ress = [float(x) for x in line[1:]]
		if name[:2] in ['RC']:
			cls = 'RC'+name[2]
		else:
			cls = name[:2]
		if not cls in dic2:
			dic2[cls] = {}

		dic2[cls][name] = ress

	for cls in ['RC1','RC2','R2','R1','C1','C2']:
		temp_res = dic1[cls]
		for key,item in temp_res.items():
			plt.plot(range(len(item)), item, marker='*', label='less')
			temp2 = dic2[cls][key]
			plt.plot(range(len(temp2)), temp2, marker='*', label='more')
			plt.legend()
			plt.title(key)
			plt.savefig('com/'+key+'.png')
			plt.show()

once_result_compare('ECG/once.csv','ECG/once2.csv')
# once_result_process('ECG/once.csv')


# mean_time_obj(data_read())
# improvement(data_read())
