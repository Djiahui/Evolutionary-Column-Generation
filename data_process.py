import csv
import os


def data_process(path):
	i = 0
	temp_content = []
	pre_name = 'large_data/'
	for line in open(pre_name + path):

		if i == 4:
			capacity = int(line.split(' ')[-1][:-1])
		if i >= 7 and i!= 8:
			line_temp = line.split(' ')
			line_temp = [x for x in line_temp if x]
			line_temp[-1] = line_temp[-1][:-1]
			temp_content.append(line_temp)
		i += 1
	temp = path.split('.')[0].split('_')
	new_name = 'large' + temp[0].upper() +'0'+temp[-1]+ '_' + str(capacity) +'_'+ temp[1] + '00'
	with open('data/' + new_name + '.csv', 'w',newline='') as f:
		writer = csv.writer(f)
		writer.writerows(temp_content)
	print(path + '-------' + new_name)


if __name__ == '__main__':
	for file in os.listdir('large_data'):
		data_process(file)

