

def read_config():
    configs = dict()
    fp = open('config.txt', 'r')
    str_config = fp.readlines()
    for line in str_config:
        line = line.strip('')
        if line.find('pos_path') >= 0:
            pos_path = line.split('=')[1].strip()
            configs['pos_path'] = pos_path
        if line.find('neg_path') >= 0:
            neg_path = line.split('=')[1].strip()
            configs['neg_path'] = neg_path

    return configs


def gen_file_for_grid(i, data_mat, label_mat):
    fp = open('/home/xsyin/libsvm-3.21/tools/data.txt', 'w')
    for j in xrange(0, i-1):
        if label_mat[j] == 1:
            s = '+1 '
        else:
            s = '-1 '
        for k in xrange(0, len(data_mat[j])):
            s = s+str(k+1)+':'+str(data_mat[j][k])+' '
        fp.write(s+'\n')
    fp.close()

config = read_config()
print config