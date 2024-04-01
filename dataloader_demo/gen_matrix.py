import torch
import torch.utils.data as data


def readFile(fileName):  # 参数为文件的路径
    data_all = []
    f = open(fileName, mode='r', encoding='utf-8')
    line = f.readline()
    while line:
        if 'io_device_gpio' in line:  # 先筛选提取每行中的数据
            data_line = []
            for i in range(8):
                data = line[i-9] # 依次读最后8位，注意[-1]是 /n
                data_line.append(int(data))
            data_all.append(data_line)
        line = f.readline() 
    f.close()
    return data_all

class gen_matrix_dataset(data.Dataset):
    def __init__(self, filename, t, pad):
        super(gen_matrix_dataset, self).__init__()
        self.data_all = readFile(filename) # 读取全部数据(list)
        self.lines_num = len(self.data_all)
        print(f'lines num: {self.lines_num}')
        assert 0 <= pad <= t
        self.t = int(t)
        self.step = int(t-pad)
        setp_num = torch.tensor((self.lines_num-self.t) / self.step)
        self.matrixes_num = int(torch.trunc(setp_num)) + 1 # 计算总矩阵个数
        print(f'matrixes num: {self.matrixes_num}')
        

    def __len__(self):
        return self.matrixes_num
    
    def __getitem__(self, idx):
        if idx < self.matrixes_num:
            list_data = self.data_all[idx*self.step: idx*self.step + self.t]
            matrix_data = torch.tensor(list_data)
        else:
            raise RuntimeError('index error')
        return matrix_data


if __name__ == '__main__':
    train_set = gen_matrix_dataset('E05output.txt', t=4, pad=3)
    train_loader = data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=8)
    for i, train_data in enumerate(train_loader):
        print(f'{i}: {train_data}')