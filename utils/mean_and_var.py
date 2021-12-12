import numpy as np 

new = [
    87.13,
87.31,
78.75,
91.70,
]

def cal_mean_std(arr, domain):
    print('number:',len(arr))
    print("domain:", domain)
    #求均值
    arr_mean = np.mean(arr)
    #求总体方差
    arr_var = np.var(arr)
    #求总体标准差
    arr_std = np.std(arr)
    print("平均值为：%f" % arr_mean)
    # print("方差为：%f" % arr_var)
    print("标准差为:%f" % arr_std)

cal_mean_std(new, "ALL")