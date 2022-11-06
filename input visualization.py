import numpy as np
import argparse
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
config = {
"font.family":'serif',
"font.size": 10,
"mathtext.fontset":'stix',
"font.serif": ['SimSun'],#SimSun
}
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
		 'style':'normal',
         'size': 30,
}
font2 = {'family': 'Times New Roman',
         'weight': 'bold',
		 'style':'normal',
         'size': 40,
}
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/electricity.txt', # solar_AL 137  exchange_rate 8   electricity 321     traffic 862
                    help='location of the data file')
args = parser.parse_args()
fin = open(args.data)
rawdat = np.loadtxt(fin, delimiter=',')
start = 0
datarange = start+100+1
indrange = np.arange(start, datarange)
indrange1 = np.arange(start, datarange-10)
data = rawdat[indrange, :]
datalen = np.arange(len(indrange))

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(12.7,12))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.05)
#------------output-----------------
for i in range(5):
    ax[i].plot(indrange, data[:, i], "--", color="black", linewidth=6) #蓝
    ax[i].plot(indrange1, data[:-10, i], color="black", linewidth=6)  # 蓝
    ax[i].spines['top'].set_visible(False)  # 去掉上边框
    ax[i].spines['right'].set_visible(False)  # 去掉右边框
#------------input-----------------
#改成100
# for i in range(5):
#     ax[i].plot(indrange, data[:, i], "--", color="black", linewidth=6) #蓝 #3778bf
#     ax[i].plot(indrange, data[:, i], color="black", linewidth=6)  # 蓝
#     ax[i].spines['top'].set_visible(False)  # 去掉上边框
#     ax[i].spines['right'].set_visible(False)  # 去掉右边框
plt.show()
