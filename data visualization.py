import numpy as np
import argparse

import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
config = {
"font.family":'serif',
"font.size": 20,
"mathtext.fontset":'stix',
"font.serif": ['Times New Roman'],#SimSun
}
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
		 'style':'normal',
         'size': 40,
}
plt.style.use('seaborn-darkgrid')
plt.rcParams.update(config)
matplotlib.rc('axes', grid = False)
#plt.rc('font',family='Times New Roman') #全局变为Times

plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/traffic.txt', # solar_AL 137  exchange_rate 8   electricity 321     traffic 862
                    help='location of the data file')
args = parser.parse_args()
fin = open(args.data)
rawdat = np.loadtxt(fin, delimiter=',')
start = 720
range = start+350+1
indrange = np.arange(start, range)
data = rawdat[indrange, :]
xrange = np.arange(len(indrange))

plt.figure(figsize=(9,6))
ax=plt.axes()
plt.grid(linestyle = "--") #设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(True) #去掉上边框
ax.spines['right'].set_visible(True) #去掉右边框
# plt.style.use('ggplot') #带灰色背景


plt.plot(indrange,data[:,0],color="slateblue",linestyle='-',label="A",linewidth=1.5) #蓝 slateblue silver
plt.plot(indrange,data[:,800],color="#3778bf",linestyle='-.',label="B",linewidth=1.5) #紫 #3778bf #3778bf
plt.plot(indrange,data[:,1],color="#db5856",linestyle='--',label="C",linewidth=1.5)    #红 #db5856 gray
plt.plot(indrange,data[:,100],color="#cba560",linestyle='-',label="D",linewidth=1.5) #黄 #cba560 black
plt.plot(indrange,data[:,20],color="#56ae57",linestyle=':',label="E",linewidth=1.5)  #绿  #06b1c4 #56ae57 dimgray


plt.xticks(indrange,xrange,fontsize=20,fontweight='normal') #默认字体大小为10
plt.yticks(fontsize=20,fontweight='normal',rotation = 45)
#plt.title("Electricity dataset",fontsize=15, fontweight='bold') #默认字体大小为12
plt.xlabel("时间步（小时）",fontproperties='STZhongsong',fontsize=30,fontweight='bold')
plt.ylabel("观测值",fontproperties='STZhongsong',fontsize=30,fontweight='bold')

# plt.legend(loc=0, numpoints=1) # 图例
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=12,fontweight='bold') #设置图例字体的大小和粗细

ax.xaxis.set_major_locator(ticker.IndexLocator(50,0))
ax.set_title("(a) Traffic dataset",font1)

# plt.savefig('D:\\filename.png') #建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
plt.show()

#-------------------注释-------------------------------
# 序号    数据集        展示周期        蓝   紫   红   黄   绿
# （a）   Traffic      720+350       0   800   1   100  20
# （b）   Solar       300+1000       0    40   80  20   100
# （c）   Electricity   0+350        3    5    4    2    6
# （d）   Exchange      0+350        2    0    3    6    7