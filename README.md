# FFANet
This is a PyTorch implementation of the paper: [基于多尺度特征融合和双注意力机制的时间序列预测]. 

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt
## Data Preparation
### Multivariate time series datasets

Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Uncompress them and move them to the data folder.


## Model Training

### Single-step

* Solar-Energy

```
#Horizon 3
python train_single_step.py --save ./model-solar-3.pt --data ./data/solar_AL.txt --num_nodes 137 --horizon 3 --batch_size 24 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 6
python train_single_step.py --save ./model-solar-6.pt --data ./data/solar_AL.txt --num_nodes 137 --horizon 6 --batch_size 24 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 12
python train_single_step.py --save ./model-solar-12.pt --data ./data/solar_AL.txt --num_nodes 137 --horizon 12 --batch_size 24 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 24
python train_single_step.py --save ./model-solar-24.pt --data ./data/solar_AL.txt --num_nodes 137 --horizon 24 --batch_size 24 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100

#sampling
python train_single_step.py --num_split 3 --save ./model-solar-sampling-3.pt --data ./data/solar_AL.txt --num_nodes 137 --batch_size 32 --epochs 100 --horizon 3
```
* Traffic 

```
#Horizon 3
python train_single_step.py --save ./model-traffic-3.pt --data ./data/traffic.txt --num_nodes 862 --horizon 3 --batch_size 16  --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 6
python train_single_step.py --save ./model-traffic-6.pt --data ./data/traffic.txt --num_nodes 862 --horizon 6 --batch_size 16  --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 12
python train_single_step.py --save ./model-traffic-12.pt --data ./data/traffic.txt --num_nodes 862 --horizon 12 --batch_size 16  --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 24
python train_single_step.py --save ./model-traffic-24.pt --data ./data/traffic.txt --num_nodes 862 --horizon 24 --batch_size 16  --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100

#sampling
python train_single_step.py --num_split 3 --save ./model-traffic-sampling-3.pt --data ./data/traffic --num_nodes 321 --batch_size 16 --epochs 100 --horizon 3
```

* Electricity

```
#Horizon 3
python train_single_step.py --save ./model-electricity-3.pt  --data ./data/electricity.txt --num_nodes 321 --horizon 3 --batch_size 16 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 6
python train_single_step.py --save ./model-electricity-6.pt  --data ./data/electricity.txt --num_nodes 321 --horizon 6 --batch_size 16 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 12
python train_single_step.py --save ./model-electricity-12.pt  --data ./data/electricity.txt --num_nodes 321 --horizon 12 --batch_size 16 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 24
python train_single_step.py --save ./model-electricity-24.pt  --data ./data/electricity.txt --num_nodes 321 --horizon 24 --batch_size 16 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100

#sampling 
python train_single_step.py --num_split 3 --save ./model-electricity-sampling-3.pt --data ./data/electricity.txt --num_nodes 321 --batch_size 16 --epochs 100 --horizon 3
```

* Exchange-Rate

```
#Horizon 3
python train_single_step.py --save ./model/model-exchange-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --horizon 3 --batch_size 32 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 6
python train_single_step.py --save ./model/model-exchange-6.pt --data ./data/exchange_rate.txt --num_nodes 8 --horizon 3 --batch_size 32 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 12
python train_single_step.py --save ./model/model-exchange-12.pt --data ./data/exchange_rate.txt --num_nodes 8 --horizon 3 --batch_size 32 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100
#Horizon 24
python train_single_step.py --save ./model/model-exchange-24.pt --data ./data/exchange_rate.txt --num_nodes 8 --horizon 3 --batch_size 32 --residual_channels 32 --timefusion_true 1 --attention_true DAU --gcn_true 1 --epochs 100

#sampling
python train_single_step.py --num_split 3 --save ./model-exchange-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --subgraph_size 2  --batch_size 32 --epochs 100 --horizon 3
```
