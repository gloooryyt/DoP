# DoP
***
### Raw Data

Raw data can be found at https://github.com/GestaltCogTeam/BasicTS/tree/master/datasets, and unzip them to ```datasets/raw_data```.

### Requirements.txt
```
numpy 1.22.3
torch 1.13.1
easy-torch 1.3.2
scipy 1.10.1
```


### Data Processing
Run the following codes to conduct data processing, (replace X with corresponding number, and set history_seq_len as sequence length)

For examples,
```
python scripts/data_preparation/PEMS0X/generate_training_data.py --history_seq_len 12
python scripts/data_preparation/PEMS0X/generate_training_data.py --history_seq_len 2016
```


### Run pre-training
Run the following code to conduct pre-training,
```
python run.py --cfg 'dop/pretrain_PEMS0X.py' --gpus=0
```
You can set up multiple gpus for parallel computing. 

### Run Forecasting
After pre-training, set the pre-trained checkpoint path in forecast_PEMS0X.py ('pre_trained_path').
Run the following code to conduct forecasting training processing,
```
python run.py --cfg 'dop/forecast_PEMS0X.py' --gpus=0
```
You can also set up multiple gpus for parallel computing.



