# PUGAN-pytorch in TSUBAME
Pytorch unofficial implementation of PUGAN (a Point Cloud Upsampling Adversarial Network, ICCV, 2019)
#### Load Modules
```
module load python/3.6.5
module load cuda/9.0.176
module load nccl/2.2.13
module load cudnn/7.1
```
#### Install  pip
```
pip install --user  --upgrade pip
```

#### Install some packages
simply by 
```
pip install --user torch==1.2.0
pip install --user torchvision==0.4.0
pip install --user ninja
pip install --user colored-traceback
pip install -r requirements.txt --user
```
#### Install Pointnet2 module
```
cd pointnet2
python setup.py install --user
```
#### Install KNN_cuda
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl --user
```
#### dataset
We use the PU-Net dataset for training, you can refer to https://github.com/yulequan/PU-Net to download the .h5 dataset file, which can be directly used in this project.
#### modify some setting in the option/train_option.py
change opt['project_dir'] to where this project is located, and change opt['dataset_dir'] to where you store the dataset.
<br/>
also change params['train_split'] and params['test_split'] to where you save the train/test split txt files.
#### training
```
cd train
python train.py --exp_name=the_project_name --gpu=0 --use_gan --batch_size=12
```

