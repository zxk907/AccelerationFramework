# GPU集群多机多卡分布式SNN训练环境构建

(1) 在云脑GPU服务器上部署基础docker镜像(`Pytorch container`)，并进入容器环境内。

```shell
docker pull nvcr.io/nvidia/pytorch:19.10-py3
```

(2)ssh服务安装、配置与启动。

`ssh`服务安装：

```shell
sudo apt-get update && apt-get install openssh-server
```

`ssh`服务配置：

修改`sshd_config`、`ssh_config`文件

```shell
sudo vi nano /etc/ssh/sshd_config
sudo vi nano /etc/ssh/ssh_config
```

在`sshd_config`文件设置`Port`为222或其他任意端口号，`PermitRootLogin`为yes。

在`ssh_config`文件设置`Port`为222或其他任意端口号。

`ssh`服务启动与状态查看

```shell
sudo service ssh start
sudo service ssh status
```

(3) 设置容器环境内root密码。

```shell
passwd root
```

(4) 添加环境变量。

```shell
sudo vi /etc/profile
在文件末尾追加：
export $(cat /proc/1/environ |tr '\0' '\n' | xargs)
随后使之立即生效：
sudo source /etc/profile
```

(5) 设置节点间免密。

假设现有节点`node1、node2、...`：

生成免密密钥对：

```shell
sudo ssh-keygen
```

将上述免密密钥对的公钥分发至所有节点：

```shell
sudo ssh-copy-id -i .ssh/id_rsa.pub  root@node x ip
```

(6) 软件安装。

`pytorch: 19.10-py3`已存在环境：

`Ubuntu 18.04`、`Python 3.6.9`、`MLNX OFED`、`OpenMPI 3.1.4`、`APEX 0.1`、`ONNX1.5.0`、

`NVIDIA CUDA 10.1.243 (including cuBLAS 10.2.1.243)`、`NVIDIA cuDNN 7.6.4`、

`NVIDIA NCCL 2.4.8(optimized for NVLink)`、

另需安装：

`Pytorch 1.7.1`安装：

```shell
sudo pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
```

`mpi4py`安装：

```shell
sudo pip install mpi4py

sudo apt-get install python3-mpi4py
```

`numpy`、`scipy`安装：

```shell
sudo pip install numpy==1.17.2
sudo pip install scipy==1.3.0
```

`pdsh`安装：

```shell
sudo apt-get update
sudo apt-get install pdsh
sudo pdsh -V
```

`spikingjelly`安装

```shell
git clone https://git.openi.org.cn/OpenI/spikingjelly.git
cd spikingjelly
git checkout 0.0.0.0.4
python setup.py install
```

`DeepSpeed`安装

```shell
pip install DeepSpeed
```

（7）构建分布式加速训练代码

本加速方案主要是建立在微软团队提出的[Deepspeed](https://github.com/microsoft/DeepSpeed)基础之上，对其中的一些API进行了针对性地封装，具体代码构建如下：

使用 `dpspd.py` 中 `intialize()` 初始化  `model`、`optimizer`

```
parser = dpspd_new.add_config_arguments(parser)
args = parser.parse_args()
model_engine, optimizer = dpspd.SpikingSail(args, model, optimizer).run()
```

训练时前向传播、反向传播和权重更新使用以下代码：

```
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
```

配置JSON文件：

```
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
}
```

将上述JSON文件命名为`args._config `. 

主机文件 `hostfile` 配置：

```
worker1  slots
worker2  slots
  ..       ..
```

`worker` 为主机IP，`slots` 为主机GPU数量

启动训练：

```
rm -rf /root/.cache/torch_extensions/*
deepspeed -H hostfile --num_nodes=xx --num_gpus=xx \
		<client_entry.py> <client args> \
		--deepspeed --deepspeed_config ds_config.json
```

