GPU集群多机多卡分布式SNN训练环境构建
====================================

本教程作者： `morainer <https://github.com/morainer>`_ 、 `zxk907 <https://github.com/zxk907>`_ 

在云脑GPU服务器上部署基础docker镜像( :class:`Pytorch container` )，并进入容器环境内。

.. code:: shell

   docker pull nvcr.io/nvidia/pytorch:19.10-py3


ssh服务安装、配置与启动。

 :class:`ssh` 服务安装：

.. code:: shell

   sudo apt-get update && apt-get install openssh-server


 :class:`ssh` 服务配置，修改 :class:`sshd_config` 、 :class:`ssh_config` 文件：

.. code:: shell

   sudo vi nano /etc/ssh/sshd_config
   sudo vi nano /etc/ssh/ssh_config


在 :class:`sshd_config` 文件设置 :class:`Port` 为222或其他任意端口号， :class:`PermitRootLogin` 为yes。

在 :class:`ssh_config` 文件设置 :class:`Port` 为222或其他任意端口号。

 :class:`ssh` 服务启动与状态查看

.. code:: shell

   sudo service ssh start
   sudo service ssh status


设置容器环境内root密码。

.. code:: shell

   passwd root


添加环境变量。

.. code:: shell

   sudo vi /etc/profile

在文件末尾追加： :class:`export $(cat /proc/1/environ |tr '\0' '\n' | xargs)` ，随后使之立即生效：
.. code:: shell

   sudo source /etc/profile


设置节点间免密。

假设现有节点 :class:`node1、node2、...` ：

生成免密密钥对：

.. code:: shell

   sudo ssh-keygen


将上述免密密钥对的公钥分发至所有节点：

.. code:: shell

   sudo ssh-copy-id -i .ssh/id_rsa.pub  root@node x ip


软件安装。

 :class:`pytorch: 19.10-py3` 已存在环境：

 :class:`Ubuntu 18.04` 、 :class:`Python 3.6.9` 、 :class:`MLNX OFED` 、 :class:`OpenMPI 3.1.4` 、 :class:`APEX 0.1` 、 :class:`ONNX1.5.0` 、

 :class:`NVIDIA CUDA 10.1.243 (including cuBLAS 10.2.1.243)` 、 :class:`NVIDIA cuDNN 7.6.4` 、

 :class:`NVIDIA NCCL 2.4.8(optimized for NVLink)` 、

另需安装：

 :class:`Pytorch 1.7.1` 安装：

.. code:: shell

   sudo pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2


 :class:`mpi4py` 安装：

.. code:: shell

   sudo pip install mpi4py

   sudo apt-get install python3-mpi4py


 :class:`numpy` 、 :class:`scipy` 安装：

.. code:: shell

   sudo pip install numpy==1.17.2
   sudo pip install scipy==1.3.0

 :class:`pdsh` 安装：

.. code:: shell

   sudo apt-get update
   sudo apt-get install pdsh
   sudo pdsh -V

 :class:`spikingjelly` 安装

.. code:: shell

   git clone https://git.openi.org.cn/OpenI/spikingjelly.git
   cd spikingjelly
   git checkout 0.0.0.0.4
   python setup.py install

 :class:`DeepSpeed` 安装

.. code:: shell

   pip install Deepspeed


构建分布式加速训练代码

本加速方案主要是建立在微软团队提出的 `Deepspeed <https://github.com/microsoft/DeepSpeed>`_ 基础之上，对其中的一些API进行了针对性地封装，具体代码构建如下：

使用 :class:`SpikingSail.py` 中 :class:`intialize()` 初始化 :class:`model` 、 :class:`optimizer` 

.. code:: python

   parser = dpspd_new.add_config_arguments(parser)
   args = parser.parse_args()
   model_engine, optimizer = dpspd_new.useDeepspeed(args, model, optimizer).run()

训练时前向传播、反向传播和权重更新使用以下代码：

.. code:: python

   for step, batch in enumerate(data_loader):
       #forward() method
       loss = model_engine(batch)

       #runs backpropagation
       model_engine.backward(loss)

       #weight update
       model_engine.step()

配置JSON文件以启用或禁用DeepSpeed的相关功能：

.. code:: JSON

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

将上述JSON文件命名为 :class:`args._config` .

主机文件 :class:`hostfile` 配置：

.. code:: shell

   worker1  slots
   worker2  slots
     ..       ..
     

 :class:`worker` 为主机IP， :class:`slots` 为主机GPU数量

启动训练：

.. code:: shell

   rm -rf /root/.cache/torch_extensions/*
   deepspeed -H hostfile --num_nodes=xx --num_gpus=xx \
           <client_entry.py> <client args> \
           --deepspeed --deepspeed_config ds_config.json


GPU集群单机多卡分布式SNN训练环境构建
====================================

当在单个节点（一个或者多个GPU）上进行模型训练，则无需配置以上的主机文件
 :class:`hostfile` ，系统会查询本地GPU数量，用户使用 :class:`localhost` 指定主机名，可利用 :class:`--include--exclude` 来指定使用的设备。例如使用GPU1：

.. code:: shell

   deepspeed --include localhost:1 ...


注意：不能使用 :class:`CUDA_VISIBLE_DEVICES` 来指定使用的设备