# 复制代码
git clone git@github.com:yuzhTHU/HuaweiCrowdSimulationCode.git
cd HuaweiCrowdSimulationCode

# 创建环境
conda create -p ./venv python=3.12 -y
conda activate ./venv
pip install setproctitle tqdm "numpy<2.0" matplotlib pandas scipy torch ipykernel seaborn pyyaml optune
pip install fastapi "uvicorn[standard]" websockets

# 复制数据
mkdir ./data
rsync -anv --info=progress2 \
    --include='ETH/***' \
    --include='UCY/***' \
    --include='SDD/***' \
    --include='GC/***' \
    --include='WayMo/' \
    --include='WayMo/summary.csv' \
    --include='WayMo/Processed/***' \
    --include='.cache/***' \
    --exclude='*' \
    dl4:~/WorkSpace/35-HuaweiCrowdSimulation/HuaweiCrowdSimulationCode/data/ ./data

# 复制日志
mkdir ./logs
rsync -anv --info=progress2 \
    --exclude='***/.syncthing*' \
    dl4:~/WorkSpace/35-HuaweiCrowdSimulation/HuaweiCrowdSimulationCode/logs/ ./logs


# 如果使用 Ascend 昇腾 NPU
pip install torch_npu attrs cloudpickle ml-dtypes absl-py
export USE_NPU=True