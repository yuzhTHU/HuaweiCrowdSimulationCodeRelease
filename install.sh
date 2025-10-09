# 复制代码
git clone git@github.com:yuzhTHU/HuaweiCrowdSimulationCode.git
cd HuaweiCrowdSimulationCode

# 创建环境
conda create -p ./venv python=3.12 -y
conda activate ./venv
pip install setproctitle tqdm numpy matplotlib pandas scipy torch ipykernel

# 复制数据
mkdir ./data
rsync -anv --progress \
    --include='ETH/***' \
    --include='UCY/***' \
    --include='SDD/***' \
    --include='GC/***' \
    --include='WayMo/Processed/***' \
    --include='.cache/***' \
    --exclude='*' \
    lm2:~/WorkSpace/35-HuaweiCrowdSimulation/HuaweiCrowdSimulationCode/data/ ./data

