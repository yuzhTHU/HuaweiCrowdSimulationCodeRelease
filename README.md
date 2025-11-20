# 训练模型

```shell
python train.py --name train --datasets All --test_ratio 0.2 --p_drop_map 0.2 --p_drop_destination 0.3 --p_drop_speed 0.3"
```
运行日志、模型参数和可视化结果将被保存在 `logs/train/xxx_train_xxx/` 中

## 恢复训练
训练到到一半退出、需要继承之前的进度继续训练时，可通过指定 `--exp_name xxx_train_xxx` 或者 `--reload_checkpoint ./logs/train/*_train_*/checkpoint.pth` 从上次运行的结果继续运行。指定 `--exp_name` 将导致运行结果继续输出到 `logs/train/xxx_train_xxx` 中；指定 `--reload_checkpoint` 将导致运行结果输出到新的 `logs/train/yyy_train_yyy` 路径中


# 测试模型

```shell
python sample.py --name sample --reload_checkpoint /path/to/logs/train/xxx_train_xxx/best.pth --roll_step 500 --sample_num 10
```
运行日志和可视化结果将被保存在 `logs/sample/xxx_sample_xxx` 中

# 网页可视化

```shell
uvicorn app:app --host 0.0.0.0 --port 12345
```
将在服务器本地的 12345 端口开启网页可视化应用。可通过浏览器打开以：加载模型、加载数据、可视化数据轨迹、运行模拟、可视化模拟结果、管理模拟结果。