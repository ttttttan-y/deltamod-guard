# DeltaMod-Guard — 基于增量调制（DM/ADPCM-lite）的时序异常检测系统（MVP）

工程化目标：将连续时间序列在线编码为 1-bit/低比特符号流，
用过载率/翻转率/游程统计等位面特征做流式异常检测，生成“码率–检出率–延迟”报告。

## 快速开始
```bash
cd deltamod-guard
pip install -r requirements.txt
python run_demo.py
# 报告：data/reports/dmguard_demo.html
```
（可选）启动最小 FastAPI 服务（占位）：
```bash
uvicorn services.dm_api.app:app --host 0.0.0.0 --port 8010
# 健康检查 http://localhost:8010/health
```
