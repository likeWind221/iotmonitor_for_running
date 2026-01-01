BC Baseline scripts

1) 训练：

```
python scripts/train_bc.py --train data/splits/train.csv --val data/splits/val.csv --test data/splits/test.csv --out models/bc
```

输出：
- `models/bc/bc_model_final.pth`（模型权重）
- `models/bc/scaler.pkl`（标准化器）
- `models/bc/train_history.csv`（训练日志）
- `models/bc/test_metrics.json`（测试评估结果）

2) 评估：

```
python scripts/evaluate_bc.py --model models/bc/bc_model_final.pth --scaler models/bc/scaler.pkl --data data/splits/test.csv
```

依赖：torch, pandas, scikit-learn, numpy
