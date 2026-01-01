# 快速启动指令

- 启动 Kafka & Zookeeper（后台）：
	- `docker compose -p iot_running up -d`

- 运行 MQTT→Kafka 桥接：
	- `python mqtt_to_kafka.py --mqtt-broker 127.0.0.1 --mqtt-port 1883 --mqtt-topics sensor/data,sensor/mpu --kafka-bootstrap 10.196.108.177:29092 --kafka-topic sensor_data`

- 运行订阅分析（Kafka 源）：
	- `python mqtt_subscriber.py --source kafka --kafka-bootstrap "10.196.108.177:29092" --kafka-topic sensor_data --target-mode walk`

- 运行订阅分析（MQTT 源直连设备）：
	- `python mqtt_subscriber.py --source mqtt --target-mode walk`

- 停止 Kafka & Zookeeper：
	- `docker compose -p iot_running down`

# 文件作用简介
- docker-compose.yml：一键启动 Zookeeper 与 Kafka（含对外监听设置）。
- mqtt_to_kafka.py：MQTT → Kafka 桥接，把设备 MQTT 主题转发到 Kafka topic。
- mqtt_subscriber.py：订阅 MQTT 或 Kafka，做窗口统计、模型推理与日志写入 logs/analysis.csv。
- mqtt_to_kafka.py 的 SQLite 选项：可把原始/分表数据落地调试。
- data/：示例/训练用数据集及切分。
- models/：行为克隆与序列模型权重等产物。
- scripts/：训练、评估、数据集拆分、推理等辅助脚本。

# 服务接口与网址
mqtt服务器网址：localhost:18083