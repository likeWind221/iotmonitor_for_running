from kafka import KafkaProducer, KafkaConsumer
import json

p = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode())
p.send('sensor_data', {"time": 1700000000000, "hr": 75})
p.flush()

c = KafkaConsumer('sensor_data', bootstrap_servers='localhost:9092', auto_offset_reset='earliest', consumer_timeout_ms=2000)
for msg in c:
    print(msg.value)