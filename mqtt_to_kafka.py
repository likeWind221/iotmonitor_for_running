"""
MQTT -> Kafka 桥接脚本
用法示例：
  python scripts/mqtt_to_kafka.py --mqtt-broker localhost --mqtt-port 1883 --mqtt-topics sensor/data,sensor/mpu --kafka-bootstrap localhost:9092 --kafka-topic sensor_data

说明：订阅指定的 MQTT 主题，尽量把 payload 解析为 JSON（失败时用 raw 字符串包装），并将消息发布到 Kafka topic（默认 sensor_data）。
"""

import argparse
import json
import logging
import time
import signal
import sys
import sqlite3
import threading
import queue

from kafka import KafkaProducer
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger('mqtt_to_kafka')

stop = False

def signal_handler(sig, frame):
    global stop
    log.info('收到停止信号')
    stop = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def make_producer(bootstrap_servers, acks=1):
    # kafka-python expects `acks` to be an int (e.g., 0, 1) or 'all'.
    # Guard against callers passing a numeric string which would cause struct packing errors.
    if isinstance(acks, str):
        a_s = acks.strip().lower()
        if a_s == 'all':
            acks = 'all'
        else:
            try:
                acks = int(a_s)
            except Exception:
                # fallback to 1
                acks = 1
    return KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                         acks=acks)


class MQTTToKafkaBridge:
    def __init__(self, mqtt_broker, mqtt_port, mqtt_topics, kafka_bootstrap, kafka_topic, qos=0, sync_send=False, sqlite_db=None, sqlite_db_mpu=None, sqlite_db_data=None, sqlite_batch_size=100, sqlite_batch_interval=1.0):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topics = mqtt_topics
        self.kafka_bootstrap = kafka_bootstrap
        self.kafka_topic = kafka_topic
        self.qos = qos
        self.sync_send = sync_send
        # add small linger and retries for better batching and reliability
        self.producer = make_producer(kafka_bootstrap)
        self.client = mqtt.Client(client_id='mqtt_to_kafka_bridge')
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # sqlite related (raw optional, plus dedicated MPU / DATA DBs)
        self.sqlite_db = sqlite_db
        self.sqlite_db_mpu = sqlite_db_mpu
        self.sqlite_db_data = sqlite_db_data
        self.sqlite_queue = None
        self.sqlite_thread = None
        self.sqlite_queue_mpu = None
        self.sqlite_thread_mpu = None
        self.sqlite_queue_data = None
        self.sqlite_thread_data = None
        self.sqlite_batch_size = sqlite_batch_size
        self.sqlite_batch_interval = sqlite_batch_interval

        if self.sqlite_db:
            self.sqlite_queue = queue.Queue(maxsize=10000)
            self.sqlite_thread = threading.Thread(target=self._sqlite_worker_raw, daemon=True)
            self.sqlite_thread.start()

    def _sqlite_worker_raw(self):
        """Background thread: batch insert messages into sqlite (raw messages table)."""
        db_path = self.sqlite_db
        try:
            conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
            cur = conn.cursor()
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.execute('PRAGMA busy_timeout=5000')
            cur.execute('''CREATE TABLE IF NOT EXISTS raw_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                payload TEXT,
                timestamp_ms INTEGER,
                recv_ts INTEGER
            )''')
            conn.commit()
        except Exception as e:
            log.exception('SQLite(raw) 初始化失败: %s', e)
            return

        buf = []
        last_flush = time.time()
        while not stop or not (self.sqlite_queue.empty()):
            try:
                item = self.sqlite_queue.get(timeout=0.5)
                if item is None:
                    break
                buf.append(item)
            except queue.Empty:
                item = None

            now = time.time()
            if len(buf) >= self.sqlite_batch_size or (buf and (now - last_flush) >= self.sqlite_batch_interval):
                try:
                    cur.executemany('INSERT INTO raw_messages(topic,payload,timestamp_ms,recv_ts) VALUES(?,?,?,?)', buf)
                    conn.commit()
                    log.info('已写入 %d 条到 SQLite(raw)', len(buf))
                except Exception as e:
                    log.exception('写入 SQLite(raw) 失败: %s', e)
                buf = []
                last_flush = now

        # flush remaining
        if buf:
            try:
                cur.executemany('INSERT INTO raw_messages(topic,payload,timestamp_ms,recv_ts) VALUES(?,?,?,?)', buf)
                conn.commit()
                log.info('退出前写入 %d 条到 SQLite(raw)', len(buf))
            except Exception as e:
                log.exception('退出时写入 SQLite(raw) 失败: %s', e)
        try:
            conn.close()
        except Exception:
            pass

    def _sqlite_worker_mpu(self, db_path):
        """Background thread: batch insert MPU messages into sqlite table 'mpu' (angleX, angleY, ts)."""
        try:
            conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
            cur = conn.cursor()
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.execute('PRAGMA busy_timeout=5000')
            cur.execute('''CREATE TABLE IF NOT EXISTS mpu (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                angleX REAL,
                angleY REAL,
                ts INTEGER
            )''')
            conn.commit()
        except Exception as e:
            log.exception('SQLite(mpu) 初始化失败: %s', e)
            return

        buf = []
        last_flush = time.time()
        while not stop or not (self.sqlite_queue_mpu.empty()):
            try:
                item = self.sqlite_queue_mpu.get(timeout=0.5)
                if item is None:
                    break
                buf.append(item)
            except queue.Empty:
                item = None

            now = time.time()
            if len(buf) >= self.sqlite_batch_size or (buf and (now - last_flush) >= self.sqlite_batch_interval):
                try:
                    cur.executemany('INSERT INTO mpu(angleX,angleY,ts) VALUES(?,?,?)', buf)
                    conn.commit()
                    log.info('已写入 %d 条到 SQLite(mpu)', len(buf))
                except Exception as e:
                    log.exception('写入 SQLite(mpu) 失败: %s', e)
                buf = []
                last_flush = now

        if buf:
            try:
                cur.executemany('INSERT INTO mpu(angleX,angleY,ts) VALUES(?,?,?)', buf)
                conn.commit()
                log.info('退出前写入 %d 条到 SQLite(mpu)', len(buf))
            except Exception as e:
                log.exception('退出时写入 SQLite(mpu) 失败: %s', e)
        try:
            conn.close()
        except Exception:
            pass

    def _sqlite_worker_data(self, db_path):
        """Background thread: batch insert DATA messages into sqlite table 'data' (hr,spo2,mileage,ts)."""
        try:
            conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
            cur = conn.cursor()
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.execute('PRAGMA busy_timeout=5000')
            cur.execute('''CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hr INTEGER,
                spo2 INTEGER,
                mileage REAL,
                ts INTEGER
            )''')
            conn.commit()
        except Exception as e:
            log.exception('SQLite(data) 初始化失败: %s', e)
            return

        buf = []
        last_flush = time.time()
        while not stop or not (self.sqlite_queue_data.empty()):
            try:
                item = self.sqlite_queue_data.get(timeout=0.5)
                if item is None:
                    break
                buf.append(item)
            except queue.Empty:
                item = None

            now = time.time()
            if len(buf) >= self.sqlite_batch_size or (buf and (now - last_flush) >= self.sqlite_batch_interval):
                try:
                    cur.executemany('INSERT INTO data(hr,spo2,mileage,ts) VALUES(?,?,?,?)', buf)
                    conn.commit()
                    log.info('已写入 %d 条到 SQLite(data)', len(buf))
                except Exception as e:
                    log.exception('写入 SQLite(data) 失败: %s', e)
                buf = []
                last_flush = now

        if buf:
            try:
                cur.executemany('INSERT INTO data(hr,spo2,mileage,ts) VALUES(?,?,?,?)', buf)
                conn.commit()
                log.info('退出前写入 %d 条到 SQLite(data)', len(buf))
            except Exception as e:
                log.exception('退出时写入 SQLite(data) 失败: %s', e)
        try:
            conn.close()
        except Exception:
            pass

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info('已连接到 MQTT Broker %s:%s', self.mqtt_broker, self.mqtt_port)
            for t in self.mqtt_topics:
                client.subscribe(t, qos=self.qos)
                log.info('已订阅 MQTT 主题: %s', t)
        else:
            log.error('连接 MQTT 失败，rc=%s', rc)

    def on_message(self, client, userdata, msg):
        # 诊断日志：显示接收到的主题与简短负载
        try:
            raw = msg.payload.decode('utf-8')
        except Exception:
            raw = msg.payload
        snippet = (raw[:200] + '...') if isinstance(raw, str) and len(raw) > 200 else raw
        log.info('收到 MQTT 消息 topic=%s payload=%s', msg.topic, snippet)

        payload = None
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(payload, dict):
                if 'timestamp_ms' not in payload:
                    payload['timestamp_ms'] = int(time.time() * 1000)
            else:
                payload = {'value': payload, 'timestamp_ms': int(time.time() * 1000)}
        except Exception:
            payload = {'raw': raw, 'timestamp_ms': int(time.time() * 1000)}

        # 发送到 Kafka，并添加回调以便诊断
        try:
            fut = self.producer.send(self.kafka_topic, payload)
            if self.sync_send:
                try:
                    meta = fut.get(timeout=5)
                    log.info('同步发送到 Kafka 成功: topic=%s partition=%s offset=%s', meta.topic, meta.partition, meta.offset)
                except Exception as e:
                    log.exception('同步发送失败: %s', e)
            else:
                fut.add_callback(lambda meta: log.info('异步发送成功: topic=%s partition=%s offset=%s', meta.topic, meta.partition, meta.offset))
                fut.add_errback(lambda exc: log.exception('异步发送失败: %s', exc))
        except Exception as e:
            log.exception('发送到 Kafka 失败: %s', e)

        # 可选：将原始消息写入 SQLite(raw)（兼容历史选项）
        if self.sqlite_queue is not None:
            try:
                payload_text = json.dumps(payload, ensure_ascii=False)
            except Exception:
                payload_text = str(payload)
            ts = payload.get('timestamp_ms') if isinstance(payload, dict) else None
            try:
                self.sqlite_queue.put_nowait((msg.topic, payload_text, ts, int(time.time()*1000)))
            except queue.Full:
                log.warning('SQLite(raw) 队列已满，已丢弃消息')

        # 如果配置了专门的 MPU / DATA sqlite，则分别解包并入表
        if self.sqlite_db_mpu and msg.topic == 'sensor/mpu':
            try:
                aX = None
                aY = None
                ts_m = None
                if isinstance(payload, dict):
                    aX = payload.get('angleX')
                    aY = payload.get('angleY')
                    ts_m = payload.get('timestamp_ms') or int(time.time()*1000)
                else:
                    ts_m = int(time.time()*1000)
                try:
                    self.sqlite_queue_mpu.put_nowait((aX, aY, ts_m))
                except queue.Full:
                    log.warning('SQLite(mpu) 队列已满，已丢弃 MPU 消息')
            except Exception:
                log.exception('处理 MPU 写入队列时出错')

        if self.sqlite_db_data and msg.topic == 'sensor/data':
            try:
                hr = None
                spo2 = None
                mileage = None
                ts_d = None
                if isinstance(payload, dict):
                    hr = payload.get('hr')
                    spo2 = payload.get('spo2')
                    mileage = payload.get('total_mileage_m')
                    ts_d = payload.get('timestamp_ms') or int(time.time()*1000)
                else:
                    ts_d = int(time.time()*1000)
                try:
                    self.sqlite_queue_data.put_nowait((hr, spo2, mileage, ts_d))
                except queue.Full:
                    log.warning('SQLite(data) 队列已满，已丢弃 DATA 消息')
            except Exception:
                log.exception('处理 DATA 写入队列时出错')

    def run(self):
        try:
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
            log.info('桥接运行中，MQTT -> Kafka (%s -> %s)', ','.join(self.mqtt_topics), self.kafka_topic)
            while not stop:
                time.sleep(0.2)
        finally:
            log.info('正在关闭桥接...')
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            try:
                self.producer.flush(timeout=5)
            except Exception:
                pass

            # shutdown sqlite threads and queues (raw / mpu / data)
            try:
                if self.sqlite_queue is not None:
                    self.sqlite_queue.put(None, timeout=1)
            except Exception:
                pass
            try:
                if self.sqlite_queue_mpu is not None:
                    self.sqlite_queue_mpu.put(None, timeout=1)
            except Exception:
                pass
            try:
                if self.sqlite_queue_data is not None:
                    self.sqlite_queue_data.put(None, timeout=1)
            except Exception:
                pass

            # join threads
            try:
                if self.sqlite_thread is not None:
                    self.sqlite_thread.join(timeout=5)
            except Exception:
                pass
            try:
                if self.sqlite_thread_mpu is not None:
                    self.sqlite_thread_mpu.join(timeout=5)
            except Exception:
                pass
            try:
                if self.sqlite_thread_data is not None:
                    self.sqlite_thread_data.join(timeout=5)
            except Exception:
                pass

            log.info('已退出')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mqtt-broker', default='127.0.0.1', help='MQTT broker 地址')
    parser.add_argument('--mqtt-port', type=int, default=1883, help='MQTT 端口')
    parser.add_argument('--mqtt-topics', default='sensor/data', help='要订阅的 MQTT 主题, 逗号分隔')
    parser.add_argument('--kafka-bootstrap', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--kafka-topic', default='sensor_data', help='目标 Kafka topic')
    parser.add_argument('--qos', type=int, default=0, choices=[0,1,2], help='MQTT 订阅 QoS')
    parser.add_argument('--sync-send', action='store_true', help='在测试时使用同步发送并等待确认（会阻塞）')
    parser.add_argument('--verbose', action='store_true', help='启用调试日志（等同于 --log-level debug）')
    parser.add_argument('--log-level', choices=['debug','info','warning','error'], default='info', help='控制输出日志级别')
    parser.add_argument('--sqlite-db', default='', help='若设置则把原始消息写入指定 sqlite db 文件')
    parser.add_argument('--sqlite-db-mpu', default='', help='若设置则将 MPU(topic sensor/mpu) 消息写入该 sqlite db 文件 (table: mpu)')
    parser.add_argument('--sqlite-db-data', default='', help='若设置则将 DATA(topic sensor/data) 消息写入该 sqlite db 文件 (table: data)')
    parser.add_argument('--sqlite-batch-size', type=int, default=100, help='SQLite 批量写入大小')
    parser.add_argument('--sqlite-batch-interval', type=float, default=1.0, help='SQLite 批量写入间隔（秒）')
    args = parser.parse_args()

    # set log level
    lvl = args.log_level.upper()
    if args.verbose:
        lvl = 'DEBUG'
    log.setLevel(getattr(logging, lvl))

    sqlite_db = args.sqlite_db if args.sqlite_db else None
    sqlite_db_mpu = args.sqlite_db_mpu if args.sqlite_db_mpu else None
    sqlite_db_data = args.sqlite_db_data if args.sqlite_db_data else None

    # initialize bridge with separate sqlite dbs if provided
    bridge = MQTTToKafkaBridge(
        args.mqtt_broker,
        args.mqtt_port,
        [t.strip() for t in args.mqtt_topics.split(',')],
        args.kafka_bootstrap,
        args.kafka_topic,
        qos=args.qos,
        sync_send=args.sync_send,
        sqlite_db=sqlite_db,
        sqlite_batch_size=args.sqlite_batch_size,
        sqlite_batch_interval=args.sqlite_batch_interval,
        # dedicated dbs (optional)
        sqlite_db_mpu=sqlite_db_mpu,
        sqlite_db_data=sqlite_db_data,
    )
    # if dedicated dbs are provided, start their threads
    if sqlite_db_mpu:
        bridge.sqlite_queue_mpu = queue.Queue(maxsize=10000)
        bridge.sqlite_thread_mpu = threading.Thread(target=bridge._sqlite_worker_mpu, args=(sqlite_db_mpu,), daemon=True)
        bridge.sqlite_thread_mpu.start()
    if sqlite_db_data:
        bridge.sqlite_queue_data = queue.Queue(maxsize=10000)
        bridge.sqlite_thread_data = threading.Thread(target=bridge._sqlite_worker_data, args=(sqlite_db_data,), daemon=True)
        bridge.sqlite_thread_data.start()

    bridge.run()
