#!/usr/bin/env python3
# sensor_data_collector.py - 双主题数据收集器
import json
import sqlite3
import time
import paho.mqtt.client as mqtt
from datetime import datetime

# ==================== 配置参数 ====================
# 数据库文件
MPU_DB_FILE = 'mpu_data.db'
SENSOR_DB_FILE = 'sensor_data.db'

# MQTT配置
MQTT_BROK = '127.0.0.1'
MQTT_PORT = 1883
MPU_TOPIC = 'sensor/mpu'      # MPU6050高频数据主题
DATA_TOPIC = 'sensor/data'    # 心率血氧GPS低频数据主题
CLIENT_ID = 'sensor_collector'

# ==================== 数据库初始化 ====================
def init_mpu_database():
    """初始化MPU6050数据库"""
    with sqlite3.connect(MPU_DB_FILE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS mpu_data (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                angleX REAL,
                angleY REAL,
                ts    INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # 创建索引提高查询性能
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_mpu_ts 
            ON mpu_data(ts);
        ''')

def init_sensor_database():
    """初始化传感器数据库"""
    with sqlite3.connect(SENSOR_DB_FILE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                hr    REAL,
                spo2  REAL,
                lat   REAL,
                lng   REAL,
                mileage REAL DEFAULT 0.0,
                ts    INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # 兼容旧表：如果没有mileage字段则添加
        try:
            conn.execute("ALTER TABLE sensor_data ADD COLUMN mileage REAL DEFAULT 0.0")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                print(f"⚠️  数据库字段检查警告: {e}")
        
        # 创建索引提高查询性能
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_sensor_ts 
            ON sensor_data(ts);
        ''')

def init_databases():
    """初始化所有数据库"""
    init_mpu_database()
    init_sensor_database()
    print("✅ 数据库初始化完成")

# ==================== MQTT回调函数 ====================
def on_mpu_message(client, userdata, msg):
    """MPU6050数据消息处理"""
    try:
        data = json.loads(msg.payload)
        current_ts = int(time.time() * 1000)
        
        # 提取MPU数据
        angleX = data.get('angleX')
        angleY = data.get('angleY')
        
        if angleX is not None and angleY is not None:
            # 存入MPU数据库
            with sqlite3.connect(MPU_DB_FILE) as conn:
                conn.execute('''
                    INSERT INTO mpu_data (angleX, angleY, ts)
                    VALUES (?, ?, ?)
                ''', (angleX, angleY, current_ts))
            
            # 控制台输出（原始格式）
            print(f"[MPU] 角度X: {angleX:.2f}°, 角度Y: {angleY:.2f}°, 时间: {datetime.now().strftime('%H:%M:%S')}")
            
    except json.JSONDecodeError as e:
        print(f'[MPU] JSON解析失败: {e}')
    except Exception as e:
        print(f'[MPU] 数据处理错误: {e}')

def on_data_message(client, userdata, msg):
    """传感器数据消息处理"""
    try:
        data = json.loads(msg.payload)
        current_ts = int(time.time() * 1000)
        
        # 提取传感器数据
        hr = data.get('hr')
        spo2 = data.get('spo2')
        lat = data.get('lat')
        lng = data.get('lng')
        mileage = data.get('total_mileage_m', 0.0)
        
        # 存入传感器数据库
        with sqlite3.connect(SENSOR_DB_FILE) as conn:
            conn.execute('''
                INSERT INTO sensor_data (hr, spo2, lat, lng, mileage, ts)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (hr, spo2, lat, lng, mileage, current_ts))
        
        # 控制台输出（原始格式，新增里程数）
        output_parts = []
        if hr is not None:
            output_parts.append(f"心率: {hr:.1f}")
        if spo2 is not None:
            output_parts.append(f"血氧: {spo2:.1f}")
        if lat is not None and lng is not None:
            output_parts.append(f"GPS: ({lat:.6f}, {lng:.6f})")
        if mileage > 0:
            output_parts.append(f"里程: {mileage:.2f} 米")
        
        if output_parts:
            output_str = ", ".join(output_parts)
            print(f"[SENSOR] {output_str}, 时间: {datetime.now().strftime('%H:%M:%S')}")
        
    except json.JSONDecodeError as e:
        print(f'[SENSOR] JSON解析失败: {e}')
    except Exception as e:
        print(f'[SENSOR] 数据处理错误: {e}')

def on_connect(client, userdata, flags, rc):
    """MQTT连接成功回调"""
    if rc == 0:
        print(f'✅ 已连接到MQTT服务器: {MQTT_BROK}:{MQTT_PORT}')
        
        # 订阅两个主题
        client.subscribe(MPU_TOPIC, qos=0)
        client.subscribe(DATA_TOPIC, qos=0)
        
        print(f'📡 已订阅主题: {MPU_TOPIC}')
        print(f'📡 已订阅主题: {DATA_TOPIC}')
    else:
        print(f'❌ 连接失败，返回码: {rc}')

def on_disconnect(client, userdata, rc):
    """MQTT断开连接回调"""
    print(f'⚠️  与MQTT服务器断开连接，返回码: {rc}')

# ==================== 主程序 ====================
def main():
    print("🚀 多传感器数据收集系统启动...")
    
    # 初始化数据库
    init_databases()
    
    # 创建MQTT客户端
    client = mqtt.Client(client_id=CLIENT_ID, protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
    try:
        print(f"🌐 连接到MQTT服务器: {MQTT_BROK}:{MQTT_PORT}")
        client.connect(MQTT_BROK, MQTT_PORT, 60)
        
        # 设置消息回调
        client.message_callback_add(MPU_TOPIC, on_mpu_message)
        client.message_callback_add(DATA_TOPIC, on_data_message)
        
        # 启动网络循环
        client.loop_start()
        
        print("\n🎉 数据收集系统已启动！")
        print("\n📡 正在接收数据...\n")
        
        # 无限循环保持程序运行（移除命令交互）
        while True:
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n👋 收到中断信号，正在退出...")
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
    finally:
        if client:
            client.loop_stop()
            client.disconnect()
        print("🛑 系统已停止")

if __name__ == '__main__':
    main()