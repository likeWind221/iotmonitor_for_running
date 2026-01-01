"""
实时数据模拟器 - MQTT上传版
用法：python simulator_mqtt.py
"""
import time
import random
import numpy as np
import paho.mqtt.client as mqtt
import json
from datetime import datetime

# ========== 配置 ==========
MQTT_BROKER = "127.0.0.1"  # 如需远程测试请改为你的服务器IP
MQTT_PORT = 1883
MPU_TOPIC = "sensor/mpu"
DATA_TOPIC = "sensor/data"

# ========== 运动模式参数 ==========
MODE_PARAMS = {
    0: {
        'name': '走路',
        'hr_range': (70, 90),
        'spo2_range': (96, 99),
        'speed': 1.0,  # m/s
        'gyro_freq': 0.5,
        'x_range': (-18.2, -9.9),
        'y_range': (152.63, 166.91)
    },
    1: {
        'name': '慢跑',
        'hr_range': (100, 130),
        'spo2_range': (94, 97),
        'speed': 2.5,
        'gyro_freq': 1.5,
        'x_range': (-25, -15),
        'y_range': (145, 160)
    },
    2: {
        'name': '冲刺',
        'hr_range': (150, 190),
        'spo2_range': (90, 95),
        'speed': 5.0,
        'gyro_freq': 3.5,
        'x_range': (-35, -25),
        'y_range': (130, 145)
    }
}

# ========== 运动模式选择 ==========
print("选择运动模式：\n 1. 走路 2. 慢跑 3. 冲刺 4. 混合模式")
mode_choice = input("输入选择（1/2/3/4）：").strip()
mode_map = {'1': 0, '2': 1, '3': 2, '4': None}
fixed_mode = mode_map.get(mode_choice, None)

print("\n输入模拟时长（秒，默认120）：", end="")
duration = input().strip()
duration = int(duration) if duration.isdigit() else 120

# ========== MQTT连接 ==========
client = mqtt.Client(client_id="esp32_simulator")
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# ========== 经纬度基准（可自定义） ========== - 移除，不再使用GPS

# ========== 主循环 ==========
start_time = time.time()
total_distance = 0.0
current_mode = fixed_mode if fixed_mode is not None else 0
next_switch_time = None
transition_progress = 1.0
transition_speed = 0.05

# 新增：计数器和角度缓存
mpu_count = 0
data_count = 0
last_angleX = 0.0
last_angleY = 0.0

def lerp(a, b, t):
    return a + (b - a) * t

def get_blended_params(current_mode, target_mode, t):
    if t >= 1.0:
        return MODE_PARAMS[target_mode]
    src = MODE_PARAMS[current_mode]
    dst = MODE_PARAMS[target_mode]
    return {
        'name': dst['name'],
        'hr_range': (
            int(lerp(src['hr_range'][0], dst['hr_range'][0], t)),
            int(lerp(src['hr_range'][1], dst['hr_range'][1], t))
        ),
        'spo2_range': (
            int(lerp(src['spo2_range'][0], dst['spo2_range'][0], t)),
            int(lerp(src['spo2_range'][1], dst['spo2_range'][1], t))
        ),
        'speed': lerp(src['speed'], dst['speed'], t),
        'gyro_freq': lerp(src['gyro_freq'], dst['gyro_freq'], t),
        'x_range': (
            lerp(src['x_range'][0], dst['x_range'][0], t),
            lerp(src['x_range'][1], dst['x_range'][1], t)
        ),
        'y_range': (
            lerp(src['y_range'][0], dst['y_range'][0], t),
            lerp(src['y_range'][1], dst['y_range'][1], t)
        )
    }

target_mode = current_mode

print("\n🚀 开始实时模拟上传... 按Ctrl+C停止\n")
try:
    mpu_counter = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > duration:
            print("\n✅ 模拟结束！")
            break
        # 模式切换逻辑
        if fixed_mode is not None:
            current_mode = fixed_mode
            target_mode = fixed_mode
            transition_progress = 1.0
        else:
            if next_switch_time is None:
                next_switch_time = elapsed + random.uniform(10, 30)
            if elapsed >= next_switch_time and transition_progress >= 1.0:
                available = [0, 1, 2]
                available.remove(target_mode)
                current_mode = target_mode
                target_mode = random.choice(available)
                transition_progress = 0.0
                next_switch_time = elapsed + random.uniform(10, 30)
                print(f"  [切换] {elapsed:.1f}s - {MODE_PARAMS[current_mode]['name']} → {MODE_PARAMS[target_mode]['name']}")
            if transition_progress < 1.0:
                transition_progress = min(1.0, transition_progress + transition_speed)
        params = get_blended_params(current_mode, target_mode, transition_progress)
        # 陀螺仪数据（每0.1秒）
        if mpu_counter % 1 == 0:
            angleX = (np.sin(2 * np.pi * params['gyro_freq'] * elapsed) *
                      (params['x_range'][1] - params['x_range'][0])/4) + \
                      (params['x_range'][0] + params['x_range'][1])/2 + random.uniform(-0.5, 0.5)
            angleY = (np.cos(2 * np.pi * params['gyro_freq'] * elapsed) *
                      (params['y_range'][1] - params['y_range'][0])/4) + \
                      (params['y_range'][0] + params['y_range'][1])/2 + random.uniform(-0.8, 0.8)
            angleX = max(params['x_range'][0], min(angleX, params['x_range'][1]))
            angleY = max(params['y_range'][0], min(angleY, params['y_range'][1]))
            last_angleX = angleX  # 缓存角度
            last_angleY = angleY
            mpu_payload = {
                "angleX": round(angleX, 2),
                "angleY": round(angleY, 2),
                "time": datetime.now().strftime("%H:%M:%S"),
                "count": mpu_count,
                "timestamp_ms": int(time.time() * 1000)
            }
            client.publish(MPU_TOPIC, json.dumps(mpu_payload), qos=0)
            mpu_count += 1
        # 生理数据（每1秒）
        if mpu_counter % 10 == 0:
            hr = random.randint(params['hr_range'][0], params['hr_range'][1])
            spo2 = random.randint(params['spo2_range'][0], params['spo2_range'][1])
            # 模拟GPS（小范围抖动） - 移除
            # lat = BASE_LAT + random.uniform(-0.0005, 0.0005)
            # lng = BASE_LNG + random.uniform(-0.0005, 0.0005)
            total_distance = total_distance + params['speed']
            step_count = int(total_distance / 0.75)  # 模拟步数（步长0.75m）
            sensor_payload = {
                "hr": hr,
                "spo2": spo2,
                "angleX": round(last_angleX, 2),
                "angleY": round(last_angleY, 2),
                "total_mileage_m": round(total_distance, 2),
                "step_count": step_count,
                "time": datetime.now().strftime("%H:%M:%S"),
                "count": data_count
            }
            client.publish(DATA_TOPIC, json.dumps(sensor_payload), qos=0)
            data_count += 1
        mpu_counter += 1
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n⏹️ 已手动停止模拟上传。")
finally:
    client.disconnect()
