#!/usr/bin/env python3
# MQTT 调试脚本：订阅跑步模式 + 发布配速建议
import paho.mqtt.client as mqtt
import json
import time
import threading

# -------------------------- 配置项（修改为你的MQTT服务器信息） --------------------------
MQTT_BROKER = "192.168.186.210"  # MQTT服务器IP
MQTT_PORT = 1883                 # MQTT端口（默认1883）
MQTT_CLIENT_ID = "MQTT_Debug_Tool"
RUN_MODE_TOPIC = "running/mode"          # 订阅跑步模式的主题
RECOMMEND_SPEED_TOPIC = "running/recommend_speed"  # 发布配速建议的主题
MQTT_USER = ""                   # 若服务器需要认证，填写用户名（无则留空）
MQTT_PASS = ""                   # 若服务器需要认证，填写密码（无则留空）
# ---------------------------------------------------------------------------------------

class MQTTDebugTool:
    def __init__(self):
        # 初始化MQTT客户端（适配2.0+版本）
        self.client = mqtt.Client(
            client_id=MQTT_CLIENT_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1
        )
        # 设置认证（若有）
        if MQTT_USER and MQTT_PASS:
            self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        # 绑定回调函数
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        # 连接状态标记
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        """连接成功回调"""
        if rc == 0:
            self.connected = True
            print(f"\n✅ MQTT连接成功！服务器: {MQTT_BROKER}:{MQTT_PORT}")
            # 订阅跑步模式主题
            client.subscribe(RUN_MODE_TOPIC, qos=1)
            print(f"📌 已订阅主题: {RUN_MODE_TOPIC}（跑步模式）")
        else:
            self.connected = False
            print(f"\n❌ MQTT连接失败！错误码: {rc} ({mqtt.connack_string(rc)})")

    def on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        if rc != 0:
            print(f"\n⚠️ MQTT意外断开连接！错误码: {rc}")
            # 自动重连
            print("🔄 尝试重新连接...")
            self.reconnect()

    def on_message(self, client, userdata, msg):
        """接收消息回调（主要接收跑步模式）"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8').strip()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # 格式化显示跑步模式消息
            if topic == RUN_MODE_TOPIC:
                try:
                    # 解析JSON
                    mode_data = json.loads(payload)
                    print(f"\n📥 【跑步模式消息】[{timestamp}]")
                    print(f"   模式标识: {mode_data.get('mode', '未知')}")
                    print(f"   模式名称: {mode_data.get('mode_name', '未知')}")
                    print(f"   发布时间戳: {mode_data.get('timestamp', '未知')}")
                    print(f"   原始消息: {payload}")
                except:
                    # 非JSON格式（兼容）
                    print(f"\n📥 【跑步模式消息】[{timestamp}] 非JSON格式: {payload}")
        except Exception as e:
            print(f"\n❌ 解析消息失败: {e}")

    def connect(self):
        """连接MQTT服务器"""
        try:
            print(f"🔌 正在连接MQTT服务器: {MQTT_BROKER}:{MQTT_PORT}...")
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            # 启动后台循环（非阻塞）
            self.client.loop_start()
            # 等待连接完成
            time.sleep(1)
        except Exception as e:
            print(f"\n❌ 连接失败: {e}")
            self.connected = False

    def reconnect(self):
        """重连MQTT服务器"""
        try:
            self.client.reconnect()
        except:
            time.sleep(5)
            self.reconnect()

    def publish_speed(self, speed_value, format_type="numeric"):
        """
        发布配速建议消息
        :param speed_value: 配速值（如8.5，单位km/h）
        :param format_type: 消息格式 - "numeric"（纯数值） / "json"（JSON格式）
        """
        if not self.connected:
            print("\n❌ MQTT未连接，无法发布消息！")
            return
        
        try:
            # 构造消息
            if format_type == "json":
                payload = json.dumps({
                    "speed": float(speed_value),
                    "timestamp": time.time(),
                    "unit": "km/h"
                }, ensure_ascii=False)
            else:
                payload = str(speed_value)
            
            # 发布消息（QoS=1，确保送达）
            result = self.client.publish(
                RECOMMEND_SPEED_TOPIC,
                payload=payload,
                qos=1,
                retain=False
            )
            # 等待发布确认
            result.wait_for_publish()
            
            if result.is_published():
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"\n📤 【配速建议发布成功】[{timestamp}]")
                print(f"   配速值: {speed_value} km/h")
                print(f"   消息格式: {format_type}")
                print(f"   发布主题: {RECOMMEND_SPEED_TOPIC}")
                print(f"   消息内容: {payload}")
            else:
                print(f"\n❌ 配速建议发布失败！错误码: {result.rc}")
        except Exception as e:
            print(f"\n❌ 发布消息异常: {e}")

    def run(self):
        """启动调试工具"""
        # 第一步：连接MQTT服务器
        self.connect()
        
        # 第二步：循环等待用户输入
        print("\n=====================================")
        print("MQTT调试工具 - 操作说明")
        print("1. 自动订阅跑步模式（running/mode）")
        print("2. 输入配速值发布建议（支持两种格式）")
        print("   - 输入示例1: 8.5 （纯数值格式）")
        print("   - 输入示例2: json 8.5 （JSON格式）")
        print("3. 输入 'quit' 退出程序")
        print("=====================================\n")
        
        while True:
            try:
                user_input = input("请输入配速值（或quit退出）: ").strip()
                
                if user_input.lower() == "quit":
                    print("\n🔚 退出程序，关闭MQTT连接...")
                    self.client.loop_stop()
                    self.client.disconnect()
                    break
                
                # 解析用户输入
                parts = user_input.split()
                if len(parts) == 1:
                    # 纯数值格式
                    speed = float(parts[0])
                    self.publish_speed(speed, format_type="numeric")
                elif len(parts) == 2 and parts[0].lower() == "json":
                    # JSON格式
                    speed = float(parts[1])
                    self.publish_speed(speed, format_type="json")
                else:
                    print("\n⚠️ 输入格式错误！请参考：")
                    print("   - 纯数值: 8.5")
                    print("   - JSON格式: json 8.5")
            
            except ValueError:
                print("\n❌ 输入的配速值不是有效数字！")
            except KeyboardInterrupt:
                print("\n\n🔚 强制退出，关闭MQTT连接...")
                self.client.loop_stop()
                self.client.disconnect()
                break
            except Exception as e:
                print(f"\n❌ 操作异常: {e}")

if __name__ == "__main__":
    # 初始化并启动调试工具
    debug_tool = MQTTDebugTool()
    debug_tool.run()