#!/usr/bin/env python3
# 跑步监测工具 - 修复模式选择窗口超出屏幕问题
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import sqlite3
import json
import time
import threading
import os
import queue
import serial
import serial.tools.list_ports
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import paho.mqtt.client as mqtt

# 全局配置（彻底移除 Tkinter 字体全局配置）
DB_FILE = "running_data.db"
# 仅保留 matplotlib 字体配置（避免中文乱码）
plt.rcParams["font.family"] = ["DejaVu Sans"]  # 移除中文字体，避免 Tkinter 冲突
plt.rcParams["axes.unicode_minus"] = False

# MQTT配置
MQTT_SERVER = "192.168.186.210"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "RunningMonitor_UI"
RUN_MODE_TOPIC = "running/mode"
RECOMMEND_SPEED_TOPIC = "running/recommend_speed"

# 全局常量
UI_UPDATE_INTERVAL = 200
DATA_QUEUE_INTERVAL = 100
SERIAL_READ_INTERVAL = 0.1

class RunningMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Running Monitor")  # 英文标题，避免字体问题
        self.root.geometry("1050x700")
        self.root.resizable(False, False)
        
        # MQTT初始化
        self.mqtt_client = None
        self.current_run_mode = ""
        self.recommend_speed = 0.0
        
        # 基础变量
        self.is_running = False
        self.start_time = 0
        self.current_hr = 0
        self.current_spo2 = 0
        self.current_mileage = 0.0
        self.current_steps = 0
        self.current_speed = 0.0
        self.run_data = []
        self.bt_serial = None
        self.scanned_devices = {}
        
        # 线程安全
        self.lock = threading.Lock()
        self.data_queue = queue.Queue(maxsize=100)
        self.thread_exit_flag = threading.Event()
        self.read_thread = None
        self.sim_thread = None
        self.is_ui_alive = True
        
        # 临时日志缓存
        self.temp_log_cache = []
        
        # 先创建UI（不设置任何字体）
        self.create_ui()
        
        # 初始化数据库
        self.init_db()
        
        # 启动任务
        self.update_ui_task()
        self.consume_data_task()
        
        # 退出钩子
        self.root.protocol("WM_DELETE_WINDOW", self.safe_exit)
        
        # 显示模式选择窗口（修复居中逻辑）
        self.show_mode_select_window()
        
        # 初始化完成
        self.log("Program initialized, waiting for device connection...")

    def show_mode_select_window(self):
        """修复模式选择窗口 - 兼容不同屏幕分辨率，避免超出屏幕"""
        self.mode_window = Toplevel(self.root)
        self.mode_window.title("Select Run Mode")
        self.mode_window.geometry("400x250")
        self.mode_window.resizable(False, False)
        self.mode_window.transient(self.root)
        self.mode_window.grab_set()
        
        # ========== 核心修复：智能居中逻辑 ==========
        # 等待窗口渲染完成后获取尺寸（避免计算错误）
        self.mode_window.update_idletasks()
        
        # 获取屏幕尺寸
        screen_width = self.mode_window.winfo_screenwidth()
        screen_height = self.mode_window.winfo_screenheight()
        
        # 获取主窗口位置和尺寸
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        
        # 计算模式窗口居中位置（相对于主窗口，且不超出屏幕）
        win_width = self.mode_window.winfo_width()
        win_height = self.mode_window.winfo_height()
        
        # 先按主窗口居中计算
        x = root_x + (root_width - win_width) // 2
        y = root_y + (root_height - win_height) // 2
        
        # 修正：如果超出屏幕左边界，设为10px
        x = max(10, x)
        # 修正：如果超出屏幕右边界，设为屏幕宽度-窗口宽度-10px
        x = min(x, screen_width - win_width - 10)
        # 修正：如果超出屏幕上边界，设为10px
        y = max(10, y)
        # 修正：如果超出屏幕下边界，设为屏幕高度-窗口高度-10px
        y = min(y, screen_height - win_height - 10)
        
        # 设置最终位置
        self.mode_window.geometry(f"400x250+{x}+{y}")
        # ==========================================
        
        # 标题（无字体）
        title_label = tk.Label(self.mode_window, text="Select Run Mode")
        title_label.pack(pady=20)
        
        # 模式选择
        mode_frame = ttk.Frame(self.mode_window)
        mode_frame.pack(pady=10)
        
        self.mode_var = tk.StringVar(value="")
        modes = [
            ("Fat Burn", "fat_burn"),
            ("Endurance", "endurance"),
            ("Sprint", "sprint")
        ]
        
        for text, value in modes:
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value)
            rb.pack(anchor=tk.W, padx=50, pady=5)
        
        # 确认按钮（无字体）
        confirm_btn = ttk.Button(self.mode_window, text="Confirm", command=self.confirm_run_mode, width=15)
        confirm_btn.pack(pady=20)

    def confirm_run_mode(self):
        self.current_run_mode = self.mode_var.get()
        if not self.current_run_mode:
            messagebox.showwarning("Warning", "Please select a run mode!")
            return
        
        mode_names = {
            "fat_burn": "Fat Burn",
            "endurance": "Endurance",
            "sprint": "Sprint"
        }
        self.current_run_mode_name = mode_names.get(self.current_run_mode, self.current_run_mode)
        
        # 初始化MQTT
        self.init_mqtt()
        self.publish_run_mode()
        
        # 关闭窗口
        self.mode_window.destroy()
        self.temp_log_cache.append(f"Selected run mode: {self.current_run_mode_name}")

    def init_mqtt(self):
        try:
            # 适配 paho-mqtt 2.0+ 版本：指定回调 API 版本
            import paho.mqtt.client as mqtt
            self.mqtt_client = mqtt.Client(
                client_id=MQTT_CLIENT_ID,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION1  # 强制使用v1回调（兼容旧逻辑）
            )
            # 绑定回调函数（新版兼容）
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_message = self.on_mqtt_message
            self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
            
            def mqtt_connect_task():
                try:
                    # 增加连接超时、重连参数（新版优化）
                    self.mqtt_client.connect(MQTT_SERVER, MQTT_PORT, keepalive=60)
                    self.mqtt_client.loop_start()  # 启动后台循环
                    self.temp_log_cache.append(f"MQTT connected: {MQTT_SERVER}:{MQTT_PORT} (paho-mqtt 2.0+)")
                except Exception as e:
                    self.temp_log_cache.append(f"MQTT connect failed: {str(e)}")
            
            threading.Thread(target=mqtt_connect_task, daemon=True).start()
        except Exception as e:
            self.temp_log_cache.append(f"MQTT init failed: {str(e)}")

    def publish_run_mode(self):
        if not self.mqtt_client or not self.current_run_mode:
            return
        
        try:
            mode_data = {
                "mode": self.current_run_mode,
                "mode_name": self.current_run_mode_name,
                "timestamp": datetime.now().timestamp()
            }
            self.mqtt_client.publish(
                RUN_MODE_TOPIC,
                json.dumps(mode_data),
                qos=1,
                retain=False
            )
            self.temp_log_cache.append(f"Published run mode: {self.current_run_mode_name}")
        except Exception as e:
            self.temp_log_cache.append(f"Publish mode failed: {str(e)}")

    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(RECOMMEND_SPEED_TOPIC, qos=1)
            self.temp_log_cache.append(f"Subscribed to: {RECOMMEND_SPEED_TOPIC}")
        else:
            self.temp_log_cache.append(f"MQTT connect failed, rc: {rc}")

    def on_mqtt_message(self, client, userdata, msg):
        try:
            if msg.topic == RECOMMEND_SPEED_TOPIC:
                payload = msg.payload.decode('utf-8')
                try:
                    speed_data = json.loads(payload)
                    self.recommend_speed = float(speed_data.get("speed", 0.0))
                except:
                    self.recommend_speed = float(payload)
                
                with self.lock:
                    self.recommend_speed = round(self.recommend_speed, 2)
                
                self.temp_log_cache.append(f"Received recommend speed: {self.recommend_speed} km/h")
        except Exception as e:
            self.temp_log_cache.append(f"Parse speed failed: {str(e)}")

    def on_mqtt_disconnect(self, client, userdata, rc):
        if rc != 0:
            self.temp_log_cache.append(f"MQTT disconnected unexpectedly, rc: {rc}")
            def reconnect():
                time.sleep(5)
                try:
                    client.reconnect()
                except:
                    pass
            threading.Thread(target=reconnect, daemon=True).start()

    def create_ui(self):
        """创建UI（完全移除所有字体配置）"""
        # 1. 蓝牙操作区
        frame_bt = ttk.Frame(self.root, padding="8 4")
        frame_bt.pack(fill=tk.X, padx=4, pady=4)
        
        bt_status_label = ttk.Label(frame_bt, text="BT Status:")
        bt_status_label.grid(row=0, column=0, padx=3)
        
        self.bt_status = ttk.Label(frame_bt, text="Disconnected", foreground="red")
        self.bt_status.grid(row=0, column=1, padx=3)
        
        self.btn_scan = ttk.Button(frame_bt, text="Scan Devices", command=self.scan_bluetooth, width=12)
        self.btn_scan.grid(row=0, column=2, padx=3)
        self.btn_connect = ttk.Button(frame_bt, text="Connect", state=tk.DISABLED, command=self.connect_device, width=10)
        self.btn_connect.grid(row=0, column=3, padx=3)
        self.btn_history = ttk.Button(frame_bt, text="History", command=self.open_history, width=10)
        self.btn_history.grid(row=0, column=4, padx=15)
        self.btn_reset = ttk.Button(frame_bt, text="Reset Mileage", state=tk.DISABLED, command=self.reset_mileage, width=10)
        self.btn_reset.grid(row=0, column=5, padx=15)

        # 2. 设备列表
        frame_device = ttk.Frame(self.root, padding="8 4")
        frame_device.pack(fill=tk.X, padx=4, pady=4)
        
        device_label = ttk.Label(frame_device, text="Scanned Devices:")
        device_label.grid(row=0, column=0, sticky=tk.W)
        
        self.device_listbox = tk.Listbox(frame_device, height=4, width=100)
        self.device_listbox.grid(row=1, column=0, columnspan=6, pady=3, sticky=tk.W+tk.E)
        self.device_listbox.insert(tk.END, "Tip: Click 'Scan Devices' to get available devices")

        # 3. 核心数据区
        frame_data = ttk.Frame(self.root, padding="10 8", relief=tk.GROOVE, borderwidth=1)
        frame_data.pack(fill=tk.X, padx=8, pady=8)
        
        # 3.1 系统时间
        time_frame = ttk.Frame(frame_data)
        time_frame.grid(row=0, column=0, padx=15)
        ttk.Label(time_frame, text="System Time").pack()
        self.real_time_label = ttk.Label(time_frame, text="2025-01-01 00:00:00", foreground="#2c3e50")
        self.real_time_label.pack()
        
        # 3.2 心率
        hr_frame = ttk.Frame(frame_data)
        hr_frame.grid(row=0, column=1, padx=15)
        ttk.Label(hr_frame, text="Heart Rate").pack()
        self.hr_label = ttk.Label(hr_frame, text="--", foreground="#e74c3c")
        self.hr_label.pack()
        ttk.Label(hr_frame, text="bpm").pack()
        
        # 3.3 血氧
        spo2_frame = ttk.Frame(frame_data)
        spo2_frame.grid(row=0, column=2, padx=15)
        ttk.Label(spo2_frame, text="SPO2").pack()
        self.spo2_label = ttk.Label(spo2_frame, text="--", foreground="#3498db")
        self.spo2_label.pack()
        ttk.Label(spo2_frame, text="%").pack()
        
        # 3.4 里程
        mileage_frame = ttk.Frame(frame_data)
        mileage_frame.grid(row=0, column=3, padx=15)
        ttk.Label(mileage_frame, text="Mileage").pack()
        self.mileage_label = ttk.Label(mileage_frame, text="--", foreground="#27ae60")
        self.mileage_label.pack()
        ttk.Label(mileage_frame, text="m").pack()
        
        # 3.5 步数
        steps_frame = ttk.Frame(frame_data)
        steps_frame.grid(row=0, column=4, padx=15)
        ttk.Label(steps_frame, text="Steps").pack()
        self.steps_label = ttk.Label(steps_frame, text="--", foreground="#f39c12")
        self.steps_label.pack()
        ttk.Label(steps_frame, text="steps").pack()
        
        # 3.6 速率
        speed_frame = ttk.Frame(frame_data)
        speed_frame.grid(row=0, column=5, padx=15)
        ttk.Label(speed_frame, text="Speed").pack()
        self.speed_label = ttk.Label(speed_frame, text="--", foreground="#8e44ad")
        self.speed_label.pack()
        ttk.Label(speed_frame, text="km/h").pack()
        
        # 3.7 推荐配速
        recommend_speed_frame = ttk.Frame(frame_data)
        recommend_speed_frame.grid(row=0, column=6, padx=15)
        ttk.Label(recommend_speed_frame, text="Recommend Speed").pack()
        self.recommend_speed_label = ttk.Label(recommend_speed_frame, text="--", foreground="#e67e22")
        self.recommend_speed_label.pack()
        ttk.Label(recommend_speed_frame, text="km/h").pack()
        
        # 3.8 跑步状态
        status_frame = ttk.Frame(frame_data)
        status_frame.grid(row=0, column=7, padx=15)
        ttk.Label(status_frame, text="Status").pack()
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="#27ae60")
        self.status_label.pack()

        # 4. 时长+控制区
        frame_ctrl = ttk.Frame(self.root, padding="8 4")
        frame_ctrl.pack(fill=tk.X, padx=4, pady=4)
        
        duration_label = ttk.Label(frame_ctrl, text="Duration:")
        duration_label.grid(row=0, column=0, padx=10)
        
        self.duration_label = ttk.Label(frame_ctrl, text="00:00:00")
        self.duration_label.grid(row=0, column=1, padx=10)
        
        self.btn_start = ttk.Button(frame_ctrl, text="Start", state=tk.DISABLED, command=self.start_run, width=15)
        self.btn_start.grid(row=0, column=2, padx=20)
        self.btn_stop = ttk.Button(frame_ctrl, text="Stop", state=tk.DISABLED, command=self.stop_run, width=15)
        self.btn_stop.grid(row=0, column=3, padx=20)

        # 5. 日志区
        frame_log = ttk.Frame(self.root, padding="8 4")
        frame_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        log_title = ttk.Label(frame_log, text="Log (Time | HR | SPO2 | Mileage | Steps | Speed):")
        log_title.pack(anchor=tk.W)
        
        self.log_text = tk.Text(frame_log, height=12, state=tk.NORMAL)
        scrollbar = ttk.Scrollbar(frame_log, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def safe_ui_update(self, func):
        if self.is_ui_alive:
            try:
                self.root.after(0, func)
            except:
                pass

    def reset_mileage(self):
        self.send_bt_cmd("reset")
        with self.lock:
            self.current_mileage = 0.0
            self.current_steps = 0
            self.current_speed = 0.0
        self.log("Mileage reset command sent, local data cleared")
        self.show_non_blocking_msg("Info", "Mileage/Steps reset!")

    def log(self, msg):
        if self.is_ui_alive and hasattr(self, 'log_text') and self.log_text:
            if self.temp_log_cache:
                for cache_msg in self.temp_log_cache:
                    self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {cache_msg}\n")
                self.temp_log_cache.clear()
            self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            self.log_text.see(tk.END)
        else:
            self.temp_log_cache.append(msg)

    def init_db(self):
        try:
            self.log("Initializing database...")
            if not os.path.exists(DB_FILE):
                conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=1)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS running_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time TEXT NOT NULL,
                        start_timestamp REAL NOT NULL,
                        end_time TEXT NOT NULL,
                        end_timestamp REAL NOT NULL,
                        duration INTEGER NOT NULL,
                        avg_hr REAL NOT NULL,
                        avg_spo2 REAL NOT NULL,
                        total_mileage REAL NOT NULL,
                        total_steps INTEGER NOT NULL,
                        avg_speed REAL NOT NULL,
                        data_points TEXT NOT NULL
                    )
                ''')
                conn.commit()
                conn.close()
                self.log(f"Database created: {os.path.abspath(DB_FILE)}")
            
            self.conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=1)
            self.cursor = self.conn.cursor()
            self.log("Database initialized")
        except Exception as e:
            self.log(f"Database init failed: {str(e)}")
            self.show_non_blocking_msg("Error", f"DB init failed: {str(e)}", "error")

    def show_non_blocking_msg(self, title, msg, msg_type="info"):
        def show_msg():
            if not self.is_ui_alive:
                return
            try:
                if msg_type == "error":
                    messagebox.showerror(title, msg)
                elif msg_type == "warning":
                    messagebox.showwarning(title, msg)
                else:
                    messagebox.showinfo(title, msg)
            except:
                pass
        self.safe_ui_update(show_msg)

    def safe_serial_op(self, func):
        try:
            with self.lock:
                if not self.bt_serial or not self.bt_serial.is_open:
                    return False
                self.bt_serial.timeout = 0.5
                self.bt_serial.write_timeout = 0.5
                return func()
        except Exception as e:
            self.log(f"Serial operation error: {str(e)}")
            return False

    def send_bt_cmd(self, cmd):
        if self.bt_serial is None:
            self.log(f"[Sim Mode] Send command: {cmd}")
            return
        
        def cmd_task():
            def send_func():
                cmd_bytes = f"{cmd}\r\n".encode('utf-8')
                self.bt_serial.write(cmd_bytes)
                return True
            
            if self.safe_serial_op(send_func):
                self.log(f"[BT Command] Sent: {cmd}")
        
        threading.Thread(target=cmd_task, daemon=True).start()

    def scan_bluetooth(self):
        def scan_task():
            self.safe_ui_update(lambda: self.btn_scan.config(state=tk.DISABLED))
            self.safe_ui_update(lambda: self.bt_status.config(text="Scanning...", foreground="orange"))
            self.safe_ui_update(lambda: self.device_listbox.delete(0, tk.END))
            self.log("Scanning serial/BT devices...")
            
            devices = {}
            try:
                ports = serial.tools.list_ports.comports()
                self.log(f"Found {len(ports)} serial devices")
                
                for port in ports:
                    port_name = port.device
                    port_desc = port.description if port.description else "Unknown Device"
                    display_name = f"{port_desc} ({port_name})"
                    devices[display_name] = port_name
                    self.log(f"Device found: {display_name}")
                
                if not devices:
                    self.log("No physical devices found, add simulated device")
                    devices["Simulated Device (No BT Hardware)"] = "simulate"
            except Exception as e:
                self.log(f"Scan failed: {str(e)}")
                devices["Simulated Device (No BT Hardware)"] = "simulate"
            
            def update_list():
                if not self.is_ui_alive:
                    return
                self.scanned_devices = devices
                self.device_listbox.delete(0, tk.END)
                for name in devices.keys():
                    self.device_listbox.insert(tk.END, name)
                self.btn_connect.config(state=tk.NORMAL if devices else tk.DISABLED)
                self.bt_status.config(text=f"Scan Complete ({len(devices)} devices)", foreground="blue")
                self.btn_scan.config(state=tk.NORMAL)
                self.btn_reset.config(state=tk.NORMAL if devices else tk.DISABLED)
            
            self.safe_ui_update(update_list)
        
        threading.Thread(target=scan_task, daemon=True).start()

    def connect_device(self):
        try:
            selected_idx = self.device_listbox.curselection()[0]
            selected_name = self.device_listbox.get(selected_idx)
            selected_addr = self.scanned_devices[selected_name]
            self.log(f"Selected device: {selected_name} | Port: {selected_addr}")
        except IndexError:
            self.show_non_blocking_msg("Warning", "Please select a device first!", "warning")
            return
        
        self.thread_exit_flag.set()
        time.sleep(0.1)
        
        if self.bt_serial:
            self.log("Closing existing serial connection...")
            self.safe_serial_op(lambda: self.bt_serial.close() or True)
        
        if selected_addr == "simulate":
            self.bt_serial = None
            self.thread_exit_flag.clear()
            self.sim_thread = threading.Thread(target=self.sim_data, daemon=True)
            self.sim_thread.start()
            self.safe_ui_update(lambda: self.bt_status.config(text="Connected: Simulated Device", foreground="green"))
            self.safe_ui_update(lambda: self.btn_start.config(state=tk.NORMAL))
            self.safe_ui_update(lambda: self.btn_reset.config(state=tk.NORMAL))
            self.log("Entered simulated device mode")
            return
        
        def connect_task():
            try:
                self.log(f"Connecting to serial port: {selected_addr} (115200 baud)")
                ser = serial.Serial(
                    port=selected_addr,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.5,
                    write_timeout=0.5,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False
                )
                
                if ser.is_open:
                    self.bt_serial = ser
                    self.thread_exit_flag.clear()
                    self.read_thread = threading.Thread(target=self.read_serial, daemon=True)
                    self.read_thread.start()
                    
                    self.safe_ui_update(lambda: self.bt_status.config(text=f"Connected: {selected_addr}", foreground="green"))
                    self.safe_ui_update(lambda: self.btn_start.config(state=tk.NORMAL))
                    self.safe_ui_update(lambda: self.btn_reset.config(state=tk.NORMAL))
                    self.log(f"Serial port {selected_addr} connected, listening for data...")
                    self.show_non_blocking_msg("Success", f"Connected to serial port: {selected_addr}\nBaud rate: 115200")
                else:
                    raise Exception("Serial port not opened")
            except Exception as e:
                self.log(f"Serial connect failed: {str(e)}")
                def show_err():
                    if not self.is_ui_alive:
                        return
                    self.bt_status.config(text="Connection Failed", foreground="red")
                    self.show_non_blocking_msg("Error", f"Serial connect failed: {str(e)}\nSwitched to simulated mode", "error")
                    self.bt_serial = None
                    self.thread_exit_flag.clear()
                    self.sim_thread = threading.Thread(target=self.sim_data, daemon=True)
                    self.sim_thread.start()
                    self.bt_status.config(text="Connected: Simulated Device", foreground="green")
                    self.btn_start.config(state=tk.NORMAL)
                    self.btn_reset.config(state=tk.NORMAL)
                self.safe_ui_update(show_err)
        
        threading.Thread(target=connect_task, daemon=True).start()

    def read_serial(self):
        self.log("Started serial data reading thread")
        while not self.thread_exit_flag.is_set() and self.is_ui_alive:
            if self.is_running and self.bt_serial and self.bt_serial.is_open:
                def read_func():
                    return self.bt_serial.readline().decode('utf-8', errors='ignore').strip()
                
                raw_data = self.safe_serial_op(read_func)
                if raw_data and raw_data.strip() != "":
                    self.log(f"Received serial data: {raw_data}")
                    self.parse_data(raw_data)
            time.sleep(SERIAL_READ_INTERVAL)
        self.log("Serial reading thread exited")

    def sim_data(self):
        import random
        self.log("Started simulated data thread")
        sim_mileage = 0.0
        sim_steps = 0
        last_sim_time = time.time()
        
        while not self.thread_exit_flag.is_set() and self.is_ui_alive:
            if self.is_running:
                hr = random.randint(60, 150)
                spo2 = random.randint(90, 100)
                
                sim_steps += 1
                sim_mileage = sim_steps * 0.75
                
                current_time = time.time()
                elapsed = current_time - last_sim_time
                speed = (0.75 / elapsed) * 3.6 if elapsed > 0 else 0.0
                last_sim_time = current_time
                
                sim_data = f"HR:{hr},SPO2:{spo2},MILEAGE:{sim_mileage:.2f},STEPS:{sim_steps}"
                self.log(f"Generated simulated data: {sim_data} | Speed: {speed:.2f} km/h")
                self.parse_data(sim_data)
                time.sleep(1)
            else:
                sim_mileage = 0.0
                sim_steps = 0
                last_sim_time = time.time()
                time.sleep(0.5)
        self.log("Simulated data thread exited")

    def parse_data(self, raw_data):
        try:
            hr = None
            spo2 = None
            mileage = 0.0
            steps = 0
            
            if "HR:" in raw_data:
                hr_part = raw_data.split("HR:")[1].split(",")[0].strip()
                hr = int(hr_part) if hr_part.isdigit() else None
            if "SPO2:" in raw_data:
                spo2_part = raw_data.split("SPO2:")[1].split(",")[0].strip()
                spo2 = int(spo2_part) if spo2_part.isdigit() else None
            
            if "MILEAGE:" in raw_data:
                mileage_part = raw_data.split("MILEAGE:")[1].split(",")[0].strip()
                try:
                    mileage = float(mileage_part)
                except:
                    mileage = 0.0
            if "STEPS:" in raw_data:
                steps_part = raw_data.split("STEPS:")[1].split(",")[0].strip()
                steps = int(steps_part) if steps_part.isdigit() else 0
            
            speed = 0.0
            if self.is_running and self.start_time > 0 and mileage > 0:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    speed = (mileage / 1000) / (elapsed / 3600)
            
            if (hr is not None and spo2 is not None and 30 < hr < 250 and 70 <= spo2 <= 100) or mileage > 0:
                self.data_queue.put({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "hr": hr if hr else 0,
                    "spo2": spo2 if spo2 else 0,
                    "mileage": mileage,
                    "steps": steps,
                    "speed": round(speed, 2),
                    "ts": datetime.now().timestamp()
                }, timeout=0.1)
            else:
                self.log(f"Invalid data: {raw_data} (out of range or format error)")
        except Exception as e:
            self.log(f"Data parse failed: {raw_data} | Error: {str(e)}")

    def consume_data_task(self):
        if not self.is_ui_alive:
            return
            
        try:
            for _ in range(5):
                data = self.data_queue.get_nowait()
                with self.lock:
                    self.current_hr = data["hr"] if data["hr"] > 0 else self.current_hr
                    self.current_spo2 = data["spo2"] if data["spo2"] > 0 else self.current_spo2
                    self.current_mileage = data["mileage"]
                    self.current_steps = data["steps"]
                    self.current_speed = data["speed"]
                    
                    self.run_data.append({
                        "receive_time": data["time"],
                        "receive_timestamp": data["ts"],
                        "hr": data["hr"],
                        "spo2": data["spo2"],
                        "mileage": data["mileage"],
                        "steps": data["steps"],
                        "speed": data["speed"]
                    })
                
                log_line = (
                    f"{data['time']} | HR: {data['hr']:3d} bpm | SPO2: {data['spo2']:3d} % "
                    f"| Mileage: {data['mileage']:6.2f} m | Steps: {data['steps']:4d} steps "
                    f"| Speed: {data['speed']:5.2f} km/h\n"
                )
                self.log_text.insert(tk.END, log_line)
            self.log_text.see(tk.END)
        except queue.Empty:
            pass
        
        if self.is_ui_alive:
            self.root.after(DATA_QUEUE_INTERVAL, self.consume_data_task)

    def update_ui_task(self):
        if not self.is_ui_alive:
            return
            
        self.real_time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        with self.lock:
            self.hr_label.config(text=f"{self.current_hr}" if self.current_hr > 0 else "--")
            self.spo2_label.config(text=f"{self.current_spo2}" if self.current_spo2 > 0 else "--")
            self.mileage_label.config(text=f"{self.current_mileage:.1f}" if self.current_mileage > 0 else "--")
            self.steps_label.config(text=f"{self.current_steps}" if self.current_steps > 0 else "--")
            self.speed_label.config(text=f"{self.current_speed:.1f}" if self.current_speed > 0 else "--")
            self.recommend_speed_label.config(text=f"{self.recommend_speed:.1f}" if self.recommend_speed > 0 else "--")
            
            if self.is_running:
                elapsed = time.time() - self.start_time
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                self.duration_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")
                self.status_label.config(text="Running", foreground="#e74c3c")
            else:
                self.status_label.config(text="Ready", foreground="#27ae60")
        
        if self.is_ui_alive:
            self.root.after(UI_UPDATE_INTERVAL, self.update_ui_task)

    def start_run(self):
        with self.lock:
            self.is_running = True
            self.current_mileage = 0.0
            self.current_steps = 0
            self.current_speed = 0.0
        self.start_time = time.time()
        self.run_data.clear()
        self.log_text.delete(1.0, tk.END)
        
        self.send_bt_cmd("start")
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.log(f"Run started | Timestamp: {self.start_time} | Mode: {self.current_run_mode_name}")

    def stop_run(self):
        with self.lock:
            self.is_running = False
        end_time = time.time()
        
        self.send_bt_cmd("stop")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.log(f"Run stopped | Timestamp: {end_time}")
        
        def save_task():
            if not self.run_data:
                self.show_non_blocking_msg("Warning", "No valid running data to save!", "warning")
                return
            
            try:
                duration = int(end_time - self.start_time)
                valid_hr = [d['hr'] for d in self.run_data if d['hr'] > 0]
                valid_spo2 = [d['spo2'] for d in self.run_data if d['spo2'] > 0]
                valid_speed = [d['speed'] for d in self.run_data if d['speed'] > 0]
                
                avg_hr = sum(valid_hr) / len(valid_hr) if valid_hr else 0
                avg_spo2 = sum(valid_spo2) / len(valid_spo2) if valid_spo2 else 0
                total_mileage = self.current_mileage
                total_steps = self.current_steps
                avg_speed = sum(valid_speed) / len(valid_speed) if valid_speed else 0
                
                start_str = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")
                end_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
                data_str = json.dumps(self.run_data)
                
                with self.lock:
                    self.cursor.execute('''
                        INSERT INTO running_sessions 
                        (start_time, start_timestamp, end_time, end_timestamp, duration, 
                         avg_hr, avg_spo2, total_mileage, total_steps, avg_speed, data_points)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (start_str, self.start_time, end_str, end_time, duration, 
                          avg_hr, avg_spo2, total_mileage, total_steps, avg_speed, data_str))
                    self.conn.commit()
                
                self.log(f"Data saved | Duration: {duration}s | Avg HR: {avg_hr:.1f} | Avg SPO2: {avg_spo2:.1f} | Total Mileage: {total_mileage:.2f}m | Total Steps: {total_steps} | Avg Speed: {avg_speed:.2f}km/h")
                self.show_non_blocking_msg(
                    "Success",
                    f"✅ Running data saved\n"
                    f"📅 Time: {start_str} ~ {end_str}\n"
                    f"⏱️  Duration: {duration} s\n"
                    f"❤️ Avg HR: {avg_hr:.1f} bpm\n"
                    f"💨 Avg SPO2: {avg_spo2:.1f} %\n"
                    f"📏 Total Mileage: {total_mileage:.2f} m\n"
                    f"👟 Total Steps: {total_steps} steps\n"
                    f"⚡ Avg Speed: {avg_speed:.2f} km/h\n"
                    f"📊 Data points: {len(self.run_data)}"
                )
            except Exception as e:
                self.log(f"Data save failed: {str(e)}")
                self.show_non_blocking_msg("Error", f"❌ Save failed: {str(e)}", "error")
        
        threading.Thread(target=save_task, daemon=True).start()

    def open_history(self):
        HistoryWindow(self.root, self.conn)

    def safe_exit(self):
        self.is_ui_alive = False
        self.log("Program exiting, cleaning up resources...")
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        self.thread_exit_flag.set()
        
        if self.bt_serial:
            self.safe_serial_op(lambda: self.bt_serial.close() or True)
        
        time.sleep(0.5)
        
        if hasattr(self, 'conn'):
            self.conn.close()
        
        self.log("Resources cleaned up, program exited")
        
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

class HistoryWindow(Toplevel):
    def __init__(self, parent, db_conn):
        super().__init__(parent)
        self.title("Running History")
        self.geometry("1200x850")
        self.db_conn = db_conn
        self.cursor = self.db_conn.cursor()
        self.selected_session_id = None
        self.is_alive = True
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_ui()
        self.load_history_sessions()

    def on_close(self):
        self.is_alive = False
        self.destroy()

    def create_ui(self):
        # 列表区
        frame_list = ttk.Frame(self, padding="10 5")
        frame_list.pack(fill=tk.X, padx=5, pady=5)
        
        list_title = ttk.Label(frame_list, text="Running History (Latest First):")
        list_title.grid(row=0, column=0, sticky=tk.W)
        
        list_frame = ttk.Frame(frame_list)
        list_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)
        
        self.session_listbox = tk.Listbox(list_frame, height=10, width=110)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.session_listbox.yview)
        self.session_listbox.configure(yscrollcommand=scrollbar.set)
        self.session_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.session_listbox.bind('<<ListboxSelect>>', self.on_session_selected)
        
        btn_frame = ttk.Frame(frame_list)
        btn_frame.grid(row=2, column=0, pady=5)
        
        self.btn_plot = ttk.Button(
            btn_frame, text="Plot Curves", command=self.plot_time_curve,
            state=tk.DISABLED, width=20
        )
        self.btn_plot.grid(row=0, column=0, padx=10)
        
        self.btn_delete = ttk.Button(
            btn_frame, text="Delete Selected", command=self.delete_selected_session,
            state=tk.DISABLED, width=20
        )
        self.btn_delete.grid(row=0, column=1, padx=10)

        # 绘图区
        frame_plot = ttk.Frame(self, padding="10 5")
        frame_plot.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, ((self.ax_hr, self.ax_spo2), (self.ax_mileage, self.ax_speed)) = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        self.fig.suptitle("HR/SPO2/Mileage/Speed vs Time", fontsize=16, y=0.98)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_history_sessions(self):
        if not self.is_alive:
            return
            
        try:
            self.cursor.execute('''
                SELECT id, start_time, end_time, duration, avg_hr, avg_spo2, total_mileage, total_steps, avg_speed
                FROM running_sessions 
                ORDER BY start_timestamp DESC
            ''')
            sessions = self.cursor.fetchall()
            
            self.session_listbox.delete(0, tk.END)
            self.session_ids = []
            
            if not sessions:
                self.session_listbox.insert(tk.END, "📭 No running history")
                return
            
            for sess in sessions:
                sess_id, start, end, duration, avg_hr, avg_spo2, total_mileage, total_steps, avg_speed = sess
                self.session_ids.append(sess_id)
                
                duration_str = f"{duration // 60}m{duration % 60}s" if duration >= 60 else f"{duration}s"
                display_text = (
                    f"📅 {start} ~ {end} | ⏱️ {duration_str} "
                    f"| ❤️ Avg HR: {avg_hr:.1f} bpm | 💨 Avg SPO2: {avg_spo2:.1f} % "
                    f"| 📏 Total Mileage: {total_mileage:.2f}m | 👟 Total Steps: {total_steps}steps "
                    f"| ⚡ Avg Speed: {avg_speed:.2f}km/h"
                )
                self.session_listbox.insert(tk.END, display_text)
                
        except Exception as e:
            messagebox.showerror("Error", f"Load history failed: {str(e)}")

    def on_session_selected(self, event):
        if not self.is_alive:
            return
            
        selected_idx = self.session_listbox.curselection()
        if selected_idx and self.session_ids:
            self.selected_session_id = self.session_ids[selected_idx[0]]
            self.btn_plot.config(state=tk.NORMAL)
            self.btn_delete.config(state=tk.NORMAL)
        else:
            self.selected_session_id = None
            self.btn_plot.config(state=tk.DISABLED)
            self.btn_delete.config(state=tk.DISABLED)

    def plot_time_curve(self):
        if not self.is_alive or not self.selected_session_id:
            messagebox.showwarning("Warning", "Please select a history record first!")
            return
        
        try:
            self.cursor.execute('SELECT data_points FROM running_sessions WHERE id = ?', (self.selected_session_id,))
            data_str = self.cursor.fetchone()[0]
            data_points = json.loads(data_str)
            
            times = [datetime.fromtimestamp(d['receive_timestamp']) for d in data_points]
            hr = [d['hr'] for d in data_points]
            spo2 = [d['spo2'] for d in data_points]
            mileage = [d['mileage'] for d in data_points]
            speed = [d['speed'] for d in data_points]
            
            self.ax_hr.clear()
            self.ax_spo2.clear()
            self.ax_mileage.clear()
            self.ax_speed.clear()
            
            # HR curve
            self.ax_hr.plot(times, hr, color="#e74c3c", linewidth=2.5, marker='o', markersize=4,
                           label=f'HR (Avg: {np.mean(hr):.1f} bpm)')
            self.ax_hr.set_ylabel("HR (bpm)", fontsize=14, color="#c0392b")
            self.ax_hr.set_ylim(50, 160)
            self.ax_hr.grid(True, alpha=0.3, linestyle='--')
            self.ax_hr.legend(loc='upper right', fontsize=12)
            self.ax_hr.tick_params(axis='y', labelcolor="#c0392b")
            self.ax_hr.axhline(y=np.mean(hr), color="#e74c3c", linestyle='--', alpha=0.7, linewidth=2)
            
            # SPO2 curve
            self.ax_spo2.plot(times, spo2, color="#3498db", linewidth=2.5, marker='s', markersize=4,
                             label=f'SPO2 (Avg: {np.mean(spo2):.1f} %)')
            self.ax_spo2.set_ylabel("SPO2 (%)", fontsize=14, color="#2980b9")
            self.ax_spo2.set_ylim(85, 101)
            self.ax_spo2.grid(True, alpha=0.3, linestyle='--')
            self.ax_spo2.legend(loc='upper right', fontsize=12)
            self.ax_spo2.tick_params(axis='y', labelcolor="#2980b9")
            self.ax_spo2.axhline(y=np.mean(spo2), color="#3498db", linestyle='--', alpha=0.7, linewidth=2)
            
            # Mileage curve
            self.ax_mileage.plot(times, mileage, color="#27ae60", linewidth=2.5, marker='^', markersize=4,
                                label=f'Mileage (Total: {mileage[-1]:.2f} m)')
            self.ax_mileage.set_ylabel("Mileage (m)", fontsize=14, color="#219653")
            self.ax_mileage.set_ylim(0, max(mileage) * 1.1 if mileage else 1)
            self.ax_mileage.grid(True, alpha=0.3, linestyle='--')
            self.ax_mileage.legend(loc='upper right', fontsize=12)
            self.ax_mileage.tick_params(axis='y', labelcolor="#219653")
            
            # Speed curve
            self.ax_speed.plot(times, speed, color="#8e44ad", linewidth=2.5, marker='*', markersize=4,
                               label=f'Speed (Avg: {np.mean(speed):.2f} km/h)')
            self.ax_speed.set_xlabel("Time", fontsize=14)
            self.ax_speed.set_ylabel("Speed (km/h)", fontsize=14, color="#7b1fa2")
            self.ax_speed.set_ylim(0, max(speed) * 1.1 if speed else 10)
            self.ax_speed.grid(True, alpha=0.3, linestyle='--')
            self.ax_speed.legend(loc='upper right', fontsize=12)
            self.ax_speed.tick_params(axis='y', labelcolor="#7b1fa2")
            self.ax_speed.axhline(y=np.mean(speed), color="#8e44ad", linestyle='--', alpha=0.7, linewidth=2)
            
            self.fig.autofmt_xdate()
            plt.setp(self.ax_speed.xaxis.get_majorticklabels(), fontsize=10, rotation=45)
            
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Plot failed: {str(e)}")

    def delete_selected_session(self):
        if not self.is_alive or not self.selected_session_id:
            return
        
        if messagebox.askyesno("Confirm", "Are you sure to delete this record?\nThis action cannot be undone!"):
            try:
                self.cursor.execute('DELETE FROM running_sessions WHERE id = ?', (self.selected_session_id,))
                self.db_conn.commit()
                messagebox.showinfo("Success", "Record deleted!")
                self.load_history_sessions()
                self.ax_hr.clear()
                self.ax_spo2.clear()
                self.ax_mileage.clear()
                self.ax_speed.clear()
                self.canvas.draw()
            except Exception as e:
                messagebox.showerror("Error", f"Delete failed: {str(e)}")

if __name__ == "__main__":
    # 自动安装依赖
    required_packages = {
        "serial": "pyserial",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "paho.mqtt.client": "paho-mqtt"
    }
    
    for pkg_import, pkg_install in required_packages.items():
        try:
            __import__(pkg_import)
        except ImportError:
            print(f"Installing dependency: {pkg_install}")
            os.system(f"pip install {pkg_install} -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet")
    
    # 启动程序
    try:
        root = tk.Tk()
        app = RunningMonitor(root)
        root.mainloop()
    except Exception as e:
        print(f"Program error: {str(e)}")
        input("Press Enter to exit...")
        
#C:\Users\sxhzx\AppData\Local\Programs\Python\Python38\python.exe d:/Users/zhu/Projects/Iot/phone_app.py  
