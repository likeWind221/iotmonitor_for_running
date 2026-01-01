#!/usr/bin/env python3
# 读取 sensor_data.db - 每1秒读取一次
import sqlite3
import time
import os
from datetime import datetime
import signal

# 配置项
DB_PATH = r"C:\Users\sxhzx\sensor_data.db"  # Windows 路径转义
READ_INTERVAL = 1.0  # 1秒读取一次
EXIT_FLAG = False

def signal_handler(sig, frame):
    """优雅退出处理"""
    global EXIT_FLAG
    EXIT_FLAG = True
    print("\n⚠️  接收到退出信号，正在停止程序...")

def init_db_connection(db_path):
    """初始化数据库连接（带容错）"""
    try:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        # 建立连接（设置超时，避免锁死）
        conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=5.0
        )
        conn.row_factory = sqlite3.Row  # 支持按列名访问
        cursor = conn.cursor()
        print(f"✅ 成功连接到数据库: {db_path}")
        return conn, cursor
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return None, None

def get_table_structure(cursor, table_name):
    """获取表结构（自动识别表名）"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        return columns
    except Exception as e:
        print(f"❌ 获取表 {table_name} 结构失败: {e}")
        return []

def get_all_tables(cursor):
    """获取数据库中所有表名"""
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        return tables
    except Exception as e:
        print(f"❌ 获取表列表失败: {e}")
        return []

def read_sensor_data():
    """主读取逻辑"""
    # 注册退出信号（Ctrl+C）
    signal.signal(signal.SIGINT, signal_handler)
    
    # 初始化数据库连接
    conn, cursor = init_db_connection(DB_PATH)
    if not conn or not cursor:
        return
    
    # 获取所有表名
    tables = get_all_tables(cursor)
    if not tables:
        print("❌ 数据库中未找到任何表")
        conn.close()
        return
    
    print(f"\n📋 检测到数据库表: {tables}")
    print(f"⏱️  开始每 {READ_INTERVAL} 秒读取一次数据（按 Ctrl+C 退出）")
    print("-" * 80)
    
    try:
        while not EXIT_FLAG:
            # 记录当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # 遍历所有表读取数据
            for table in tables:
                try:
                    # 获取最新一条数据（按rowid降序，兼容无时间戳表）
                    cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 1")
                    row = cursor.fetchone()
                    
                    if row:
                        # 获取列名并格式化输出
                        columns = get_table_structure(cursor, table)
                        print(f"[{current_time}] 表 {table} 最新数据:")
                        for col in columns:
                            print(f"  - {col}: {row[col]}")
                    else:
                        print(f"[{current_time}] 表 {table}: 暂无数据")
                except Exception as e:
                    print(f"[{current_time}] 读取表 {table} 失败: {e}")
            
            # 分隔线
            print("-" * 80)
            
            # 等待指定间隔（响应退出信号）
            start_wait = time.time()
            while (time.time() - start_wait) < READ_INTERVAL and not EXIT_FLAG:
                time.sleep(0.01)
    
    finally:
        # 关闭数据库连接
        conn.close()
        print("\n✅ 数据库连接已关闭，程序退出")

if __name__ == "__main__":
    read_sensor_data()