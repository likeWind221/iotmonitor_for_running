"""
MQTT 订阅脚本 - 监听传感器数据
用法：python mqtt_subscriber.py
"""
import paho.mqtt.client as mqtt
import json
import threading
import time
from queue import Queue, Empty
from collections import deque
import statistics
import os
import csv
import sys
import pickle
import torch
import logging
# module logger
log = logging.getLogger('mqtt_subscriber')
logging.basicConfig(level=logging.INFO)
# ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
try:
    from generate_training_data import fatigue_score_from_window, expert_rule_action
except Exception:
    # fallback simple fatigue
    def fatigue_score_from_window(avg_hr, hr_slope, angle_std, age=25, resting_hr=60):
        try:
            max_hr = 220 - age
            hr_rel = 0.0
            if avg_hr is not None:
                hr_rel = (avg_hr - resting_hr) / max(1, (max_hr - resting_hr))
                hr_rel = min(max(hr_rel, 0.0), 1.0)
            ang_norm = min(max(angle_std / 20.0, 0.0), 1.0)
            slope_norm = min(max(hr_slope / 5.0, 0.0), 1.0)
            w1, w2, w3 = 0.5, 0.3, 0.2
            score = w1 * hr_rel + w2 * slope_norm + w3 * ang_norm
            return round(min(max(score, 0.0), 1.0), 3)
        except Exception:
            return 0.0
    # fallback expert rule (same logic as in generate_training_data)
    def expert_rule_action(speed, target_speed, spo2, fatigue, hr_ratio):
        if spo2 is not None and spo2 < 92:
            return 'dec'
        if hr_ratio is not None and hr_ratio > 0.95:
            return 'dec'
        if speed is None:
            return 'keep'
        if speed < target_speed * 0.9:
            return 'acc'
        if speed > target_speed * 1.1:
            return 'dec'
        if fatigue > 0.7 and speed > target_speed * 0.95:
            return 'dec'
        return 'keep'

# local mapping for target speeds
TARGET_SPEED = {'walk':1.0, 'jog':2.5, 'sprint':5.0}

# helper utilities for pace and fatigue text
def mps_to_pace_seconds(speed_mps):
    if speed_mps is None or speed_mps <= 0:
        return None
    return 1000.0 / speed_mps

def mps_to_pace_str(speed_mps):
    secs = mps_to_pace_seconds(speed_mps)
    if secs is None:
        return 'N/A'
    mins = int(secs // 60)
    s = int(round(secs % 60))
    return f"{mins}:{s:02d} min/km"

def mps_to_pace_min_float(speed_mps):
    secs = mps_to_pace_seconds(speed_mps)
    if secs is None:
        return None
    return round(secs / 60.0, 3)

def fatigue_level_text(score):
    if score is None:
        return 'unknown'
    if score < 0.3:
        return 'low'
    if score < 0.6:
        return 'medium'
    return 'high'

# model artifacts (optional)
MODEL_PATH = 'models/bc/bc_model_final.pth'
SCALER_PATH = 'models/bc/scaler.pkl'
TEMPERATURE_PATH = 'models/bc/temperature.txt'
MODEL_AVAILABLE = False
model = None
scaler = None
TEMPERATURE = 1.0
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        from scripts.train_bc import MLP, FEATURES, INV_LABEL_MAP
        scaler = pickle.load(open(SCALER_PATH, 'rb'))
        model = MLP(in_dim=len(FEATURES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        MODEL_AVAILABLE = True
        # load temperature if available
        try:
            if os.path.exists(TEMPERATURE_PATH):
                with open(TEMPERATURE_PATH, 'r', encoding='utf-8') as f:
                    TEMPERATURE = float(f.read().strip())
                print(f'✅ 已加载 BC 模型与 scaler，并启用温度缩放 T={TEMPERATURE:.3f}')
            else:
                print('✅ 已加载 BC 模型与 scaler（未找到 temperature，默认 T=1.0）')
        except Exception:
            print('⚠️ 加载 temperature 失败，使用默认 T=1.0')
    else:
        print('⚠️ 未找到模型或 scaler，跳过实时推理（预计路径：', MODEL_PATH, SCALER_PATH, ')')
except Exception as e:
    print('⚠️ 加载模型失败，跳过推理：', e)

# ========== 序列模型（运动模式识别）配置与加载 ==========
SEQ_MODEL_PATH = 'models/seq_full/gru_best.pth'
SEQ_SCALER_PATH = 'models/seq_full/scaler.pkl'
SEQ_LABELMAP_PATH = 'models/seq_full/label_map.json'
SEQ_LEN = 12
SEQ_HIDDEN = 128
SEQ_FEATURES = ['avg_hr','hr_slope','avg_spo2','speed_mps','speed_trend','angleX_mean','angleX_std','angleY_mean','angleY_std','fatigue_score']
SEQ_MODEL_AVAILABLE = False
seq_model = None
seq_scaler = None
seq_inv_label = None
seq_buffer = deque(maxlen=SEQ_LEN)
# mapping from numeric codes (if present) to mode names
CODE_NAME_MAP = {'0':'walk', '1':'jog', '2':'sprint'}
# human readable Chinese names
CHINESE_MODE = {'walk':'走路', 'jog':'慢跑', 'sprint':'快跑'}

try:
    if os.path.exists(SEQ_MODEL_PATH) and os.path.exists(SEQ_SCALER_PATH) and os.path.exists(SEQ_LABELMAP_PATH):
        try:
            from scripts.train_sequence import GRUClassifier
        except Exception as e:
            # fallback local GRU definition so mqtt_subscriber doesn't depend on scripts imports
            print('⚠️ 导入 scripts.train_sequence 失败，使用本地 GRUClassifier 作为后备：', e)
            import torch.nn as nn
            class GRUClassifier(nn.Module):
                def __init__(self, input_dim, hidden_dim=64, num_layers=1, num_classes=3, dropout=0.2):
                    super().__init__()
                    self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
                    self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
                def forward(self, x):
                    out, h = self.gru(x)
                    last = out[:, -1, :]
                    return self.fc(last)
        seq_scaler = pickle.load(open(SEQ_SCALER_PATH, 'rb'))
        with open(SEQ_LABELMAP_PATH, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        # label_map: original_label -> index (JSON keys are strings); invert to index->label
        seq_inv_label = {int(v): k for k, v in label_map.items()}
        n_classes = len(seq_inv_label)
        seq_model = GRUClassifier(input_dim=len(SEQ_FEATURES), hidden_dim=SEQ_HIDDEN, num_classes=n_classes)
        seq_model.load_state_dict(torch.load(SEQ_MODEL_PATH, map_location='cpu'))
        seq_model.eval()
        SEQ_MODEL_AVAILABLE = True
        print('✅ 已加载序列模型与 scaler，用于实时模式识别')
    else:
        print('⚠️ 未找到序列模型或 scaler，跳过序列推理（预计路径：', SEQ_MODEL_PATH, SEQ_SCALER_PATH, SEQ_LABELMAP_PATH, ')')
except Exception as e:
    print('⚠️ 加载序列模型失败：', e)


# ========== 配置 ==========
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
TOPICS = ["sensor/mpu", "sensor/data"]

# ========== 非阻塞队列与日志配置 ==========
MSG_QUEUE_MAXSIZE = 2000
msg_queue = Queue(maxsize=MSG_QUEUE_MAXSIZE)
stop_event = threading.Event()
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'analysis.csv')
LOG_MSGS = False  # 若为 True，则打印收到的原始消息以便调试
# 创建 CSV 表头（若不存在），若存在但缺失列则进行补头处理
EXPECTED_HEADER = [
    'window_start_ms','window_end_ms','count_mpu','count_data','avg_hr','avg_spo2','speed_mps','current_pace_min_per_km','angleX_mean','angleX_std','angleY_mean','angleY_std','target_mode','target_speed','target_pace_min_per_km','rule_action','final_action','final_action_source','suggested_speed_mps','suggested_pace_min_per_km','model_action','model_conf','seq_mode','seq_conf','fatigue_score','fatigue_level'
]
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_HEADER)
else:
    # validate existing header, and if missing columns, rewrite header and pad older rows
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        if len(rows) == 0:
            # empty file
            with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(EXPECTED_HEADER)
        else:
            header = rows[0]
            if header != EXPECTED_HEADER:
                # pad existing rows to new width
                new_rows = []
                new_rows.append(EXPECTED_HEADER)
                old_width = len(header)
                new_width = len(EXPECTED_HEADER)
                for r in rows[1:]:
                    new_r = r + [''] * (new_width - old_width) if len(r) < new_width else r[:new_width]
                    new_rows.append(new_r)
                with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(new_rows)
    except Exception as e:
        # fall back: recreate header
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(EXPECTED_HEADER)

# ========== MQTT 回调函数 ==========
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ 连接到 MQTT Broker 成功")
        for topic in TOPICS:
            client.subscribe(topic)
            print(f"📡 已订阅主题: {topic}")
    else:
        print(f"❌ 连接失败，代码: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        # 非阻塞入队：(topic, payload, timestamp_ms)
        timestamp_ms = payload.get('timestamp_ms') or int(time.time() * 1000)
        # 可选：打印接收到的原始消息，用于调试网络/设备侧是否正常发上来数据
        try:
            if LOG_MSGS:
                try:
                    print(f"📥 MQTT {msg.topic} @{timestamp_ms}: {json.dumps(payload, ensure_ascii=False)[:200]}")
                except Exception:
                    print(f"📥 MQTT {msg.topic} @{timestamp_ms}: {str(payload)[:200]}")
        except Exception:
            pass
        try:
            msg_queue.put_nowait((msg.topic, payload, int(timestamp_ms)))
        except Exception:
            print("⚠️ 队列已满，消息被丢弃")
    except json.JSONDecodeError:
        print(f"❌ 解析 JSON 失败: {msg.payload.decode()}")
    except Exception as e:
        print(f"❌ 处理消息时出错: {e}")

# ========== 分析线程（5s 窗口） ==========
def analysis_worker(window_size=5, target_mode=None, acc_delta=0.3, dec_delta=0.5, max_pace_diff_sec=20.0, smooth_alpha=0.5, model_conf_high=0.95, require_consecutive=2, debug_action_log=False):
    """后台线程：每 window_size 秒从队列收集消息，计算简单统计并写日志。
    新增参数用于控制建议平滑与置信度判断：
    - max_pace_diff_sec: 单次建议与当前配速的最大差异（以秒/公里计）
    - smooth_alpha: 建议平滑系数（0-1，1 表示不平滑）
    - model_conf_high: 若模型置信度高于该值则优先采信模型
    - require_consecutive: 若连续 N 窗口给出相同模型动作则也采信模型
    """
    print(f"🧠 分析线程已启动，窗口大小={window_size}s, target_mode={target_mode}, acc_delta={acc_delta}, dec_delta={dec_delta}")
    if target_mode not in TARGET_SPEED:
        print("❌ 错误：必须显式指定 --target-mode 为 walk/jog/sprint")
        stop_event.set()
        return
    prev_hr = None
    prev_speed = None
    prev_suggested = None
    from collections import deque
    model_action_history = deque(maxlen=max(1, require_consecutive))
    # helper to clip pace difference in seconds/km
    def clip_suggested_speed_by_pace(cur_speed, suggested_speed, max_diff_sec):
        try:
            cur_secs = mps_to_pace_seconds(cur_speed)
            sugg_secs = mps_to_pace_seconds(suggested_speed)
            if cur_secs is None or sugg_secs is None:
                return suggested_speed, False
            delta = sugg_secs - cur_secs
            if abs(delta) <= max_diff_sec:
                return suggested_speed, False
            # clip
            sign = 1 if delta > 0 else -1
            new_secs = cur_secs + sign * max_diff_sec
            new_speed = 1000.0 / new_secs
            return new_speed, True
        except Exception:
            return suggested_speed, False
    while not stop_event.is_set():
        t0 = time.time()
        window = []
        # 收集窗口内消息
        while time.time() - t0 < window_size and not stop_event.is_set():
            try:
                item = msg_queue.get(timeout=0.5)
                window.append(item)
            except Empty:
                continue
        if not window:
            continue
        # 聚合
        window_start_ms = min(x[2] for x in window)
        window_end_ms = max(x[2] for x in window)
        mpu_msgs = [p for (topic, p, ts) in window if topic == 'sensor/mpu']
        data_msgs = [p for (topic, p, ts) in window if topic == 'sensor/data']

        count_mpu = len(mpu_msgs)
        count_data = len(data_msgs)

        # 生理统计
        hr_values = [p.get('hr') for p in data_msgs if isinstance(p.get('hr'), (int, float))]
        spo2_values = [p.get('spo2') for p in data_msgs if isinstance(p.get('spo2'), (int, float))]
        avg_hr = round(statistics.mean(hr_values), 2) if hr_values else None
        avg_spo2 = round(statistics.mean(spo2_values), 2) if spo2_values else None

        # 角度统计
        angleX_vals = [p.get('angleX') for p in mpu_msgs if isinstance(p.get('angleX'), (int, float))]
        angleY_vals = [p.get('angleY') for p in mpu_msgs if isinstance(p.get('angleY'), (int, float))]
        angleX_mean = round(statistics.mean(angleX_vals), 2) if angleX_vals else None
        angleX_std = round(statistics.pstdev(angleX_vals), 2) if angleX_vals else None
        angleY_mean = round(statistics.mean(angleY_vals), 2) if angleY_vals else None
        angleY_std = round(statistics.pstdev(angleY_vals), 2) if angleY_vals else None

        # 速度估算（使用 data 消息的 total_mileage_m）
        speeds = None
        if len(data_msgs) >= 2:
            # 按时间排序
            sorted_data = sorted(data_msgs, key=lambda p: p.get('time', ''))
            # find first and last with mileage
            first = next((p for p in sorted_data if isinstance(p.get('total_mileage_m'), (int, float, float))), None)
            last = next((p for p in reversed(sorted_data) if isinstance(p.get('total_mileage_m'), (int, float, float))), None)
            if first and last and first != last:
                # 尝试使用 their timestamps from queue (approximate)
                first_ts = next(ts for (topic,p,ts) in window if p is first)
                last_ts = next(ts for (topic,p,ts) in reversed(window) if p is last)
                delta_m = last.get('total_mileage_m', 0) - first.get('total_mileage_m', 0)
                delta_t = max(1, (last_ts - first_ts) / 1000.0)
                speeds = round(delta_m / delta_t, 3)

        # 额外特征：hr_slope, speed_trend, fatigue
        hr_slope = None
        speed_trend = None
        if avg_hr is not None and prev_hr is not None:
            try:
                hr_slope = round(avg_hr - prev_hr, 3)
            except Exception:
                hr_slope = 0.0
        else:
            hr_slope = 0.0
        if speeds is not None and prev_speed is not None:
            try:
                speed_trend = round(speeds - prev_speed, 3)
            except Exception:
                speed_trend = 0.0
        else:
            speed_trend = 0.0
        angle_std = (angleX_std or 0.0) + (angleY_std or 0.0)
        fatigue = fatigue_score_from_window(avg_hr, hr_slope, angle_std)

        # compute hr_ratio for safety/rule
        max_hr = 220 - 25
        hr_ratio = None
        if avg_hr is not None:
            hr_ratio = (avg_hr - 60) / max(1, (max_hr - 60))

        # determine target_mode and target_speed (user must specify target_mode)
        tmode = target_mode
        if tmode not in TARGET_SPEED:
            print("❌ 错误：target_mode 必须为 'walk'|'jog'|'sprint'.")
            stop_event.set()
            return
        target_speed_val = TARGET_SPEED[tmode]

        # rule-based suggestion (immediate) - still computed as fallback but not used as primary
        rule_action = expert_rule_action(speeds, target_speed_val, avg_spo2, fatigue, hr_ratio)

        # translated action labels
        ACTION_CHINESE = {'acc':'加速', 'dec':'减速', 'keep':'保持'}
        CHINESE_FATIGUE = {'low':'低', 'medium':'中', 'high':'高', 'unknown':'未知'}

        # suggested_speed will be computed after model inference (prefer model's action)
        suggested_speed = None

        # compute human readable paces
        current_pace_min = mps_to_pace_min_float(speeds)
        target_pace_min = mps_to_pace_min_float(target_speed_val)
        fatigue_lvl = fatigue_level_text(fatigue)

        # inference (if model available) - include target_speed in features
        model_action = ''
        model_conf = ''
        # sequence model outputs
        seq_mode = ''
        seq_conf = ''
        if MODEL_AVAILABLE:
            try:
                feat = [avg_hr or 0.0, hr_slope or 0.0, avg_spo2 or 0.0, speeds or 0.0, speed_trend or 0.0, angleX_mean or 0.0, angleX_std or 0.0, angleY_mean or 0.0, angleY_std or 0.0, target_speed_val or 0.0, fatigue or 0.0]
                X = scaler.transform([feat])
                xb = torch.from_numpy(X.astype('float32'))
                with torch.no_grad():
                    logits = model(xb)
                    # raw probs (before temperature)
                    probs_raw = torch.softmax(logits, dim=1).cpu().numpy().ravel()
                    # apply temperature scaling if available
                    try:
                        temp = float(TEMPERATURE) if TEMPERATURE is not None else 1.0
                        logits_scaled = logits / temp
                        probs = torch.softmax(logits_scaled, dim=1).cpu().numpy().ravel()
                    except Exception:
                        probs = probs_raw
                max_p = float(probs.max())
                act = int(probs.argmax())
                model_action = INV_LABEL_MAP[act]
                model_conf = round(max_p,3)
                # also keep raw max prob for debugging
                raw_max_p = round(float(probs_raw.max()),3)
            except Exception as e:
                print('⚠️ 推理失败：', e)
                model_action = ''
                model_conf = ''
                raw_max_p = ''

        else:
            model_action = ''
            model_conf = ''

        # decide final action (prefer model if available)
        final_action = model_action if MODEL_AVAILABLE and model_action else rule_action
        final_action_ch = ACTION_CHINESE.get(final_action, final_action)
        fatigue_ch = CHINESE_FATIGUE.get(fatigue_lvl, fatigue_lvl)

        # suggested speed now based on model-decision (or fallback to rule) with stability和常识校验
        action_adjusted = False
        final_action_use = final_action
        final_action_before = final_action_use
        final_action_after_sanity = final_action_use
        try:
            model_conf_val = 0.0
            try:
                model_conf_val = float(model_conf) if model_conf != '' else 0.0
            except Exception:
                model_conf_val = 0.0

            # update model action history
            model_action_history.append(model_action if model_action else None)
            # decide whether to trust model: either high conf or repeated same action
            trust_model = False
            if MODEL_AVAILABLE and model_action:
                if model_conf_val >= model_conf_high:
                    trust_model = True
                else:
                    # require consecutive occurrences
                    if model_action_history.count(model_action) >= model_action_history.maxlen and model_action_history.maxlen > 1:
                        trust_model = True
            # tentative decision: trust model only if trust_model True
            final_action_use = final_action if trust_model else rule_action
            final_action_before = final_action_use

            # sensible current speed
            if speeds is None or speeds <= 0.05:
                cur = target_speed_val
            else:
                cur = speeds

            # sanity: 若已低于或接近目标配速且疲劳不高，不接受“减速”
            under_tol = 0.02  # 2% 容差
            if cur is not None and final_action_use == 'dec':
                if cur < target_speed_val * (1 - under_tol) and fatigue_lvl != 'high':
                    log.info('常识兜底：当前已慢于目标且疲劳%s，跳过减速，改为加速', fatigue_lvl)
                    final_action_use = 'acc'
                elif fatigue_lvl == 'low' and cur <= target_speed_val * (1 + under_tol):
                    log.info('常识兜底：低疲劳且已接近/慢于目标，保持不减速')
                    final_action_use = 'keep'
            final_action_after_sanity = final_action_use

            # additional conservative check: reject model if its suggested direction contradicts current speed
            model_rejected_direction = False
            if trust_model and model_action:
                try:
                    if model_action == 'dec' and cur < 0.9 * target_speed_val:
                        model_rejected_direction = True
                    if model_action == 'acc' and cur > 1.1 * target_speed_val:
                        model_rejected_direction = True
                except Exception:
                    model_rejected_direction = False
                if model_rejected_direction:
                    trust_model = False
                    final_action_use = rule_action
                    log.info('拒绝模型建议（方向不符）：model=%s cur=%.3f target=%.3f', model_action, cur, target_speed_val)

            # compute deltas but limit changes to reasonable fractions
            acc_abs = min(acc_delta, 0.2 * target_speed_val)
            dec_abs = min(dec_delta, 0.3 * target_speed_val)
            min_allowed_speed = max(0.1, 0.3 * target_speed_val)

            if final_action_use == 'acc':
                suggested_speed = min(target_speed_val, cur + acc_abs)
            elif final_action_use == 'dec':
                suggested_speed = max(min_allowed_speed, cur - dec_abs)
            else:
                suggested_speed = target_speed_val

            # clip suggested speed so that pace difference not exceed max_pace_diff_sec
            clipped = False
            if cur is not None and suggested_speed is not None:
                suggested_speed, clipped = clip_suggested_speed_by_pace(cur, suggested_speed, max_pace_diff_sec)

            # smoothing with previous suggestion
            prev_before = prev_suggested
            if prev_suggested is not None and suggested_speed is not None and smooth_alpha is not None:
                suggested_speed = smooth_alpha * suggested_speed + (1 - smooth_alpha) * prev_suggested

            # ensure printed动作方向与最终速度一致，避免“减速但配速更快”
            if suggested_speed is not None and cur is not None:
                eps = 1e-6
                if suggested_speed > cur + eps:
                    action_dir = 'acc'
                elif suggested_speed < cur - eps:
                    action_dir = 'dec'
                else:
                    action_dir = 'keep'
                if action_dir != final_action_use:
                    action_adjusted = True
                    final_action_use = action_dir
                    log.info('平滑/裁剪后方向矫正: 原=%s 新=%s (cur=%.3f, sugg=%.3f)', final_action_before, final_action_use, cur, suggested_speed)

            if clipped:
                log.info('建议已按最大配速差限制裁剪: new_pace=%s min/km', mps_to_pace_str(suggested_speed))

            if debug_action_log:
                try:
                    dbg = {
                        'cur_speed_mps': round(cur,3) if cur is not None else None,
                        'target_speed_mps': round(target_speed_val,3),
                        'final_action_use': final_action_use,
                        'model_action': model_action,
                        'model_conf': round(model_conf_val,3),
                        'model_raw_conf': raw_max_p if 'raw_max_p' in locals() else None,
                        'model_history': list(model_action_history),
                        'acc_abs': round(acc_abs,3),
                        'dec_abs': round(dec_abs,3),
                        'min_allowed_speed': round(min_allowed_speed,3),
                        'suggested_speed_mps': round(suggested_speed,3) if suggested_speed is not None else None,
                        'clipped': clipped,
                        'prev_suggested_mps': round(prev_before,3) if prev_before is not None else None,
                        'smooth_alpha': round(smooth_alpha,3),
                        'max_pace_diff_sec': round(max_pace_diff_sec,3),
                        'temp_T': round(TEMPERATURE,3) if TEMPERATURE is not None else None,
                        'model_rejected_direction': model_rejected_direction if 'model_rejected_direction' in locals() else False,
                        'final_action_before_adjust': final_action_before,
                        'final_action_after_sanity': final_action_after_sanity,
                        'final_action_after_adjust': final_action_use,
                        'action_adjusted': action_adjusted
                    }
                    print('DEBUG_ACTION:', dbg)
                except Exception:
                    pass

            prev_suggested = suggested_speed
        except Exception:
            suggested_speed = target_speed_val
            prev_suggested = suggested_speed

        # compute suggested pace now that suggested_speed is known
        suggested_pace_min = mps_to_pace_min_float(suggested_speed)

        # determine final action label and which source (模型/规则) produced it
        final_action_ch = ACTION_CHINESE.get(final_action_use, final_action_use)
        final_action_src = '模型' if (MODEL_AVAILABLE and model_action and trust_model) else '规则'

        # print unified Chinese output reflecting the actual decision source
        if final_action_src == '模型':
            print(f"💬 最终配速建议（基于模型）：{final_action_ch} (conf={model_conf})；建议配速：{mps_to_pace_str(suggested_speed)} ({round(suggested_speed,3)} m/s)；疲劳：{fatigue_ch}({round(fatigue,3)})；当前配速：{mps_to_pace_str(speeds)}；目标配速：{mps_to_pace_str(target_speed_val)}")
        else:
            print(f"💬 最终配速建议（基于规则）：{final_action_ch}；建议配速：{mps_to_pace_str(suggested_speed)} ({round(suggested_speed,3)} m/s)；疲劳：{fatigue_ch}({round(fatigue,3)})；当前配速：{mps_to_pace_str(speeds)}；目标配速：{mps_to_pace_str(target_speed_val)}")

        # sequence model inference (activity recognition)
        if SEQ_MODEL_AVAILABLE:
            try:
                current_feat = [avg_hr or 0.0, hr_slope or 0.0, avg_spo2 or 0.0, speeds or 0.0, speed_trend or 0.0, angleX_mean or 0.0, angleX_std or 0.0, angleY_mean or 0.0, angleY_std or 0.0, fatigue or 0.0]
                seq_buffer.append(current_feat)
                if len(seq_buffer) < SEQ_LEN:
                    print(f"ℹ️ 序列模型缓冲尚未填满：{len(seq_buffer)}/{SEQ_LEN}（需要{SEQ_LEN}个窗口后才开始预测）")
                if len(seq_buffer) == SEQ_LEN:
                    import numpy as np
                    seq_arr = np.array(seq_buffer, dtype=np.float32)  # (L, F)
                    seq_norm = seq_scaler.transform(seq_arr)
                    xb = torch.from_numpy(seq_norm.reshape(1, SEQ_LEN, -1)).float()
                    with torch.no_grad():
                        logits = seq_model(xb)
                        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
                    act = int(probs.argmax())
                    raw_label = seq_inv_label.get(act, '')
                    seq_conf = round(float(probs.max()), 3)
                    # translate to readable name (prefer Chinese)
                    s = str(raw_label)
                    if s in CHINESE_MODE:
                        seq_mode_display = CHINESE_MODE[s]
                    elif s in CODE_NAME_MAP:
                        seq_mode_display = CHINESE_MODE.get(CODE_NAME_MAP[s], CODE_NAME_MAP[s])
                    else:
                        seq_mode_display = s
                    seq_mode = seq_mode_display
                    print(f"🔎 模式识别（模型）：{seq_mode} (conf={seq_conf})")
            except Exception as e:
                print('⚠️ 序列推理失败：', e)

        # 输出与记录（含 pace 与建议）
        # print(f"🔬 窗口 {window_start_ms} - {window_end_ms}: MPU={count_mpu}, DATA={count_data}, avg_hr={avg_hr}, avg_spo2={avg_spo2}, speed_mps={speeds}, pace={mps_to_pace_str(speeds)}")
        # 写 CSV（新增 target_mode/target_speed/rule_action 等）
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                window_start_ms,
                window_end_ms,
                count_mpu,
                count_data,
                avg_hr if avg_hr is not None else '',
                avg_spo2 if avg_spo2 is not None else '',
                speeds if speeds is not None else '',
                current_pace_min if current_pace_min is not None else '',
                angleX_mean if angleX_mean is not None else 0.0,
                angleX_std if angleX_std is not None else 0.0,
                angleY_mean if angleY_mean is not None else 0.0,
                angleY_std if angleY_std is not None else 0.0,
                tmode,
                target_speed_val,
                target_pace_min if target_pace_min is not None else '',
                ACTION_CHINESE.get(rule_action, rule_action),
                final_action_ch,
                final_action_src,
                round(suggested_speed,3) if suggested_speed is not None else '',
                suggested_pace_min if suggested_pace_min is not None else '',
                ACTION_CHINESE.get(model_action, model_action),
                model_conf,
                seq_mode,
                seq_conf,
                round(fatigue,3) if fatigue is not None else '',
                CHINESE_FATIGUE.get(fatigue_lvl, fatigue_lvl)
            ])
        prev_hr = avg_hr if avg_hr is not None else prev_hr
        prev_speed = speeds if speeds is not None else prev_speed

# ========== 主程序 ==========

def kafka_consumer_thread(bootstrap_servers='localhost:9092', topic='sensor_data'):
    try:
        from kafka import KafkaConsumer
    except Exception:
        print('⚠️ kafka-python 未安装或无法导入，请运行: pip install kafka-python')
        stop_event.set()
        return
    try:
        consumer = KafkaConsumer(topic,
                                 bootstrap_servers=bootstrap_servers,
                                 auto_offset_reset='latest',
                                 value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                                 consumer_timeout_ms=1000)
        print(f"✅ 已连接到 Kafka ({bootstrap_servers}), 订阅 topic {topic}")
        while not stop_event.is_set():
            for msg in consumer:
                if stop_event.is_set():
                    break
                payload = msg.value
                timestamp_ms = payload.get('timestamp_ms') if isinstance(payload, dict) and 'timestamp_ms' in payload else int(time.time()*1000)
                # 可选：打印接收到的原始 Kafka 消息
                try:
                    if LOG_MSGS:
                        try:
                            print(f"📥 Kafka @{timestamp_ms}: {json.dumps(payload, ensure_ascii=False)[:200]}")
                        except Exception:
                            print(f"📥 Kafka @{timestamp_ms}: {str(payload)[:200]}")
                except Exception:
                    pass
                # use mqtt-like topic name for downstream processing
                topic_name = 'sensor/data'
                try:
                    msg_queue.put_nowait((topic_name, payload, int(timestamp_ms)))
                except Exception:
                    print("⚠️ 队列已满，已丢弃 kafka 消息")
            time.sleep(0.1)
    except Exception as e:
        print('⚠️ Kafka consumer 错误：', e)
    finally:
        try:
            consumer.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', choices=['mqtt','kafka'], default='mqtt', help='消息源: mqtt 或 kafka')
    p.add_argument('--kafka-bootstrap', default='localhost:9092', help='Kafka bootstrap servers')
    p.add_argument('--kafka-topic', default='sensor_data', help='Kafka topic to subscribe')
    p.add_argument('--target-mode', choices=['walk','jog','sprint'], required=True, help='必须显式指定目标模式：walk/jog/sprint')
    p.add_argument('--window', type=int, default=5, help='窗口大小（秒）')
    p.add_argument('--acc-delta', type=float, default=0.3, help='规则建议中的加速增量 (m/s)')
    p.add_argument('--dec-delta', type=float, default=0.5, help='规则建议中的减速减量 (m/s)')
    p.add_argument('--max-pace-diff-sec', type=float, default=20.0, help='单次建议与当前配速的最大差值（秒/公里）')
    p.add_argument('--smooth-alpha', type=float, default=0.65, help='建议平滑系数（0..1）; 1 表示不平滑')
    p.add_argument('--model-conf-high', type=float, default=0.95, help='模型置信度高阈值，超过则优先采信模型')
    p.add_argument('--require-consecutive', type=int, default=2, help='需要多少个连续窗口相同模型动作才认为稳定')
    p.add_argument('--debug-action-log', action='store_true', help='打印每个窗口的决策内部变量以便调试')
    p.add_argument('--log-msgs', action='store_true', help='打印接收到的原始消息（用于调试）')
    args = p.parse_args()
    # apply runtime debug flag for printing messages
    LOG_MSGS = True if args.log_msgs else LOG_MSGS

    kafka_thread = None
    if args.source == 'mqtt':
        client = mqtt.Client(client_id="mqtt_subscriber")
        client.on_connect = on_connect
        client.on_message = on_message
        print("🔄 连接到 MQTT Broker...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    else:
        client = None
        kafka_thread = threading.Thread(target=kafka_consumer_thread, args=(args.kafka_bootstrap, args.kafka_topic), daemon=True)
        kafka_thread.start()

    # 启动后台分析线程（传入 acc/dec delta 及平滑/置信度参数）
    t = threading.Thread(target=analysis_worker, args=(args.window, args.target_mode, args.acc_delta, args.dec_delta, args.max_pace_diff_sec, args.smooth_alpha, args.model_conf_high, args.require_consecutive, args.debug_action_log), daemon=True)
    t.start()

    source_desc = args.source
    print(f"🚀 开始监听消息（source={source_desc}）... 按 Ctrl+C 停止（target_mode={args.target_mode}, window={args.window}s）\n")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n⏹️ 收到停止信号，正在退出...")
        stop_event.set()
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()
        if kafka_thread is not None:
            kafka_thread.join(timeout=2)
        t.join(timeout=2)
        print("✅ 已退出。")