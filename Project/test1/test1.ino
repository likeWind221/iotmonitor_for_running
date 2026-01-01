#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <HardwareSerial.h>
#include "BluetoothSerial.h"

// ==================== 硬件引脚定义 ====================
#define HR_SENSOR_RX_PIN    16  
#define HR_SENSOR_TX_PIN    17  

// ==================== 心率传感器配置 ====================
#define HR_FRAME_LEN   50
#define HR_OFFSET      40
#define SPO2_OFFSET    41
HardwareSerial HrSerial(2);  
uint8_t hrBuffer[HR_FRAME_LEN];
uint8_t hrIdx = 0;
bool hrDataValid = false;
uint8_t heartRate = 0, spo2 = 0;

// ==================== 蓝牙配置 ====================
BluetoothSerial SerialBT;
bool isSampling = false;     
String btCmdBuffer = "";

// ==================== WiFi/MQTT配置 ====================
const char *ssid       = "Xiaomi 13";
const char *password   = "5151336173";
const char *mqttServer = "192.168.186.210";
const int   mqttPort   = 1883;
const char *mpuTopic   = "sensor/mpu";
const char *dataTopic  = "sensor/data";

WiFiClient espClient;
PubSubClient client(espClient);

// ==================== 传感器对象 ====================
Adafruit_MPU6050 mpu;
bool mpuInitOk = false; // MPU初始化状态标记（核心修复）

// ==================== MPU6050参数 ====================
float ax_offset = 0, ay_offset = 0, gx_offset = 0, gy_offset = 0;
float rad2deg = 57.29578;
float angleX = 0, angleY = 0;
unsigned long lastMPUUpdate = 0;
unsigned long lastDataUpdate = 0;
int mpuUploadCount = 0;  
int dataUploadCount = 0;

// ==================== 新增：里程/步数计算配置（适配手腕）====================
const float STEP_LENGTH = 0.75;    // 步长（米/步，可根据身高调整：身高*0.45~0.5）
const float ANGLE_Y_THRESHOLD = 8.0; // 手腕上下摆动角度阈值（°）
const float ANGLE_CHANGE_THRESHOLD = 3.0; // 角度变化量阈值（过滤微小抖动）
const unsigned long STEP_COOLDOWN = 300; // 步频冷却时间（ms，避免重复计数）

float totalMileage = 0.0;          // 总里程（米）
int stepCount = 0;                 // 总步数
unsigned long lastStepTime = 0;    // 上一次计步时间
float lastAngleY = 0.0;            // Y轴角度历史值（检测摆动）
bool isAngleYUp = false;           // Y轴角度上升/下降标记
bool isStepDetected = false;       // 步检测标记

// ==================== 函数声明 ====================
void setupWiFi();
void connectMQTT();
void initSensors();
void readHRData();
void readMPUData();
void handleBluetoothCmd();
void sendBluetoothData();
void publishMPU();
void publishData();
String getFormattedTime(); 
void calculateStepAndMileage();    // 新增：里程/步数计算

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("=== ESP32核心功能稳定版（无GPS）===");
  
  SerialBT.begin("RunningMonitor");
  Serial.println("蓝牙设备名：RunningMonitor");
  
  Wire.begin(21, 22);
  Wire.setClock(400000);
  
  setupWiFi();
  client.setServer(mqttServer, mqttPort);
  connectMQTT();
  
  initSensors();
  
  // 初始化角度历史值
  lastAngleY = angleY;
  
  Serial.println("系统初始化完成，等待蓝牙start指令...");
}

void loop() {
  static unsigned long lastFeed = 0;
  if (millis() - lastFeed > 50) {
    lastFeed = millis();
  }
  
  handleBluetoothCmd();
  
  if (isSampling) {
    // MPU：严格100ms/次上传（核心：无阻塞）
    if (millis() - lastMPUUpdate > 100) {
      readMPUData();
      calculateStepAndMileage(); // 新增：每次读取MPU后计算步数/里程
      publishMPU();
      lastMPUUpdate = millis();
      mpuUploadCount++;
    }
    
    // 心率读取（非阻塞）
    readHRData();
    
    // 全量数据：1000ms/次上传（包含里程）
    if (millis() - lastDataUpdate > 1000) {
      sendBluetoothData();       // 蓝牙1s上传：心率+里程+步数
      publishData();             // MQTT 1s上传：心率+角度+里程+步数
      lastDataUpdate = millis();
      dataUploadCount++;
      Serial.printf("📊 频率验证：MPU上传%d次，全量数据上传%d次（理论10:1），里程=%.2f米，步数=%d\n", 
                    mpuUploadCount, dataUploadCount, totalMileage, stepCount);
    }
    
    // MQTT重连（非阻塞）
    if (!client.connected()) {
      static unsigned long lastReconnect = 0;
      if (millis() - lastReconnect > 5000) {
        connectMQTT();
        lastReconnect = millis();
      }
    } else {
      client.loop();
    }
  } else {
    static unsigned long lastStatus = 0;
    if (millis() - lastStatus > 5000) {
      Serial.println("🔴 等待蓝牙发送start启动采样");
      lastStatus = millis();
    }
  }
  
  delayMicroseconds(50);
}

// -------------------- 新增：里程/步数计算（适配手腕佩戴）--------------------
void calculateStepAndMileage() {
  if (!mpuInitOk) return;
  
  // 计算Y轴角度变化量（手腕上下摆动核心）
  float deltaAngleY = abs(angleY - lastAngleY);
  bool currentAngleYUp = (angleY > lastAngleY);
  
  // 检测有效摆动：角度超过阈值 + 变化量足够 + 冷却时间已过
  bool isValidSwing = (abs(angleY) > ANGLE_Y_THRESHOLD) && 
                      (deltaAngleY > ANGLE_CHANGE_THRESHOLD) && 
                      (millis() - lastStepTime) > STEP_COOLDOWN;
  
  // 上升沿触发计步（模拟一步的完整摆动：从下往上）
  if (isValidSwing && currentAngleYUp && !isAngleYUp && !isStepDetected) {
    stepCount++;
    lastStepTime = millis();
    totalMileage = stepCount * STEP_LENGTH; // 步数转里程
    isStepDetected = true; // 避免重复计数
    Serial.printf("👟 计步：%d步，总里程：%.2f米（手腕Y角度：%.1f°）\n", stepCount, totalMileage, angleY);
  }
  
  // 下降沿重置标记（完成一步摆动）
  if (!currentAngleYUp && isAngleYUp) {
    isStepDetected = false;
  }
  
  // 更新历史数据
  isAngleYUp = currentAngleYUp;
  lastAngleY = angleY;
}

// -------------------- 传感器初始化（核心：MPU只初始化1次）--------------------
void initSensors() {
  // 初始化MPU6050（仅1次）
  Serial.println("初始化MPU6050...");
  unsigned long mpuStart = millis();
  while (millis() - mpuStart < 10000) {
    if (mpu.begin()) {
      mpuInitOk = true; // 标记初始化成功
      break;
    }
    delay(1000);
  }
  
  if (mpuInitOk) {
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    
    // 快速校准
    Serial.println("校准MPU6050...");
    for (int i = 0; i < 300; i++) {
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      ax_offset += a.acceleration.x;
      ay_offset += a.acceleration.y;
      gx_offset += g.gyro.x;
      gy_offset += g.gyro.y;
      delayMicroseconds(1000);
    }
    ax_offset /= 300;
    ay_offset /= 300;
    gx_offset /= 300;
    gy_offset /= 300;
    Serial.println("✅ MPU6050初始化完成");
  } else {
    Serial.println("⚠️ MPU6050初始化超时");
  }
  
  // 初始化心率传感器
  Serial.println("初始化心率传感器...");
  HrSerial.begin(115200, SERIAL_8N1, HR_SENSOR_RX_PIN, HR_SENSOR_TX_PIN);
  delay(200);
  
  HrSerial.write(0xFF);
  delay(1500);
  HrSerial.write(0xFF);
  delay(500);
  
  Serial.println("✅ 心率传感器初始化完成");
}

// -------------------- MPU读取（核心修复：移除重复begin()）--------------------
void readMPUData() {
  if (!mpuInitOk) return; // 仅初始化成功后读取
  
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp); // 直接读取，无需重新初始化
  
  float accX = a.acceleration.x - ax_offset;
  float accY = a.acceleration.y - ay_offset;
  float accZ = a.acceleration.z;
  
  angleX = atan2(accY, sqrt(accX*accX + accZ*accZ)) * rad2deg;
  angleY = atan2(-accX, accZ) * rad2deg;
}

// -------------------- 心率读取 --------------------
void readHRData() {
  if (HrSerial.available() > 0) {
    uint8_t b = HrSerial.read();
    
    if (hrIdx == 0 && b != 0xFF) return;
    
    hrBuffer[hrIdx++] = b;
    
    if (hrIdx == HR_FRAME_LEN) {
      heartRate = hrBuffer[HR_OFFSET];
      spo2 = hrBuffer[SPO2_OFFSET];
      
      hrDataValid = (heartRate >= 30 && heartRate <= 250 && spo2 >= 70 && spo2 <= 100);
      if (hrDataValid) {
        Serial.printf("❤️ 心率：%d, 血氧：%d\n", heartRate, spo2);
      }
      
      hrIdx = 0;
    }
  }
}

// -------------------- 格式化时间（和你的订阅脚本对齐）--------------------
String getFormattedTime() {
  unsigned long now = millis();
  unsigned long seconds = now / 1000;
  unsigned long minutes = seconds / 60;
  unsigned long hours = minutes / 60;
  
  // 模拟24小时制时间（可根据实际需求调整，或接入RTC）
  hours = (hours % 24) + 8; // 偏移到北京时间
  minutes = minutes % 60;
  seconds = seconds % 60;
  
  char timeStr[9];
  snprintf(timeStr, sizeof(timeStr), "%02d:%02d:%02d", hours, minutes, seconds);
  return String(timeStr);
}

// -------------------- MPU MQTT发布（带毫秒级时间，0.1s/次，仅角度）--------------------
void publishMPU() {
  if (!mpuInitOk) return;
  
  StaticJsonDocument<150> doc;
  doc["angleX"] = round(angleX * 100.0) / 100.0; // 保留2位小数
  doc["angleY"] = round(angleY * 100.0) / 100.0;
  doc["time"] = getFormattedTime();
  doc["count"] = mpuUploadCount;
  doc["timestamp_ms"] = millis(); // 加入毫秒级时间戳
  
  char buf[150];
  serializeJson(doc, buf);
  
  if (client.connected()) {
    client.publish(mpuTopic, buf);
    // 每100ms打印一次，验证频率
    Serial.printf("[MPU] 角度X: %.2f°, 角度Y: %.2f°, 时间: %s (计数:%d)\n", 
                  angleX, angleY, getFormattedTime().c_str(), mpuUploadCount);
  }
}

// -------------------- 全量数据 MQTT发布（1s/次，包含里程+步数）--------------------
void publishData() {
  StaticJsonDocument<250> doc;
  
  if (hrDataValid) {
    doc["hr"] = heartRate;
    doc["spo2"] = spo2;
  } else {
    doc["hr"] = 0;
    doc["spo2"] = 0;
  }
  doc["angleX"] = round(angleX * 100.0) / 100.0;
  doc["angleY"] = round(angleY * 100.0) / 100.0;
  doc["total_mileage_m"] = round(totalMileage * 100.0) / 100.0; // 新增：里程（米）
  doc["step_count"] = stepCount;                                // 新增：步数
  doc["time"] = getFormattedTime();
  doc["count"] = dataUploadCount;
  
  char buf[250];
  serializeJson(doc, buf);
  
  if (client.connected()) {
    client.publish(dataTopic, buf);
    Serial.printf("[SENSOR] 心率: %.1f, 血氧: %.1f, 里程: %.2f米, 步数: %d, 时间: %s (计数:%d)\n", 
                  (float)heartRate, (float)spo2, totalMileage, stepCount, getFormattedTime().c_str(), dataUploadCount);
  }
}

// -------------------- 蓝牙指令处理 --------------------
void handleBluetoothCmd() {
  while (SerialBT.available() > 0) {
    char c = SerialBT.read();
    
    if (c == '\n' || c == '\r') {
      if (btCmdBuffer.length() > 0) {
        btCmdBuffer.trim();
        Serial.printf("📱 蓝牙指令：%s\n", btCmdBuffer.c_str());
        
        if (btCmdBuffer.equalsIgnoreCase("start")) {
          isSampling = true;
          Serial.println("✅ 启动采样");
          SerialBT.println("ACK:START");
        } else if (btCmdBuffer.equalsIgnoreCase("stop")) {
          isSampling = false;
          Serial.println("❌ 停止采样");
          SerialBT.println("ACK:STOP");
        } else if (btCmdBuffer.equalsIgnoreCase("reset")) { // 新增：重置里程/步数
          totalMileage = 0.0;
          stepCount = 0;
          Serial.println("🔄 重置里程/步数");
          SerialBT.println("ACK:RESET");
        } else {
          SerialBT.println("ERR:UNKNOWN_CMD");
        }
        
        btCmdBuffer = "";
      }
    } else {
      btCmdBuffer += c;
    }
    break;
  }
}

// -------------------- 蓝牙数据发送（1s/次，包含心率+里程+步数，无角度）--------------------
void sendBluetoothData() {
  if (hrDataValid) {
    SerialBT.printf("HR:%d,SPO2:%d,MILEAGE:%.2f,STEPS:%d\n", heartRate, spo2, totalMileage, stepCount);
  } else {
    SerialBT.printf("HR:0,SPO2:0,MILEAGE:%.2f,STEPS:%d\n", totalMileage, stepCount);
  }
}

// -------------------- WiFi/MQTT辅助函数 --------------------
void setupWiFi() {
  Serial.print("连接WiFi...");
  WiFi.begin(ssid, password);
  
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 10000) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi连接成功");
    Serial.println("IP：" + WiFi.localIP().toString());
  } else {
    Serial.println("\n⚠️ WiFi连接超时");
  }
}

void connectMQTT() {
  Serial.print("连接MQTT...");
  if (client.connect("ESP32Sensor")) {
    Serial.println("成功");
  } else {
    Serial.printf("失败 (%d)\n", client.state());
  }
}