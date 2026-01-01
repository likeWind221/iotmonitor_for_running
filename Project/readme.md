1.test1.py是esp32的程序，注意要改esp32的内存配置，设置为custom

2.okcomputer.py是从mqtt订阅心率、血氧、里程数和两个角度到数据库的程序。数据库分别是mpu_data.db和sensor_data.db，位于C:\Users\sxhzx下面

3.phone_app.py模拟手机app程序，选择跑步模式：减脂跑、冲刺跑、耐力跑，并会显示实时心率、血氧、速率、里程数、步数、推荐配速等信息。这个程序发布跑步模式消息并订阅推荐配速消息。这个程序会存储本次跑步的历史信息，在数据库running_data.db，位于C:\Users\sxhzx下面



开启mqtt平台：D:\Users\zhu\Projects\Iot\emqx-5.3.2-windows-amd64\bin> ./emqx start

启动顺序  1、3、2





测试用

4.speed_command.py是我模拟的算法要用到的接口，就是订阅跑步模式和发布推荐配速。

5.read_sensorDB.py测试实时读取数据库

6.read_mpuDB.py测试实时读取数据库





启动顺序  1、3、2、5/6

后面iot文件夹下的oled文件夹是esp32订阅配速推荐，当配速改变时现在会打印在串口，之后也可以用来灯双闪一下（已实现）

"D:\Users\zhu\Projects\Iot\test_oled\oled\oled.ino"


