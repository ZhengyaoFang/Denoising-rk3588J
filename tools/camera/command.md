# 记录摄像头相关命令行及测试脚本
1. 检查摄像头是否连接以及设备信息

    - ```lsusb```
    显示：```Bus 001 Device 003: ID 174f:2411 Syntek HD WebCam```说明连接成功

2. 查看摄像头画面

    **环境配置**
    - 安装v4l-utils: ```sudo apt install v4l-utils```

    - 板子安装视频流服务工具：```sudo apt install motion```

    - 配置motion：```sudo vim /etc/motion/motion.conf```

        确保```stream_localhost off```（允许远程访问），```stream_port 8081```（设置端口）。摄像头设备路径（改为你的实际设备）```video_device /dev/video20```

    - 启动服务：```sudo systemctl start motion``` (或重启服务 ```sudo systemctl restart motion```)

    - 停止服务： ```sudo systemctl stop motion```

    **画面访问**
    - 在本地浏览器访问：```http://远程Ubuntu的IP:8081```

3. 摄像头控制命令

    - 查看设备路径及对应的摄像头信息: ```sudo v4l2-ctl --list-devices```

    - 查看摄像头支持的格式和分辨率: ```sudo v4l2-ctl -d /dev/video21 --list-formats-ext```

    - 用v4l2-ctl直接捕获一帧图像: ```sudo v4l2-ctl -d /dev/video21 --stream-mmap --stream-count=1 --stream-to=test_frame.jpg```

    - 查看设备占用信息：```sudo lsof | grep /dev/video20```
    
    



    