# qt 环境配置
1. 在 linaro 的桌面终端里放行 firefly
在 linaro 的桌面终端（不要 sudo）里。直接执行：
```bash
echo $DISPLAY                 # 期望是 :0
xhost +SI:localuser:firefly   # 放行 firefly 访问 :0
```

2. 然后切回 firefly 终端运行
```bash
export DISPLAY=:0
export QT_PLUGIN_PATH="/usr/lib/aarch64-linux-gnu/qt5/plugins"
export QT_QPA_PLATFORM=xcb
```