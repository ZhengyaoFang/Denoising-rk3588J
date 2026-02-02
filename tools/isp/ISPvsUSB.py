import cv2
import numpy as np

ISP_DEV = "/dev/video11"
USB_DEV = "/dev/video20"

# 打开两个视频源
cap_isp = cv2.VideoCapture(ISP_DEV)
cap_usb = cv2.VideoCapture(USB_DEV)

if not cap_isp.isOpened():
    raise RuntimeError(f"Failed to open ISP device {ISP_DEV}")

if not cap_usb.isOpened():
    raise RuntimeError(f"Failed to open USB device {USB_DEV}")

# 尝试设置分辨率（失败也没关系）
TARGET_W, TARGET_H = 800, 600
cap_isp.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
cap_isp.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

ret_isp, frame_isp = cap_isp.read()
ret_usb, frame_usb = cap_usb.read()

if not ret_isp or not ret_usb:
    print("Frame grab failed")


# 统一尺寸，避免拼接失败
h = min(frame_isp.shape[0], frame_usb.shape[0])
w = min(frame_isp.shape[1], frame_usb.shape[1])

frame_isp = cv2.resize(frame_isp, (w, h))
frame_usb = cv2.resize(frame_usb, (w, h))

# 标注来源
cv2.putText(
    frame_isp, "ISP /dev/video11",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0, (0, 255, 0), 2
)
cv2.putText(
    frame_usb, "USB /dev/video20",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0, (0, 255, 0), 2
)

# 左右拼接
combined = np.hstack([frame_isp, frame_usb])
cv2.imwrite("ISPvsUSB.jpg", combined)

cap_isp.release()
cap_usb.release()
cv2.destroyAllWindows()
