import cv2

gst = (
    "v4l2src device=/dev/video11 ! "
    "video/x-raw,format=NV12,width=800,height=600 ! "
    "videoconvert ! "
    "appsink"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
ret, frame = cap.read()
print("ret:", ret, frame.shape if ret else None)

if ret:
    cv2.imwrite("isp_gst.jpg", frame)

cap.release()
