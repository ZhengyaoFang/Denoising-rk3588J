sudo gst-launch-1.0 \
  v4l2src device=/dev/video11 io-mode=mmap ! \
  video/x-raw,format=NV12,width=800,height=600 ! \
  fakesink
