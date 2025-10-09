import cv2

def play_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 创建窗口
    cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 如果读取失败（通常是视频结束），则重新开始播放
        if not ret:
            # 将视频指针重置到开头
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # 显示帧
        cv2.imshow('Video Player', frame)
        
        # 等待按键，按'q'键退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 视频文件路径
    video_path = "/home/firefly/Denoising-rk3588J/output/inference_videos/serial_infer_result_20250928_160541_960x720.mp4"
    # 播放视频
    play_video(video_path)
    
