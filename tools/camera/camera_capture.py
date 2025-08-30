import cv2
import os
import time
from datetime import datetime

def capture_100_frames(device_path='/dev/video20', save_dir='/home/firefly/Denoising/data/20250826'):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"图像将保存到目录: {save_dir}")

    # 设置摄像头参数
    width, height = 960, 720  # 分辨率设置
    target_fps = 60           # 帧率设置
    
    # 打开摄像头
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头设备: {device_path}")
        return

    try:
        # 配置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # 获取实际设置的参数
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"摄像头配置: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        print(f"开始捕获924张图像...")

        start_time = time.time()
        frame_count = 0
        
        # 捕获100张图像
        while frame_count < 924:
            ret, frame = cap.read()
            if ret:
                # 生成图像文件名 (包含时间戳和帧编号)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 毫秒级时间戳
                filename = f"{save_dir}/frame_{timestamp}_{frame_count:03d}.jpg"
                
                # 保存图像
                cv2.imwrite(filename, frame)
                frame_count += 1
                
                # 显示进度
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"已保存 {frame_count}/924 张图像 ({elapsed:.2f}秒)")
            else:
                print(f"\n⚠️  无法捕获帧 (第 {frame_count+1} 帧)")
                # 尝试重新获取帧，如果连续5次失败则退出
                fail_count = 0
                while fail_count < 5 and not ret:
                    ret, frame = cap.read()
                    fail_count += 1
                if not ret:
                    print("❌ 连续获取帧失败，退出程序")
                    break

        # 计算总耗时和实际帧率
        elapsed_time = time.time() - start_time
        actual_capture_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print("\n" + "="*50)
        print(f"捕获完成: 共保存 {frame_count} 张图像")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        print(f"平均帧率: {actual_capture_fps:.2f} fps")
        print(f"保存路径: {save_dir}")
        print("="*50)

    except Exception as e:
        print(f"处理时出错: {str(e)}")
    
    finally:
        # 释放摄像头资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 可以修改为您的摄像头设备路径
    capture_100_frames('/dev/video21')
