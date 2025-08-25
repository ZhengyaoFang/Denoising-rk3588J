import cv2
import os
import time
from datetime import datetime

def capture_frames(device_path='/dev/video20'):
    # 创建保存图像和视频的目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"output/camera_captures_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"数据将保存到目录: {output_dir}")

    # 定义MJPG格式下的所有配置 (分辨率和帧率)
    configurations = [
        # 960x720 分辨率的各种帧率
        (960, 720, 60),
        (960, 720, 40),
        (960, 720, 30),
        # 640x480 分辨率的各种帧率
        (640, 480, 60),
        (640, 480, 40),
        (640, 480, 30)
    ]
    
    # 存储每个配置的帧率统计结果
    fps_results = []

    # 遍历所有配置并捕获视频
    for idx, (width, height, target_fps) in enumerate(configurations):
        print(f"\n处理配置 {idx+1}/{len(configurations)}: {width}x{height} @ {target_fps}fps")
        
        # 打开摄像头
        cap = cv2.VideoCapture(device_path)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头设备: {device_path}")
            continue

        try:
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # 获取实际设置的参数
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_target_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"设置: {actual_width}x{actual_height} @ {actual_target_fps:.1f}fps (目标: {target_fps}fps)")

            # 定义视频编写器
            video_filename = f"{output_dir}/video_{actual_width}x{actual_height}_{target_fps}fps.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, target_fps, (actual_width, actual_height))

            # 捕获5秒钟的视频
            start_time = time.time()
            frame_count = 0
            duration = 5  # 捕获时长(秒)
            
            print(f"开始捕获 {duration} 秒视频...")
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if ret:
                    # 写入视频
                    out.write(frame)
                    frame_count += 1
                    # 每10帧打印一次进度
                    if frame_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"已捕获 {frame_count} 帧 ({elapsed:.1f}s/{duration}s)", end='\r')
                else:
                    print(f"\n⚠️  无法捕获帧 (第 {frame_count+1} 帧)")
                    break

            # 计算实际帧率
            elapsed_time = time.time() - start_time
            actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 保存最后一帧作为图片
            if frame_count > 0:
                last_frame_filename = f"{output_dir}/last_frame_{actual_width}x{actual_height}_{target_fps}fps.jpg"
                cv2.imwrite(last_frame_filename, frame)
                print(f"\n已保存最后一帧: {last_frame_filename}")

            # 保存视频
            out.release()
            print(f"已保存视频: {video_filename}")
            
            # 记录结果
            fps_results.append({
                'width': actual_width,
                'height': actual_height,
                'target_fps': target_fps,
                'actual_fps': actual_fps,
                'frame_count': frame_count,
                'duration': elapsed_time
            })
            
            print(f"统计: 实际帧率 = {actual_fps:.2f}fps ({frame_count} 帧 / {elapsed_time:.2f} 秒)")

        except Exception as e:
            print(f"处理时出错: {str(e)}")
        
        finally:
            # 释放摄像头资源
            cap.release()

    # 生成并打印汇总报告
    print("\n" + "="*60)
    print("📊 摄像头性能测试汇总报告")
    print("="*60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备路径: {device_path}")
    print(f"保存目录: {output_dir}")
    print("-"*60)
    # 表头设置更精确的宽度
    print(f"{'分辨率':<12} {'目标FPS':<10} {'实际FPS':<10} {'捕获帧数':<10} {'时长(秒)':<10}")
    print("-" * 52)  # 调整分隔线长度匹配表头

    for result in fps_results:
        # 每个字段设置固定宽度，确保对齐
        print(f"{result['width']}x{result['height']:<8} "
            f"{result['target_fps']:<10} "
            f"{result['actual_fps']:10.2f}"
            f"{result['frame_count']:<10} "
            f"{result['duration']:.2f}")
    
    print("="*60)
    print("所有配置的视频捕获完成")

if __name__ == "__main__":
    # 可以修改为您的摄像头设备路径
    capture_frames('/dev/video21')
