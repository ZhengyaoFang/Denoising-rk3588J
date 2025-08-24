import cv2
import time
import argparse

def get_camera_resolutions(cap):
    """获取摄像头支持的所有分辨率"""
    resolutions = []
    # 尝试常见的分辨率，从高到低
    common_resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 2K
        (1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (1024, 768),
        (800, 600),
        (640, 480),    # VGA
        (320, 240)     # QVGA
    ]
    
    # 检查每个分辨率是否被支持
    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 读取实际设置的分辨率
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if (actual_width, actual_height) == (width, height):
            resolutions.append((width, height))
    
    return resolutions

def test_max_fps(cap, width, height, test_duration=5):
    """测试特定分辨率下的最大帧率"""
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 确保分辨率设置正确
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (width, height):
        return 0.0
    
    print(f"测试分辨率: {width}x{height}...")
    
    # 开始测试
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    return fps

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试摄像头支持的最大分辨率和帧率')
    parser.add_argument('--device', type=int, default=20, help='摄像头设备编号，默认为0')
    args = parser.parse_args()
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"无法打开摄像头设备 {args.device}")
        return
    
    print(f"成功打开摄像头设备 {args.device}")
    print("正在检测支持的分辨率...")
    
    # 获取所有支持的分辨率
    resolutions = get_camera_resolutions(cap)
    
    if not resolutions:
        print("未检测到任何支持的分辨率")
        cap.release()
        return
    
    print(f"检测到 {len(resolutions)} 种支持的分辨率:")
    for res in resolutions:
        print(f"  {res[0]}x{res[1]}")
    
    # 测试每种分辨率的帧率
    print("\n正在测试各分辨率下的帧率...")
    results = []
    
    for width, height in resolutions:
        fps = test_max_fps(cap, width, height)
        results.append((width, height, fps))
        print(f"  {width}x{height}: {fps:.2f} FPS")
    
    # 找出最大分辨率
    max_res = max(resolutions, key=lambda x: x[0] * x[1])
    max_res_fps = next(fps for w, h, fps in results if (w, h) == max_res)
    
    # 找出最高帧率
    max_fps_entry = max(results, key=lambda x: x[2])
    
    print("\n测试结果总结:")
    print(f"最大分辨率: {max_res[0]}x{max_res[1]} ({max_res_fps:.2f} FPS)")
    print(f"最高帧率: {max_fps_entry[2]:.2f} FPS ({max_fps_entry[0]}x{max_fps_entry[1]})")
    
    # 释放资源
    cap.release()
    print("\n测试完成")

if __name__ == "__main__":
    main()
