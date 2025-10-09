import os
import cv2
import time
import random
import multiprocessing as mp
import numpy as np
import queue
from hailo_platform import (
    HEF,
    ConfigureParams,
    Device,
    FormatType,
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

# ---------------- 全局参数 ----------------
MAX_PENDING = 2        # 每个设备最多缓冲任务数
TARGET_FPS = 20        # 目标帧率
HEF_PATH = "/home/firefly/Denoising-rk3588J/models/dncnn_4split/dncnn_4split_16pad.hef"
NUM_DEVICES = 2        # 启用设备数量

# 摄像头参数
CAMERA_DEVICE_PATH = "/dev/video20"  # 摄像头设备路径（根据实际情况修改）
TARGET_RESOLUTION = (960, 720)       # 目标分辨率 (width, height)
TARGET_FPS = 20                      # 目标帧率
VIDEO_FORMAT = cv2.VideoWriter_fourcc(*"MJPG")  # 摄像头格式（MJPG支持高帧率）


# -------------------------- 设备初始化（复用原推理逻辑） --------------------------
def init_device(hef_path, device_id):
    """初始化单个Hailo设备并加载模型，与原推理代码逻辑一致"""
    device_ids = Device.scan()
    if len(device_ids) <= device_id:
        raise RuntimeError(f"设备ID {device_id} 不存在，仅检测到 {len(device_ids)} 个设备")
    
    print(f"初始化设备 {device_id}（硬件ID: {device_ids[device_id]}）...")
    
    # 创建设备参数（PCIe接口）
    vdevice_params = VDevice.create_params()
    vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
    vdevice_params.device_ids.append(device_id)
    target = VDevice(params=vdevice_params)
    
    # 加载HEF模型
    hef = HEF(hef_path)
    
    # 配置网络组
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()
    
    # 创建输入/输出流参数（输入float32，输出uint8）
    input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
    
    # 获取流信息
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    
    device_info = {
        "target": target,
        "hef": hef,
        "network_group": network_group,
        "network_group_params": network_group_params,
        "input_vstreams_params": input_vstreams_params,
        "output_vstreams_params": output_vstreams_params,
        "input_vstream_info": input_vstream_info,
        "output_vstream_info": output_vstream_info,
        "device_id": device_id
    }
    
    print(f"设备 {device_id} 初始化完成 | 输入形状: {input_vstream_info.shape} | 输出形状: {output_vstream_info.shape}")
    return device_info

def process_frame(frame):
    """
    处理摄像头BGR帧为模型输入格式（RGB+CHW+float32）
    :param frame: cv2读取的BGR帧（HWC格式）
    :return: 预处理后的数据（CHW格式）、预处理耗时、调整后原始BGR帧（用于拼接）
    """
    start_time = time.time()
    
    # 1. 调整分辨率（确保与目标一致，后续拼接时尺寸统一）
    frame_resized = cv2.resize(
        frame, 
        dsize=TARGET_RESOLUTION, 
        interpolation=cv2.INTER_LANCZOS4  # 高质量插值（与原代码PIL.LANCZOS对应）
    )
    
    # 2. BGR转RGB（cv2默认BGR，模型需要RGB）
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # 3. 格式转换：HWC -> CHW， dtype -> float32
    # frame_chw = frame_rgb.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
    frame_float = frame_rgb.astype(np.float32)
    
    process_time = time.time() - start_time
    return frame_float, process_time


# ---------------- 工作者逻辑 ----------------
def device_worker(device_id, task_queue, queue2, result_queue, hef_path):
    """
    每个Hailo设备的推理进程，优先处理自己的队列task_queue，
    若空则尝试从queue2中偷取任务。
    """
    device = init_device(hef_path, device_id)
    print(f"[Device {device_id}] worker started, PID={os.getpid()}")
    while True:
        try:
            try:
                task = task_queue.get(timeout=0.05)
            except queue.Empty:
                try:
                    # 从queue2偷取任务（负载均衡）
                    task = queue2.get(timeout=0.05)
                except queue.Empty:
                    continue
            if task is None:
                break
            
            frame_id, frame = task
            frame = np.expand_dims(frame, axis=0)

            frame = split_and_stack(frame)
            output, infer_time = run_inference(device, frame)
            output = stack_to_original(output)
            result_queue.put((frame_id, output, infer_time))

        except Exception as e:
            print(f"[Device {device_id}] Error: {e}")
            continue
    device["target"].release()
    print(f"[Device {device_id}] exited.")

def run_inference(device, input_batch):
    """在单个设备上运行推理，返回推理结果与耗时"""
    network_group = device["network_group"]
    input_vstreams_params = device["input_vstreams_params"]
    output_vstreams_params = device["output_vstreams_params"]
    network_group_params = device["network_group_params"]
    input_vstream_info = device["input_vstream_info"]

    start_time = time.time()
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            input_data = {input_vstream_info.name: input_batch}
            infer_results = infer_pipeline.infer(input_data)
    
    inference_time = time.time() - start_time
    output_tensor = infer_results[device["output_vstream_info"].name]

    return output_tensor, inference_time

def split_and_stack(batch_tensor):
    """
    将形状为[1, 720, 960, 3]的图像数组分割为4张子图并堆叠为[4, 360, 480, 3]
    
    参数:
        batch_tensor: 形状为[1, 720, 960, 3]的numpy数组
        
    返回:
        形状为[4, 360, 480, 3]的numpy数组
    """
    # 检查输入形状是否正确
    if batch_tensor.shape != (1, 720, 960, 3):
        raise ValueError(f"输入数组形状必须为[1, 720, 960, 3], 实际输入数组形状为{batch_tensor.shape}")
    
    # 移除批次维度，得到[720, 960, 3]
    image = batch_tensor[0]
    
    # 计算子图的高度和宽度
    sub_height = 720 // 2
    sub_width = 960 // 2
    
    # 分割为4个子图
    # 左上角
    sub1 = image[:sub_height+16, :sub_width+16, :]
    # 左下角
    sub2 = image[sub_height-16:, :sub_width+16, :]
    # 右上角
    sub3 = image[:sub_height+16, sub_width-16:, :]
    # 右下角
    sub4 = image[sub_height-16:, sub_width-16:, :]
    
    # 堆叠成[4, 360, 480, 3]的数组
    stacked = np.stack([sub1, sub2, sub3, sub4], axis=0)
    
    return stacked

def stack_to_original(sub_images):
    """
    将形状为[4, 360, 480, 3]的子图数组拼接为原始图像[1, 720, 960, 3]
    
    参数:
        sub_images: 形状为[4, 360, 480, 3]的numpy数组，包含4个子图
                    顺序应为[左上, 左下, 右上, 右下]
        
    返回:
        形状为[1, 720, 960, 3]的numpy数组，原始图像
    """
    # 检查输入形状是否正确
    if sub_images.shape != (4, 376, 496, 3):
        raise ValueError(f"输入数组形状必须为[4, 360, 480, 3], 实际形状为{sub_images.shape}")
    
    # 提取4个子图
    sub1, sub2, sub3, sub4 = sub_images[0], sub_images[1], sub_images[2], sub_images[3]
    
    # 水平拼接第一行（上半部分）
    top_row = np.concatenate([sub1[:360,:480,:], sub3[:360,16:]], axis=1)
    
    # 水平拼接第二行（下半部分）
    bottom_row = np.concatenate([sub2[16:,:480,:], sub4[16:, 16:,:]], axis=1)
    
    # 垂直拼接两行，得到完整图像
    full_image = np.concatenate([top_row, bottom_row], axis=0)
    
    # 添加批次维度，形状变为[1, 720, 960, 3]
    return np.expand_dims(full_image, axis=0)

# ---------------- 主循环 ----------------
def main():
    cap = cv2.VideoCapture("/dev/video20")
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头设备: {CAMERA_DEVICE_PATH}")
    
    # 设置摄像头参数（MJPG格式支持高帧率）
    cap.set(cv2.CAP_PROP_FOURCC, VIDEO_FORMAT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    # 验证参数是否设置成功（部分摄像头可能不支持目标配置）
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    drop_prob = max(0, 1 - TARGET_FPS / actual_fps) if actual_fps > 0 else 0
    print(f"🎥 Target {TARGET_FPS} fps | Actual {actual_fps:.2f} fps | Drop Prob = {drop_prob:.2f}")

    # 初始化队列与进程
    queues = [mp.Queue(MAX_PENDING) for _ in range(NUM_DEVICES)]  # q0/q1
    queue2 = mp.Queue()   # 备用队列
    result_queue = mp.Queue()

    procs = []
    for i in range(NUM_DEVICES):
        p = mp.Process(target=device_worker, args=(i, queues[i], queue2, result_queue, HEF_PATH))
        p.start()
        procs.append(p)

    frame_id = 0
    last_show_time = 0
    next_display_id = 1
    result_buffer = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 随机丢帧：控制帧率
            if random.random() < drop_prob:
                continue

            frame_id += 1
            processed_frame, _ = process_frame(frame)

            # 分配队列（负载均衡 + 限流）
            assigned = False
            for q in queues:
                if q.qsize() < MAX_PENDING:
                    q.put((frame_id, processed_frame))
                    assigned = True
                    break
            if not assigned:
                queue2.put((frame_id, processed_frame))  # 所有都满了放queue2
                

            # 非阻塞显示最近的结果
            try:
                while True:
                    fid, output_tensor, infer_time = result_queue.get(block=False)
                    frame_to_show = output_tensor[0]
                    if frame_to_show.dtype != np.uint8:
                        frame_to_show = np.clip(frame_to_show, 0, 255).astype(np.uint8)
                  
                    if next_display_id == fid:
                        cv2.imshow("Denoised Stream", frame_to_show)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt
                        next_display_id += 1
                    else:
                        result_buffer[fid] = frame_to_show
                    
                    while next_display_id in result_buffer:
                        frame_to_show = result_buffer.pop(next_display_id)
                        next_display_id += 1
                        cv2.imshow("Denoised Stream", frame_to_show)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

            except queue.Empty:
                pass
            
            

    except KeyboardInterrupt:
        print("\n🛑 User interrupt, shutting down.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # 发送停止信号
        for q in queues:
            q.put(None)
        for p in procs:
            p.join()
        print("✅ All workers terminated.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
