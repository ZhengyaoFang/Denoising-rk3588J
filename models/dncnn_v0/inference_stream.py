import os
import time
import cv2
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
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

# -------------------------- 核心配置参数 --------------------------
# 摄像头参数
CAMERA_DEVICE_PATH = "/dev/video21"  # 摄像头设备路径（根据实际情况修改）
TARGET_RESOLUTION = (960, 720)       # 目标分辨率 (width, height)
TARGET_FPS = 60                      # 目标帧率
VIDEO_FORMAT = cv2.VideoWriter_fourcc(*"MJPG")  # 摄像头格式（MJPG支持高帧率）

# 推理参数
HEF_PATH = "/home/firefly/Denoising-rk3588J/models/dncnn_v0/dncnn_bs3.hef"  # Hailo模型路径
BATCH_SIZE = 2                        # 单设备批次大小（平衡实时性与效率）
INPUT_SHAPE = (3, 720, 960)          # 模型输入形状 (channel, height, width)
NUM_DEVICES = 2                       # 启用的Hailo加速棒数量
QUEUE_MAX_SIZE = 50                   # 任务队列最大缓存（避免帧堆积）
RUN_DURATION = 10                     # 测试运行时长（秒，可修改）

# -------------------------- 帧预处理（适配摄像头输入） --------------------------
def process_frame(frame):
    """
    处理摄像头BGR帧为模型输入格式（RGB+CHW+float32）
    :param frame: cv2读取的BGR帧（HWC格式）
    :return: 预处理后的数据（CHW格式）、预处理耗时
    """
    start_time = time.time()
    
    # 1. BGR转RGB（cv2默认BGR，模型需要RGB）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. 调整分辨率（确保与目标一致，防止摄像头参数设置失效）
    frame_resized = cv2.resize(
        frame_rgb, 
        dsize=TARGET_RESOLUTION, 
        interpolation=cv2.INTER_LANCZOS4  # 高质量插值（与原代码PIL.LANCZOS对应）
    )
    
    # 3. 格式转换：HWC -> CHW， dtype -> float32
    frame_chw = frame_resized.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
    frame_float = frame_chw.astype(np.float32)
    
    process_time = time.time() - start_time
    return frame_float, process_time

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

# -------------------------- 推理函数（复用原逻辑） --------------------------
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

# -------------------------- 工作进程（移除图像保存，专注推理） --------------------------
def worker_process(device_id, task_queue, result_queue, hef_path):
    """
    设备工作进程：接收摄像头帧批次 → 推理 → 返回结果（无图像保存）
    :param task_queue: 任务队列（元素：(batch_tensor, actual_batch_size, batch_index)）
    :param result_queue: 结果队列（元素：(batch_index, actual_batch_size, infer_time)）
    """
    try:
        device = init_device(hef_path, device_id)
        print(f"设备 {device_id} 工作进程启动（PID: {os.getpid()}）")
        
        while True:
            task = task_queue.get()
            if task is None:  # 终止信号
                break
            
            batch_tensor, actual_batch_size, batch_index = task
            # 执行推理（忽略输出结果，仅统计时间）
            _, infer_time = run_inference(device, batch_tensor)
            # 向主进程返回统计信息（批次索引、实际有效帧数、推理耗时）
            result_queue.put((batch_index, actual_batch_size, infer_time))
    
    except Exception as e:
        print(f"设备 {device_id} 工作进程出错: {str(e)}")
    finally:
        # 释放设备资源
        if "device" in locals():
            device["target"].release()
        print(f"设备 {device_id} 工作进程退出")

# -------------------------- 主流程（摄像头读取+实时推理+FPS统计） --------------------------
def main():
    total_start_time = time.time()
    print("="*60)
    print(f"开始实时推理测试 | 目标: {TARGET_RESOLUTION[0]}×{TARGET_RESOLUTION[1]} @ {TARGET_FPS}fps | 运行时长: {RUN_DURATION}s")
    print(f"设备数量: {NUM_DEVICES} | 批次大小: {BATCH_SIZE} | 模型路径: {HEF_PATH}")
    print("="*60)

    # -------------------------- 1. 初始化进程与队列 --------------------------
    # 任务队列（每个设备一个）：缓存待推理的帧批次
    task_queues = [mp.Queue(maxsize=QUEUE_MAX_SIZE) for _ in range(NUM_DEVICES)]
    # 结果队列：接收推理统计信息
    result_queue = mp.Queue()
    
    # 启动工作进程
    processes = []
    for device_id in range(NUM_DEVICES):
        p = mp.Process(
            target=worker_process,
            args=(device_id, task_queues[device_id], result_queue, HEF_PATH)
        )
        p.start()
        processes.append(p)
    time.sleep(2)  # 等待设备初始化完成

    # -------------------------- 2. 初始化摄像头 --------------------------
    cap = cv2.VideoCapture(CAMERA_DEVICE_PATH)
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
    print(f"\n摄像头参数验证 | 实际分辨率: {actual_width}×{actual_height} | 实际帧率: {actual_fps:.1f}fps")
    if (actual_width, actual_height) != TARGET_RESOLUTION:
        print(f"⚠️  摄像头不支持 {TARGET_RESOLUTION[0]}×{TARGET_RESOLUTION[1]}，将使用实际分辨率 {actual_width}×{actual_height}")

    # -------------------------- 3. 初始化统计变量 --------------------------
    read_total_frames = 0          # 摄像头读取的总帧数
    infer_total_frames = 0         # 成功推理的总帧数
    read_start_time = time.time()  # 摄像头读取开始时间
    infer_start_time = None        # 推理开始时间（第一个批次发送时记录）
    total_infer_compute_time = 0   # 推理计算总耗时（所有设备累加）
    batch_index = 0                # 批次索引（用于匹配任务与结果）
    device_buffers = [[] for _ in range(NUM_DEVICES)]  # 各设备的帧缓存（积累批次）

    # -------------------------- 4. 实时读取+推理循环 --------------------------
    print(f"\n开始读取摄像头帧（按 'q' 提前退出）...")
    while (time.time() - read_start_time) < RUN_DURATION:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("⚠️  无法读取摄像头帧，退出循环")
            break
        
        read_total_frames += 1
        current_time = time.time()

        # 预处理帧（BGR→RGB→CHW→float32）
        frame_processed, _ = process_frame(frame)

        # 分配帧到设备缓存（轮询分配，均衡负载）
        target_device_id = read_total_frames % NUM_DEVICES
        device_buffers[target_device_id].append(frame_processed)

        # 当缓存达到批次大小时，发送推理任务
        if len(device_buffers[target_device_id]) >= BATCH_SIZE:
            # 提取批次帧并补零（不足批次大小时）
            batch_frames = device_buffers[target_device_id][:BATCH_SIZE]
            actual_batch_size = len(batch_frames)
            if actual_batch_size < BATCH_SIZE:
                pad_size = BATCH_SIZE - actual_batch_size
                batch_frames += [np.zeros_like(batch_frames[0]) for _ in range(pad_size)]
            
            # 转换为批次张量（shape: [batch_size, C, H, W]）
            batch_tensor = np.stack(batch_frames, axis=0)

            # 发送任务到队列（避免队列满导致阻塞）
            try:
                task_queues[target_device_id].put(
                    (batch_tensor, actual_batch_size, batch_index),
                    block=False  # 非阻塞模式，队列满时丢弃
                )
                batch_index += 1
                # 记录推理开始时间（第一个任务发送时）
                if infer_start_time is None:
                    infer_start_time = current_time
                print(f"📤 设备 {target_device_id} 发送批次 {batch_index-1}（有效帧: {actual_batch_size}）", end="\r")
            except mp.Queue.Full:
                print(f"⚠️  设备 {target_device_id} 任务队列已满，丢弃当前批次", end="\r")

            # 清空已发送的缓存
            device_buffers[target_device_id] = device_buffers[target_device_id][BATCH_SIZE:]

        # 按 'q' 键提前退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n🛑 用户按下 'q' 键，提前退出")
            break

    # -------------------------- 5. 处理剩余帧（缓存中未发送的帧） --------------------------
    print(f"\n\n处理缓存中剩余的帧...")
    for device_id in range(NUM_DEVICES):
        remaining_frames = device_buffers[device_id]
        if len(remaining_frames) == 0:
            continue
        
        # 处理剩余帧（不足批次大小时补零）
        actual_batch_size = len(remaining_frames)
        if actual_batch_size < BATCH_SIZE:
            pad_size = BATCH_SIZE - actual_batch_size
            remaining_frames += [np.zeros_like(remaining_frames[0]) for _ in range(pad_size)]
        
        batch_tensor = np.stack(remaining_frames, axis=0)
        try:
            task_queues[device_id].put((batch_tensor, actual_batch_size, batch_index), block=True, timeout=5)
            batch_index += 1
            print(f"📤 设备 {device_id} 发送剩余批次 {batch_index-1}（有效帧: {actual_batch_size}）")
        except mp.Queue.Full:
            print(f"⚠️  设备 {device_id} 队列满，无法发送剩余 {actual_batch_size} 帧")

    # -------------------------- 6. 发送终止信号并收集结果 --------------------------
    # 向所有工作进程发送终止信号
    for q in task_queues:
        q.put(None)
    
    # 收集推理结果（进度条显示）
    print(f"\n收集推理结果...")
    processed_batches = 0
    progress_bar = tqdm(total=batch_index, desc="推理进度")
    while processed_batches < batch_index:
        try:
            # 从结果队列获取数据（阻塞等待）
            batch_idx, actual_frames, infer_time = result_queue.get(block=True, timeout=30)
            infer_total_frames += actual_frames
            total_infer_compute_time += infer_time
            processed_batches += 1
            progress_bar.update(1)
        except mp.Queue.Empty:
            print(f"⚠️  结果队列超时，未收集到所有批次结果（已处理 {processed_batches}/{batch_index}）")
            break
    progress_bar.close()

    # -------------------------- 7. 释放资源 --------------------------
    cap.release()
    cv2.destroyAllWindows()
    for p in processes:
        p.join(timeout=10)
        print(f"🔚 设备进程 {p.pid} 退出状态: {'正常' if p.exitcode == 0 else '异常'}")

    # -------------------------- 8. 计算并输出FPS统计 --------------------------
    print("\n" + "="*60)
    print("📊 实时推理FPS统计结果")
    print("="*60)

    # 1. 摄像头读取性能
    read_total_time = time.time() - read_start_time
    read_fps = read_total_frames / read_total_time if read_total_time > 1e-3 else 0.0
    print(f"1. 摄像头读取性能")
    print(f"   - 总读取帧数: {read_total_frames} 帧")
    print(f"   - 读取总时间: {read_total_time:.2f} 秒")
    print(f"   - 实际读取FPS: {read_fps:.2f} fps")

    # 2. 推理性能（仅统计有有效推理的情况）
    if infer_start_time is not None and infer_total_frames > 0:
        infer_total_wall_time = time.time() - infer_start_time  # 推理阶段墙钟时间
        infer_fps = infer_total_frames / infer_total_wall_time  # 实际推理FPS
        device_utilization = (total_infer_compute_time / (NUM_DEVICES * infer_total_wall_time)) * 100  # 设备利用率
        parallel_speedup = (total_infer_compute_time / NUM_DEVICES) / infer_total_wall_time  # 并行加速比

        print(f"\n2. 推理性能（{NUM_DEVICES} 设备并行）")
        print(f"   - 总推理帧数: {infer_total_frames} 帧")
        print(f"   - 推理墙钟时间: {infer_total_wall_time:.2f} 秒")
        print(f"   - 推理计算总耗时（累加）: {total_infer_compute_time:.2f} 秒")
        print(f"   - 实际推理FPS: {infer_fps:.2f} fps")
        print(f"   - 设备平均利用率: {device_utilization:.1f}%")
        print(f"   - 并行加速比: {parallel_speedup:.2f}x")
    else:
        print(f"\n2. 推理性能")
        print(f"   - 未完成有效推理（可能设备初始化失败或无帧输入）")

    # 3. 整体性能瓶颈分析
    print(f"\n3. 瓶颈分析")
    if read_fps < TARGET_FPS * 0.9:
        print(f"   ⚠️  摄像头读取FPS（{read_fps:.2f}）低于目标（{TARGET_FPS}），读取为瓶颈")
    elif infer_start_time is not None and infer_fps < read_fps * 0.9:
        print(f"   ⚠️  推理FPS（{infer_fps:.2f}）低于读取FPS（{read_fps:.2f}），推理为瓶颈")
    else:
        print(f"   ✅ 性能匹配：推理FPS（{infer_fps:.2f}）≥ 读取FPS（{read_fps:.2f}），无明显瓶颈")

    total_time = time.time() - total_start_time
    print(f"\n4. 整体耗时: {total_time:.2f} 秒")
    print("="*60)

if __name__ == "__main__":
    # Windows系统需强制使用spawn启动方式（跨平台兼容）
    mp.set_start_method("spawn", force=True)
    main()