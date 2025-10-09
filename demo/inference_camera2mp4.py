import os
import time
import cv2
import numpy as np
import multiprocessing as mp
import queue
from tqdm import tqdm
import argparse
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
CAMERA_DEVICE_PATH = "/dev/video20"  # 摄像头设备路径（根据实际情况修改）
TARGET_RESOLUTION = (960, 720)       # 目标分辨率 (width, height)
TARGET_FPS = 20                      # 目标帧率
VIDEO_FORMAT = cv2.VideoWriter_fourcc(*"MJPG")  # 摄像头格式（MJPG支持高帧率）

# 推理参数
HEF_PATH = "/home/firefly/Denoising-rk3588J/demo/dncnn_4split_16pad.hef"  # Hailo模型路径
BATCH_SIZE = 1                        # 单设备批次大小（平衡实时性与效率）
INPUT_SHAPE = (3, 720, 960)          # 模型输入形状 (channel, height, width)
NUM_DEVICES = 2                       # 启用的Hailo加速棒数量
QUEUE_MAX_SIZE = 200                  # 任务队列最大缓存（避免帧堆积）


# 视频保存参数（新增）
VIDEO_CODEC = cv2.VideoWriter_fourcc(*"mp4v")  # 输出视频编码（mp4格式兼容好）
VIDEO_EXT = ".mp4"                     # 视频文件后缀
SPLIT_LINE_COLOR = (0, 255, 0)        # 拼接帧分割线颜色（绿色）
SPLIT_LINE_WIDTH = 2                   # 分割线宽度（像素）
TEXT_COLOR = (255, 255, 255)          # 文字颜色（白色）
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX   # 文字字体
TEXT_SIZE = 0.8                        # 文字大小
TEXT_THICKNESS = 2                     # 文字粗细


# -------------------------- 帧预处理（适配摄像头输入） --------------------------
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
    return frame_float, process_time, frame_resized  # 新增返回调整后原始帧


# -------------------------- 推理结果后处理（新增） --------------------------
def postprocess_infer_result(infer_tensor):
    """
    将推理输出张量转换为BGR帧（适配OpenCV显示/保存）
    :param infer_tensor: 推理输出张量（CHW格式，uint8）
    :return: 后处理后的BGR帧（HWC格式）
    """
    # 1. 移除多余维度（若有）
    tensor_squeezed = np.squeeze(infer_tensor)
    
    # 2. CHW → HWC（OpenCV需要HWC格式）
    if tensor_squeezed.shape[0] in [3, 1]:  # 若为单通道/三通道CHW格式
        frame_hwc = tensor_squeezed.transpose(1, 2, 0)
    else:
        frame_hwc = tensor_squeezed  # 若已为HWC，直接使用
    
    # 3. 单通道 → 三通道（若模型输出为灰度图，转为RGB兼容格式）
    if frame_hwc.shape[-1] == 1:
        frame_hwc = np.repeat(frame_hwc, 3, axis=-1)
    
    # 4. 尺寸裁剪（确保与原始帧分辨率一致，避免拼接错位）
    frame_resized = cv2.resize(
        frame_hwc, 
        dsize=TARGET_RESOLUTION, 
        interpolation=cv2.INTER_LANCZOS4
    )
    
    # 5. 格式转换：RGB → BGR（OpenCV默认BGR）
    frame_bgr = cv2.cvtColor(frame_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return frame_bgr

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
        raise ValueError("输入数组形状必须为[1, 720, 960, 3]")
    
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

# -------------------------- 帧拼接与标注（新增） --------------------------
def stitch_frames(original_frame, infer_frame, fps):
    """
    横向拼接原始帧与推理帧，并添加标注（文字+分割线）
    :param original_frame: 调整后原始BGR帧（HWC）
    :param infer_frame: 后处理后的推理BGR帧（HWC）
    :param fps: 当前实时FPS（用于标注）
    :return: 拼接后的BGR帧（HWC）
    """
    # 1. 横向拼接两帧（宽度叠加，高度一致）
    stitched_frame = cv2.hconcat([original_frame, infer_frame])
    
    # 2. 添加分割线（区分原始帧与推理帧）
    split_x = original_frame.shape[1]  # 分割线X坐标（原始帧宽度处）
    cv2.line(
        stitched_frame, 
        (split_x, 0),  # 起点（分割线顶部）
        (split_x, stitched_frame.shape[0]),  # 终点（分割线底部）
        SPLIT_LINE_COLOR, 
        SPLIT_LINE_WIDTH
    )
    
    # 3. 添加文字标注（左上角：原始帧标识，右上角：推理帧标识，右下角：FPS）
    # 3.1 原始帧标识
    cv2.putText(
        stitched_frame, 
        "Original Frame", 
        (20, 40),  # 文字位置（左上角偏移）
        TEXT_FONT, 
        TEXT_SIZE, 
        TEXT_COLOR, 
        TEXT_THICKNESS
    )
    
    # 3.2 推理帧标识
    cv2.putText(
        stitched_frame, 
        "Inferred Frame", 
        (split_x + 20, 40),  # 推理帧区域左上角偏移
        TEXT_FONT, 
        TEXT_SIZE, 
        TEXT_COLOR, 
        TEXT_THICKNESS
    )    
    return stitched_frame


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


# -------------------------- 工作进程（修改：保留推理结果用于拼接） --------------------------
def worker_process(device_id, task_queue, result_queue, hef_path):
    """
    设备工作进程：接收摄像头帧批次 → 推理 → 返回结果（含推理张量）
    :param task_queue: 任务队列（元素：(batch_tensor, actual_batch_size, batch_index, original_frames)）
    :param result_queue: 结果队列（元素：(batch_index, actual_batch_size, infer_time, infer_tensors)）
    """
    try:
        device = init_device(hef_path, device_id)
        print(f"设备 {device_id} 工作进程启动（PID: {os.getpid()}）")
        
        while True:
            task = task_queue.get()
            if task is None:  # 终止信号
                break
            
            batch_tensor, actual_batch_size, batch_index, _ = task  # 忽略原始帧（主进程保留）
            batch_tensor = split_and_stack(batch_tensor)
            # 执行推理（保留输出张量，用于后处理）
            infer_tensors, infer_time = run_inference(device, batch_tensor)
            infer_tensors = stack_to_original(infer_tensors)
            # 向主进程返回：批次索引、有效帧数、推理耗时、推理张量（仅返回有效部分）
            result_queue.put((batch_index, actual_batch_size, infer_time, infer_tensors[:actual_batch_size]))
    
    except Exception as e:
        print(f"设备 {device_id} 工作进程出错: {str(e)}")
    finally:
        # 释放设备资源
        if "device" in locals():
            device["target"].release()
        print(f"设备 {device_id} 工作进程退出")

def parse_args():
    parser = argparse.ArgumentParser(description="Video inference settings")
    parser.add_argument(
        "--camera",
        type=str,
        default="/dev/video20",
        help="Camera id."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/inference_videos",
        help="Directory to save inference results (default: output/inference_videos)"
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=10,
    )
    
    args = parser.parse_args()
    return args

# -------------------------- 主流程（核心修改：帧缓存+拼接+视频保存） --------------------------
def main():
    args = parse_args()
    RUN_DURATION = args.seconds
    SAVE_VIDEO_DIR = args.save_dir
    total_start_time = time.time()
    print("="*60)
    print(f"开始实时推理测试 | 目标: {TARGET_RESOLUTION[0]}×{TARGET_RESOLUTION[1]} @ {TARGET_FPS}fps | 运行时长: {RUN_DURATION}s")
    print(f"设备数量: {NUM_DEVICES} | 批次大小: {BATCH_SIZE} | 模型路径: {HEF_PATH}")
    print(f"视频保存目录: {SAVE_VIDEO_DIR} | 视频编码: {VIDEO_CODEC.to_bytes(4, 'little').decode('utf-8')}")
    print("="*60)

    # -------------------------- 1. 初始化视频保存目录与对象（新增） --------------------------
    os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
    # 生成唯一视频文件名（时间戳+分辨率+帧率）
    video_filename = f"infer_stitched_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}_{TARGET_FPS}fps{VIDEO_EXT}"
    video_save_path = os.path.join(SAVE_VIDEO_DIR, video_filename)
    # 计算拼接后视频分辨率（宽度=2*原始宽度，高度=原始高度）
    stitched_resolution = (TARGET_RESOLUTION[0] * 2, TARGET_RESOLUTION[1])
    # 初始化视频写入对象
    video_writer = cv2.VideoWriter(
        video_save_path,
        VIDEO_CODEC,
        TARGET_FPS,  # 视频帧率（与目标一致）
        stitched_resolution  # 拼接后分辨率
    )
    if not video_writer.isOpened():
        raise RuntimeError(f"无法初始化视频写入对象，路径：{video_save_path}")
    print(f"\n✅ 视频写入对象已初始化 | 保存路径: {video_save_path} | 拼接分辨率: {stitched_resolution[0]}×{stitched_resolution[1]}")

    # -------------------------- 2. 初始化进程与队列 --------------------------
    # 任务队列（每个设备一个）：新增原始帧缓存（用于拼接）
    task_queues = [mp.Queue(maxsize=QUEUE_MAX_SIZE) for _ in range(NUM_DEVICES)]
    # 结果队列：接收推理统计+推理张量
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

    # -------------------------- 3. 初始化摄像头 --------------------------
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
        # 更新拼接分辨率（若实际分辨率与目标不一致）
        stitched_resolution = (actual_width * 2, actual_height)
        video_writer.set(cv2.CAP_PROP_FRAME_WIDTH, stitched_resolution[0])
        video_writer.set(cv2.CAP_PROP_FRAME_HEIGHT, stitched_resolution[1])

    # -------------------------- 4. 初始化统计与缓存变量（新增帧缓存） --------------------------
    read_total_frames = 0          # 摄像头读取的总帧数
    infer_total_frames = 0         # 成功推理的总帧数
    saved_total_frames = 0         # 成功保存的拼接帧数
    read_start_time = time.time()  # 摄像头读取开始时间
    infer_start_time = None        # 推理开始时间（第一个批次发送时记录）
    total_infer_compute_time = 0   # 推理计算总耗时（所有设备累加）
    batch_index = 0                # 批次索引（用于匹配任务与结果）
    
    # 设备帧缓存：新增原始帧缓存（每个批次对应一组原始帧，用于拼接）
    device_buffers = {
        "processed_frames": [[] for _ in range(NUM_DEVICES)],  # 预处理后帧（用于推理）
        "original_frames": [[] for _ in range(NUM_DEVICES)],    # 调整后原始帧（用于拼接）
    }
    
    # 结果缓存：匹配批次索引与推理结果（解决多设备返回顺序乱序问题）
    result_cache = {}

    # -------------------------- 5. 实时读取+推理+拼接+保存循环（核心修改） --------------------------
    print(f"\n开始读取摄像头帧（按 'q' 提前退出）...")
    while (time.time() - read_start_time) < RUN_DURATION:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("⚠️  无法读取摄像头帧，退出循环")
            break
        
        read_total_frames += 1
        current_time = time.time()

        # 预处理帧（新增返回调整后原始帧）
        frame_processed, _, frame_original = process_frame(frame)

        # 分配帧到设备缓存（轮询分配，均衡负载）
        target_device_id = read_total_frames % NUM_DEVICES
        device_buffers["processed_frames"][target_device_id].append(frame_processed)
        device_buffers["original_frames"][target_device_id].append(frame_original)  # 缓存原始帧

        # 当缓存达到批次大小时，发送推理任务（新增原始帧列表）
        if len(device_buffers["processed_frames"][target_device_id]) >= BATCH_SIZE:
            # 提取批次帧并补零（不足批次大小时）
            batch_processed = device_buffers["processed_frames"][target_device_id][:BATCH_SIZE]
            batch_original = device_buffers["original_frames"][target_device_id][:BATCH_SIZE]  # 原始帧批次
            actual_batch_size = len(batch_processed)
            
            if actual_batch_size < BATCH_SIZE:
                pad_size = BATCH_SIZE - actual_batch_size
                batch_processed += [np.zeros_like(batch_processed[0]) for _ in range(pad_size)]
            
            # 转换为批次张量
            batch_tensor = np.stack(batch_processed, axis=0)

            # 发送任务到队列（包含原始帧批次，非阻塞避免阻塞读取）
            try:
                task_queues[target_device_id].put(
                    (batch_tensor, actual_batch_size, batch_index, batch_original),
                    block=False
                )
                # 缓存原始帧批次（主进程保留，用于后续拼接）
                result_cache[batch_index] = {
                    "original_frames": batch_original,
                    "processed": False  # 标记是否已处理推理结果
                }
                batch_index += 1
                # 记录推理开始时间（第一个任务发送时）
                if infer_start_time is None:
                    infer_start_time = current_time
                print(f"📤 设备 {target_device_id} 发送批次 {batch_index-1}（有效帧: {actual_batch_size}）", end="\r")
            except mp.Queue.Full:
                print(f"⚠️  设备 {target_device_id} 任务队列已满，丢弃当前批次", end="\r")
                # 丢弃对应的原始帧缓存
                device_buffers["original_frames"][target_device_id] = device_buffers["original_frames"][target_device_id][BATCH_SIZE:]

            # 清空已发送的缓存
            device_buffers["processed_frames"][target_device_id] = device_buffers["processed_frames"][target_device_id][BATCH_SIZE:]
            device_buffers["original_frames"][target_device_id] = device_buffers["original_frames"][target_device_id][BATCH_SIZE:]

        # 处理推理结果（非阻塞，避免阻塞读取）
        while not result_queue.empty():
            try:
                # 从结果队列获取数据（批次索引、有效帧数、推理耗时、推理张量）
                batch_idx, actual_frames, infer_time, infer_tensors = result_queue.get(block=False)
                total_infer_compute_time += infer_time
                infer_total_frames += actual_frames
                
                # 检查该批次原始帧是否在缓存中
                if batch_idx not in result_cache or result_cache[batch_idx]["processed"]:
                    print(f"⚠️  批次 {batch_idx} 原始帧缓存丢失或已处理，跳过拼接")
                    continue
                
                # 获取该批次原始帧
                batch_original = result_cache[batch_idx]["original_frames"]
                
                # 逐帧处理：后处理推理结果 → 拼接 → 保存
                for i in range(actual_frames):
                    # 后处理推理张量为BGR帧
                    infer_frame = postprocess_infer_result(infer_tensors[i])
                    # 计算实时FPS（用于标注）
                    elapsed_time = time.time() - read_start_time
                    current_fps = read_total_frames / elapsed_time if elapsed_time > 1e-3 else 0.0
                    # 拼接原始帧与推理帧
                    stitched_frame = stitch_frames(batch_original[i], infer_frame, current_fps)
                    # 写入视频
                    video_writer.write(stitched_frame)
                    saved_total_frames += 1
                
                # 标记该批次已处理，释放缓存
                result_cache[batch_idx]["processed"] = True
                del result_cache[batch_idx]
                
            except Exception as e:
                print(f"⚠️  处理推理结果时出错: {str(e)}", end="\r")

        # 按 'q' 键提前退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n🛑 用户按下 'q' 键，提前退出")
            break

    # -------------------------- 6. 处理剩余帧（缓存中未发送的帧） --------------------------
    print(f"\n\n处理缓存中剩余的帧...")
    for device_id in range(NUM_DEVICES):
        remaining_processed = device_buffers["processed_frames"][device_id]
        remaining_original = device_buffers["original_frames"][device_id]
        if len(remaining_processed) == 0:
            continue
        
        # 处理剩余帧（不足批次大小时补零）
        actual_batch_size = len(remaining_processed)
        if actual_batch_size < BATCH_SIZE:
            pad_size = BATCH_SIZE - actual_batch_size
            remaining_processed += [np.zeros_like(remaining_processed[0]) for _ in range(pad_size)]
        
        batch_tensor = np.stack(remaining_processed, axis=0)
        try:
            # 发送剩余任务（包含原始帧）
            task_queues[device_id].put(
                (batch_tensor, actual_batch_size, batch_index, remaining_original),
                block=True, 
                timeout=5
            )
            # 缓存原始帧
            result_cache[batch_index] = {
                "original_frames": remaining_original,
                "processed": False
            }
            batch_index += 1
            print(f"📤 设备 {device_id} 发送剩余批次 {batch_index-1}（有效帧: {actual_batch_size}）")
        except (mp.Queue.Full, TimeoutError):
            print(f"⚠️  设备 {device_id} 队列满/超时，无法发送剩余 {actual_batch_size} 帧")

    # -------------------------- 7. 处理剩余推理结果（确保所有帧都被拼接） --------------------------
    print(f"\n处理剩余推理结果...")
    remaining_batches = len(result_cache)
    if remaining_batches > 0:
        print(f"等待 {remaining_batches} 个批次的推理结果...")
        start_wait_time = time.time()
        # 等待剩余结果（超时时间10秒）
        while len(result_cache) > 0 and (time.time() - start_wait_time) < 10:
            if not result_queue.empty():
                try:
                    batch_idx, actual_frames, infer_time, infer_tensors = result_queue.get(block=False)
                    total_infer_compute_time += infer_time
                    infer_total_frames += actual_frames
                    
                    if batch_idx not in result_cache or result_cache[batch_idx]["processed"]:
                        continue
                    
                    # 拼接并保存剩余帧
                    batch_original = result_cache[batch_idx]["original_frames"]
                    for i in range(actual_frames):
                        infer_frame = postprocess_infer_result(infer_tensors[i])
                        elapsed_time = time.time() - read_start_time
                        current_fps = read_total_frames / elapsed_time if elapsed_time > 1e-3 else 0.0
                        stitched_frame = stitch_frames(batch_original[i], infer_frame, current_fps)
                        video_writer.write(stitched_frame)
                        saved_total_frames += 1
                    
                    result_cache[batch_idx]["processed"] = True
                    del result_cache[batch_idx]
                    print(f"✅ 处理剩余批次 {batch_idx}，还剩 {len(result_cache)} 个批次", end="\r")
                except Exception as e:
                    print(f"⚠️  处理剩余结果出错: {str(e)}", end="\r")
            time.sleep(0.1)  # 避免CPU空转

    # -------------------------- 8. 发送终止信号并清理 --------------------------
    # 向所有工作进程发送终止信号
    for q in task_queues:
        q.put(None)
    
    # 收集剩余结果（仅统计，不保存）
    print(f"\n\n收集剩余推理统计...")
    processed_batches = 0
    total_batches = batch_index
    progress_bar = tqdm(total=total_batches, desc="推理进度")
    while processed_batches < total_batches:
        try:
            batch_idx, actual_frames, infer_time, _ = result_queue.get(block=True, timeout=5)
            total_infer_compute_time += infer_time
            infer_total_frames += actual_frames
            processed_batches += 1
            progress_bar.update(1)
        except queue.Empty:
            print(f"⚠️  结果队列超时，未收集到所有批次结果（已处理 {processed_batches}/{total_batches}）")
            break
    progress_bar.close()

    # -------------------------- 9. 释放资源（关键：关闭视频写入对象） --------------------------
    video_writer.release()  # 必须关闭，否则视频文件损坏
    cap.release()
    cv2.destroyAllWindows()
    for p in processes:
        p.join(timeout=10)
        print(f"🔚 设备进程 {p.pid} 退出状态: {'正常' if p.exitcode == 0 else '异常'}")
    print(f"\n✅ 视频已保存至: {video_save_path} | 共保存 {saved_total_frames} 帧拼接画面")

    # -------------------------- 10. 计算并输出FPS统计（保留原逻辑） --------------------------
    # 3. 视频保存性能（新增）
    print(f"\n3. 视频保存性能")
    print(f"   - 总保存拼接帧数: {saved_total_frames} 帧")
    print(f"   - 视频保存路径: {video_save_path}")
    print(f"   - 视频分辨率: {stitched_resolution[0]}×{stitched_resolution[1]}")
    print(f"   - 视频目标帧率: {TARGET_FPS} fps")


if __name__ == "__main__":
    from datetime import datetime  # 延迟导入，避免未使用时加载
    # Windows系统需强制使用spawn启动方式（跨平台兼容）
    mp.set_start_method("spawn", force=True)
    main()