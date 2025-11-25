#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
该版本延迟较小，但帧乱序显示
"""
import os
import time
import cv2

cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(True)

import numpy as np
import multiprocessing as mp
import queue
from datetime import datetime
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


# ---------------- 配置 ----------------
CAMERA_DEVICE_PATH = "/dev/video20"
TARGET_RESOLUTION = (960, 720)
TARGET_FPS_DISPLAY = 20           # 显示目标FPS（仅用于sleep/画面节奏，不强制摄像头）
HEF_PATH = "dncnn_80ep_l9_4split_16pad.hef"
NUM_DEVICES = 2
BATCH_SIZE = 1
# 将队列设小以降低延迟（优先丢弃旧帧），可根据设备吞吐微调为 2-6
QUEUE_MAX_SIZE = 1
# 主循环最长运行（秒），设为 None 则无限运行直到按 q 退出
RUN_DURATION = None

# 显示窗口名字
WINDOW_NAME = "Live Denoise (Original | Inferred)"

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
        interpolation=cv2.INTER_AREA  # 高质量插值（与原代码PIL.LANCZOS对应）
    )
    
    # 2. BGR转RGB（cv2默认BGR，模型需要RGB）
    #frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_resized

    # 3. 格式转换：HWC -> CHW， dtype -> float32
    # frame_chw = frame_rgb.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)
    frame_float = frame_rgb.astype(np.float32)
    
    process_time = time.time() - start_time
    return frame_float, process_time, frame_resized

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
    """
    在单个设备上运行推理，返回推理结果与耗时
    device: 包含 network_group、vstream 参数的字典
    input_batch: numpy 数组或 tensor，形状 [N, H, W, C]
    """
    network_group = device["network_group"]
    input_vstreams_params = device["input_vstreams_params"]
    output_vstreams_params = device["output_vstreams_params"]
    network_group_params = device["network_group_params"]
    input_vstream_info = device["input_vstream_info"]
    output_vstream_info = device["output_vstream_info"]

    start_time = time.time()

    # 与 worker_process 统一结构：activate 与 InferVStreams 同层
    with network_group.activate(network_group_params), \
         InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:

        # 准备输入字典
        input_data = {input_vstream_info.name: input_batch}
        # 执行推理
        infer_results = infer_pipeline.infer(input_data)

    inference_time = time.time() - start_time
    output_tensor = infer_results[output_vstream_info.name]

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

# ---------------- worker_process (更简单的版本) ----------------
def worker_process(device_id, task_queue, result_queue, hef_path):
    """
    每个设备的工作进程：持续从队列取任务并推理，然后将结果放入结果队列。
    任务格式: (batch_index, actual_batch_size, batch_tensor)
    返回格式: (batch_index, actual_batch_size, infer_time, infer_tensors)
    """
    try:
        # 初始化设备
        device = init_device(hef_path, device_id)
        ng = device["network_group"]
        ng_params = device["network_group_params"]
        input_vp = device["input_vstreams_params"]
        output_vp = device["output_vstreams_params"]
        print(f"[Worker {device_id}] started, PID={os.getpid()}")

        # 将 InferVStreams 和 activate 保持一次性上下文管理（效率更高）
        with ng.activate(ng_params), InferVStreams(ng, input_vp, output_vp) as pipeline:
            while True:
                task = task_queue.get()
                if task is None:
                    break  # None 表示退出
                batch_index, actual_batch_size, ori_tensor = task

                try:
                    batch_tensor = split_and_stack(ori_tensor)
                    input_data = {device["input_vstream_info"].name: batch_tensor}

                    start_time = time.time()
                    infer_results = pipeline.infer(input_data)
                    infer_time = time.time() - start_time

                    infer_tensors = infer_results[device["output_vstream_info"].name]
                    infer_tensors = stack_to_original(infer_tensors)

                    result_queue.put((batch_index, actual_batch_size, infer_time, infer_tensors))

                except Exception as e:
                    print(f"[Worker {device_id}] inference error: {e}")

    except Exception as e:
        print(f"[Worker {device_id}] init error: {e}")

    finally:
        # 安全释放设备
        if "device" in locals():
            try:
                device["target"].release()
            except Exception:
                pass
        print(f"[Worker {device_id}] exiting")

# ---------------- 主流程 (低延迟显示) ----------------
def main():
    mp.set_start_method("spawn", force=True)

    print("启动实时显示模式 (低延迟, 丢帧优先)")
    # ---------------- 启动 worker 进程 ----------------
    task_queues = [mp.Queue(maxsize=QUEUE_MAX_SIZE) for _ in range(NUM_DEVICES)]
    result_queue = mp.Queue(maxsize=QUEUE_MAX_SIZE * NUM_DEVICES * 2)

    processes = []
    for dev_id in range(NUM_DEVICES):
        p = mp.Process(target=worker_process, args=(dev_id, task_queues[dev_id], result_queue, HEF_PATH))
        p.start()
        processes.append(p)
    time.sleep(1.0)

    # ---------------- 打开摄像头 ----------------
    cap = cv2.VideoCapture(CAMERA_DEVICE_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {CAMERA_DEVICE_PATH}")

    # 设置摄像头期望参数（并不保证一定生效）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 60)  # 摄像头最低30fps，尽量读满

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_cam_fps:.1f}fps")

    # ---------------- 状态变量 ----------------
    batch_index = 0
    # result_cache 只保存原始 frames 用于后续拼接 (key=batch_index)
    result_cache = {}
    latest_infer_frame = None       # 最新推理帧（用于显示）
    latest_original_frame = None    # 对应的原始帧（用于拼接显示）
    last_display_time = time.time()
    ema_fps = None
    alpha = 0.1  # EMA smoothing for fps

    read_start = time.time()

    print("按 q 键退出。")
    show_idx = 0
    # 在循环外初始化帧率统计相关变量
    frame_count = 0  # 统计显示的总帧数
    start_time = time.time()  # 起始时间
    last_print_time = start_time  # 上次打印帧率的时间
    try:
        while True:
            if RUN_DURATION is not None and (time.time() - read_start) > RUN_DURATION:
                print("达到运行时长，退出")
                break

            # 1) 读取摄像头帧 (non-blocking best effort)
            ret, frame = cap.read()
            if not ret:
                # 读取失败时短暂sleep避免忙等
                time.sleep(0.005)
                continue

            # 预处理（快速）
            frame_processed, _, frame_original = process_frame(frame)

            # 2) 将帧打包为 batch（此处 BATCH_SIZE=1，大多数情况立即发送）
            batch_tensor = np.expand_dims(frame_processed, axis=0)  # [1, H, W, C] 或与模型输入匹配
            actual_batch_size = 1

            # round-robin 选择设备
            target_dev = batch_index % NUM_DEVICES

            # 3) 尝试把任务放入对应设备队列；若满 -> 丢弃队列中最旧的任务以保证最新任务入队（优先实时性）
            try:
                task_queues[target_dev].put_nowait((batch_index, actual_batch_size, batch_tensor))
                # 缓存原始帧以便后续拼接显示（只保存必要信息）
                result_cache[batch_index] = frame_original
                # debug
                # print(f"sent batch {batch_index} to dev {target_dev}")
                batch_index += 1
            except Exception as e:
                # 队列满时尝试弹出最旧任务以腾空间（丢帧策略）
                try:
                    batch_index,_,_ = task_queues[target_dev].get_nowait()  # 丢弃最旧
                    if batch_index in result_cache:
                        del result_cache[batch_index]
                    # 现在再尝试放入新任务
                    task_queues[target_dev].put_nowait((batch_index, actual_batch_size, batch_tensor))
                    result_cache[batch_index] = frame_original
                    batch_index += 1
                except Exception:
                    # 如果仍失败，放弃当前帧（立即丢帧以保证低延迟）
                    # print("drop frame due to full queue")
                    pass

            # 4) 处理 result_queue（尽可能清空，保留最新结果显示）
            # 我们会把 result_queue 中的所有结果读取出来，用最后一个覆盖 latest_infer_frame
            
            

            while True:
                try:
                    batch_idx_res, actual_frames, infer_time, infer_tensors = result_queue.get_nowait()
                    if batch_idx_res < show_idx:
                        continue

                    show_idx = batch_idx_res
                    try:
                        infer_tensors_full = infer_tensors
                    except Exception:
                        infer_tensors_full = infer_tensors

                    try:
                        infer_frame_bgr = infer_tensors_full[0]
                    except Exception:
                        print(1)
                        continue


                    latest_original_frame = result_cache.get(batch_idx_res, None)
                    latest_infer_frame = infer_frame_bgr
                    try:
                        if latest_original_frame is not None:
                            # 确保尺寸一致
                            h1, w1 = latest_original_frame.shape[:2]
                            h2, w2 = latest_infer_frame.shape[:2] 
                            if (h1, w1) != (h2, w2):
                                latest_infer_frame = cv2.resize(latest_infer_frame, (w1, h1))


                            # 左右拼接 (原图 | 推理图)
                            combined = np.concatenate((latest_original_frame, latest_infer_frame), axis=1)
                            #cv2.imshow(WINDOW_NAME, combined)
                            cv2.imshow(WINDOW_NAME, combined)
                        else:
                            # 仅显示推理帧
                            cv2.imshow(WINDOW_NAME, latest_infer_frame)

                    except Exception as e:
                        print(f"[Display warning] failed to display combined frame: {e}")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User requested exit.")
                        raise KeyboardInterrupt

                except queue.Empty:
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("KeyboardInterrupt -> exiting main loop")
    finally:
        # 结束：向 worker 发送终止信号
        for q in task_queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass

        # 等待进程退出
        for p in processes:
            p.join(timeout=5)
            print(f"Worker {p.pid} join status: exitcode={p.exitcode}")

        cap.release()
        cv2.destroyAllWindows()
        print("Cleaned up and exiting.")

if __name__ == "__main__":
    main()
