#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
该版本延迟较小，但帧乱序显示
"""
import os
import time
import cv2
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
# --- 请把你之前定义的 process_frame, postprocess_infer_result,
#     split_and_stack, stack_to_original, stitch_frames, init_device, run_inference
#     worker_process 等函数直接复用到这个文件中（下文代码假设这些函数已存在） ---
#
# 这里我会重新定义 worker_process 以更好支持低延迟行为（尽量快速推理并返回）
# 并给出 main() 的实时显示实现。
#
# ---------------- 配置 ----------------
CAMERA_DEVICE_PATH = "/dev/video20"
TARGET_RESOLUTION = (960, 720)
TARGET_FPS_DISPLAY = 20           # 显示目标FPS（仅用于sleep/画面节奏，不强制摄像头）
HEF_PATH = "/home/firefly/Denoising-rk3588J/models/dncnn_4split/dncnn_4split_16pad.hef"
NUM_DEVICES = 2
BATCH_SIZE = 1
# 将队列设小以降低延迟（优先丢弃旧帧），可根据设备吞吐微调为 2-6
QUEUE_MAX_SIZE = 4
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
        interpolation=cv2.INTER_LANCZOS4  # 高质量插值（与原代码PIL.LANCZOS对应）
    )
    
    # 2. BGR转RGB（cv2默认BGR，模型需要RGB）
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

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

# ---------------- worker_process (更简单的版本) ----------------
def worker_process(device_id, task_queue, result_queue, hef_path):
    """
    每个设备的工作进程：从 task_queue 尽快取出任务并推理，然后把结果放到 result_queue。
    任务格式: (batch_index, actual_batch_size, batch_tensor)
    返回格式: (batch_index, actual_batch_size, infer_time, infer_tensors)
    """
    try:
        device = init_device(hef_path, device_id)
        print(f"[Worker {device_id}] started, PID={os.getpid()}")
        while True:
            task = task_queue.get()  # blocking; keep device busy
            if task is None:
                break
            batch_index, actual_batch_size, batch_tensor = task
            try:
                batch_tensor = split_and_stack(batch_tensor)
                # 预期 batch_tensor 为形如 [N, H, W, C] (uint8/float32) -> run_inference 内处理
                infer_tensors, infer_time = run_inference(device, batch_tensor)
                # 如果 run_inference 返回的是 [N, ...]，需要把形状恢复到原始拼接方式
                # 保持与你原来一致：stack_to_original(infer_tensors) 等由 run_inference 的输出决定
                # 这里直接返回 infer_tensors（主进程做后处理/拼接）
                infer_tensors = stack_to_original(infer_tensors)
                result_queue.put((batch_index, actual_batch_size, infer_time, infer_tensors))
            except Exception as e:
                print(f"[Worker {device_id}] inference error: {e}")
    except Exception as e:
        print(f"[Worker {device_id}] init error: {e}")
    finally:
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
    cap.set(cv2.CAP_PROP_FPS, 30)  # 摄像头最低30fps，尽量读满

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
                    _ = task_queues[target_dev].get_nowait()  # 丢弃最旧
                    task_queues[target_dev].put_nowait((batch_index, actual_batch_size, batch_tensor))
                    # 同时要丢弃 result_cache 中对应最旧的索引（无法精确知道被丢弃的 batch_index — 但我们尽量保持缓存大小）
                    # 为简单处理：若 result_cache 太大则修剪最旧项
                    if len(result_cache) > QUEUE_MAX_SIZE * 4:
                        # 按 insertion order 剪掉最旧若干
                        keys = list(result_cache.keys())
                        for k in keys[:4]:
                            result_cache.pop(k, None)
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
                    # 注意：infer_tensors 可能是压缩/子图形式，需要经过 stack_to_original 或 postprocess
                    try:
                        # 如果需要把 infer_tensors 转回原始形状：stack_to_original(infer_tensors)
                        # 我沿用你原来的 stack_to_original()：若输出为 [4,376,496,3] 则该函数会报错，视你 run_inference 返回格式而定
                        infer_tensors_full = infer_tensors
                    except Exception:
                        infer_tensors_full = infer_tensors

                    # 这里只取第0张（因为我们用 batch_size=1）
                    try:
                        infer_frame_bgr = infer_tensors_full[0]
                        
                    except Exception:
                        # 若 postprocess 失败，跳过
                        print(1)
                        continue

                    # 如果缓存中存在对应原始帧，则一起拼接；否则只显示推理帧
                    orig = result_cache.pop(batch_idx_res, None)
                    if orig is None:
                        latest_infer_frame = infer_frame_bgr
                        latest_original_frame = None
                    else:
                        latest_original_frame = orig
                        latest_infer_frame = infer_frame_bgr
                    latest_infer_frame = cv2.cvtColor(latest_infer_frame, cv2.COLOR_BGR2RGB)
                    cv2.imshow(WINDOW_NAME, latest_infer_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User requested exit.")
                        raise KeyboardInterrupt

                    # 继续读取队列，最后一个结果会成为最新显示
                except queue.Empty:
                    break

            # # 5) 显示最新帧（优先显示推理帧+原始拼接；若没有推理帧则显示原始）
            # now = time.time()
            # # 控制显示频率（减少 CPU 占用）— 以 TARGET_FPS_DISPLAY 为目标
            # if now - last_display_time >= (1.0 / max(1.0, TARGET_FPS_DISPLAY)):
            #     if latest_infer_frame is not None:
            #         # 只有推理帧
            #         print(1)
            #         cv2.imshow(WINDOW_NAME, latest_infer_frame)
            #     else:
            #         print(2)
            #         # 没有推理帧：显示原始帧（降低延迟）
            #         cv2.imshow(WINDOW_NAME, frame_original)

            #     # 键盘监听：按 q 退出
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         print("User requested exit.")
            #         raise KeyboardInterrupt

            # short sleep to yield
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
