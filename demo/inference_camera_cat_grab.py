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
from multiprocessing import Manager
import queue
import argparse
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
TARGET_FPS_DISPLAY = 40           # 显示目标FPS（仅用于sleep/画面节奏，不强制摄像头）
# 去噪强度 -> HEF 文件名（与 demo 目录下文件名一致）
STRENGTH_TO_HEF = {
    0.3: "dncnn_ch32_lite_rgb_376x496_alpha0_3.hef",
    0.5: "dncnn_ch32_lite_rgb_376x496_alpha0_5.hef",
    0.8: "dncnn_ch32_lite_rgb_376x496_alpha0_8.hef",
    1.0: "dncnn_ch32_lite_rgb_376x496_alpha1_0.hef",
}
NUM_DEVICES = 2
BATCH_SIZE = 1
ENHANCE_HEF_PATH = "enhance.hef"
# 将队列设小以降低延迟（优先丢弃旧帧），可根据设备吞吐微调为 2-6
QUEUE_MAX_SIZE = 1
# 主循环最长运行（秒），设为 None 则无限运行直到按 q 退出
RUN_DURATION = None

# 显示窗口名字
WINDOW_NAME = "Live Denoise (Original | Inferred)"


def parse_args():
    parser = argparse.ArgumentParser(description="实时去噪（低延迟）")
    parser.add_argument(
        "--strength", "-s",
        type=float,
        choices=[0.3, 0.5, 0.8, 1.0],
        default=0.5,
        help="去噪强度，对应加载的 HEF 模型 (默认: 0.3)",
    )
    return parser.parse_args()


def get_hef_path(strength):
    """根据去噪强度返回对应 HEF 的完整路径（脚本所在 demo 目录）。"""
    name = STRENGTH_TO_HEF.get(strength)
    if name is None:
        raise ValueError(f"不支持的强度 {strength}，可选: 0.3, 0.5, 0.8, 1.0")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def process_frame(frame):
    """
    处理摄像头BGR帧为模型输入格式（RGB+CHW+float32）
    :param frame: cv2读取的BGR帧（HWC格式）
    :return: 预处理后的数据（CHW格式）、预处理耗时、调整后原始BGR帧（用于拼接）
    """
    start_time = time.time()
    
    # 1. 调整分辨率（确保与目标一致，后续拼接时尺寸统一）
    # 如果frame的宽小于496或者高小于376，则进行resize。否则直接返回原帧。
    # if frame.shape[1] < 496 or frame.shape[0] < 376:
    #     frame_resized = cv2.resize(
    #         frame, 
    #         dsize=TARGET_RESOLUTION, 
    #         interpolation=cv2.INTER_AREA  # 高质量插值（与原代码PIL.LANCZOS对应）
    #     )
    # else:
    frame_resized = frame
    
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
    将图像数组分割为4张子图并堆叠为[4, 360, 480, 3]
    
    参数:
        batch_tensor: 形状为[1, 720, 960, 3]的numpy数组
        
    返回:
        形状为[4, 360, 480, 3]的numpy数组
    """

    
    # 移除批次维度，得到[720, 960, 3]
    image = batch_tensor[0]
    
    # 计算子图的高度和宽度
    sub_height = 720 // 2
    sub_width = 960 // 2
    
    # 分割为4个子图
    # 左上角
    sub1 = image[:sub_height+16, :sub_width+16, :]
    # 左下角
    sub2 = image[-(sub_height+16):, :sub_width+16, :]
    # 右上角
    sub3 = image[:sub_height+16, -(sub_width+16):, :]
    # 右下角
    sub4 = image[-(sub_height+16):, -(sub_width+16):, :]
    
    # 堆叠成[4, 376, 496, 3]的数组
    stacked = np.stack([sub1, sub2, sub3, sub4], axis=0)
    
    return stacked

def stack_to_original(sub_images, original_image):
    # 检查输入形状是否正确
    if sub_images.shape != (4, 376, 496, 3):
        raise ValueError(f"输入数组形状必须为[4, 360, 480, 3], 实际形状为{sub_images.shape}")
    if original_image.shape[0] != 1:
        original_image = np.expand_dims(original_image, axis=0)
    # 提取4个子图
    sub1, sub2, sub3, sub4 = sub_images[0], sub_images[1], sub_images[2], sub_images[3]
    # 将四个子图替换original_image的对应区域
    original_image[0, :360,:480,:] = sub1[:360,:480,:]
    original_image[0, -360:,:480,:] = sub2[16:,:480,:]
    original_image[0, :360,-480:,:] = sub3[:360,16:,:]
    original_image[0, -360:,-480:,:] = sub4[16:,16:,:]
    
    # 添加批次维度
    return original_image.astype(np.uint8)


def run_enhance_on_image(device, denoised_img):
    """对单张去噪图用 enhance 模型处理（分 4 块推理再拼回），返回 (720, 960, 3) BGR。"""
    if denoised_img.shape[0] != 720 or denoised_img.shape[1] != 960:
        denoised_img = cv2.resize(denoised_img, TARGET_RESOLUTION)
    batch = np.expand_dims(denoised_img.astype(np.float32), axis=0)
    blocks = split_and_stack(batch)
    out_blocks, _ = run_inference(device, blocks)
    out_blocks, _ = run_inference(device, out_blocks.astype(np.float32))
    out_blocks, _ = run_inference(device, out_blocks.astype(np.float32))
    result = stack_to_original(out_blocks, denoised_img.copy())
    return result[0]


def process_pending_denoised_with_enhance(base_dir):
    """对 grab/denoised 中尚未在 grab/enhanced 中有对应结果的图像跑 enhance，并保存 original|denoised|enhanced 拼接图到 grab/enhanced。"""
    dir_original = os.path.join(base_dir, "grab", "original")
    dir_denoised = os.path.join(base_dir, "grab", "denoised")
    dir_enhanced = os.path.join(base_dir, "grab", "enhanced")
    if not os.path.isdir(dir_denoised):
        return
    os.makedirs(dir_enhanced, exist_ok=True)
    pending = []
    for name in os.listdir(dir_denoised):
        if not name.endswith((".png", ".jpg", ".jpeg")):
            continue
        path_enhanced = os.path.join(dir_enhanced, name)
        if os.path.isfile(path_enhanced):
            continue
        path_denoised = os.path.join(dir_denoised, name)
        path_original = os.path.join(dir_original, name)
        if not os.path.isfile(path_denoised) or not os.path.isfile(path_original):
            continue
        pending.append((name, path_original, path_denoised, path_enhanced))
    if not pending:
        return
    if not os.path.isfile(ENHANCE_HEF_PATH):
        print(f"Enhance HEF not found: {ENHANCE_HEF_PATH}, skip pending enhance.")
        return
    try:
        print("Waiting ...")
        time.sleep(10)
        device = init_device(ENHANCE_HEF_PATH, 0)
    except Exception as e:
        # 等待3秒后重试
        time.sleep(8)
        try:
            device = init_device(ENHANCE_HEF_PATH, 0)
        except Exception as e:
            print(f"Enhance device init failed: {e}, skip pending enhance.")
            return
    try:
        for name, path_original, path_denoised, path_enhanced in pending:
            try:
                original = cv2.imread(path_original)
                denoised = cv2.imread(path_denoised)
                if original is None or denoised is None:
                    continue
                enhanced = run_enhance_on_image(device, denoised)
                h, w = original.shape[:2]
                if (denoised.shape[0], denoised.shape[1]) != (h, w):
                    denoised = cv2.resize(denoised, (w, h))
                if (enhanced.shape[0], enhanced.shape[1]) != (h, w):
                    enhanced = cv2.resize(enhanced, (w, h))
                combined = np.concatenate((original, denoised, enhanced), axis=1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                color = (255, 255, 255)
                cv2.putText(combined, "Original", (20, h - 20), font, font_scale, color, thickness)
                cv2.putText(combined, "Denoised", (w + 20, h - 20), font, font_scale, color, thickness)
                cv2.putText(combined, "Enhanced", (2 * w + 20, h - 20), font, font_scale, color, thickness)
                cv2.imwrite(path_enhanced, combined)
                print(f"Enhanced saved: {path_enhanced}")
            except Exception as e:
                print(f"Enhance failed for {name}: {e}")
    finally:
        try:
            device["target"].release()
        except Exception:
            pass


# ---------------- worker_process (更简单的版本) ----------------
def worker_process(device_id, task_queue, result_queue, hef_path):
    """每个设备的工作进程：持续从队列取任务并推理，然后将结果放入结果队列。"""
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
                    infer_tensors = stack_to_original(infer_tensors, ori_tensor.copy())

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
    args = parse_args()
    hef_path = get_hef_path(args.strength)
    if not os.path.isfile(hef_path):
        raise FileNotFoundError(f"HEF 不存在: {hef_path}")

    mp.set_start_method("spawn", force=True)

    print(f"启动实时显示模式 (低延迟, 丢帧优先) | 去噪强度: {args.strength} | HEF: {hef_path}")

    # ---------------- 启动 worker 进程 ----------------
    task_queues = [mp.Queue(maxsize=QUEUE_MAX_SIZE) for _ in range(NUM_DEVICES)]
    result_queue = mp.Queue(maxsize=QUEUE_MAX_SIZE * NUM_DEVICES * 2)

    processes = []
    for dev_id in range(NUM_DEVICES):
        p = mp.Process(target=worker_process, args=(dev_id, task_queues[dev_id], result_queue, hef_path))
        p.start()
        processes.append(p)
    time.sleep(1.0)

    # ---------------- 打开摄像头 ----------------
    cap = cv2.VideoCapture(CAMERA_DEVICE_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {CAMERA_DEVICE_PATH}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_RESOLUTION[0])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, 60)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_cam_fps:.1f}fps")

    batch_index = 0
    result_cache = {}
    latest_infer_frame = None
    latest_original_frame = None
    last_display_original = None  # 上次显示用的原图，用于 latest_original_frame 为 None 时回退
    last_display_time = time.time()
    ema_fps = None
    alpha = 0.1  # EMA smoothing for fps
    grab_buffer = []  # 缓存 latest_original_frame，用于按 g 键抓拍

    read_start = time.time()

    print("按 q 键退出。")
    show_idx = 0
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    try:
        while True:
            if RUN_DURATION is not None and (time.time() - read_start) > RUN_DURATION:
                print("达到运行时长，退出")
                break

            # 1) 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            # 预处理（快速）
            frame_processed, _, frame_original = process_frame(frame)

            batch_tensor = np.expand_dims(frame_processed, axis=0)
            actual_batch_size = 1

            target_dev = batch_index % NUM_DEVICES

            try:
                task_queues[target_dev].put_nowait((batch_index, actual_batch_size, batch_tensor))
                result_cache[batch_index] = frame_original
                batch_index += 1
            except Exception:
                try:
                    batch_index, _, _ = task_queues[target_dev].get_nowait()
                    if batch_index in result_cache:
                        del result_cache[batch_index]
                    task_queues[target_dev].put_nowait((batch_index, actual_batch_size, batch_tensor))
                    result_cache[batch_index] = frame_original
                    batch_index += 1
                except Exception:
                    pass

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

                    latest_original_frame = result_cache.pop(batch_idx_res, None)
                    latest_infer_frame = infer_frame_bgr

                    # 维护 latest_original_frame 的 buffer（最多 4 帧）
                    if latest_original_frame is not None:
                        grab_buffer.append(latest_original_frame.copy())
                        if len(grab_buffer) > 4:
                            grab_buffer.pop(0)

                    try:
                        orig_for_display = latest_original_frame if latest_original_frame is not None else last_display_original
                        if orig_for_display is not None:
                            if latest_original_frame is not None:
                                last_display_original = latest_original_frame.copy()
                            h1, w1 = orig_for_display.shape[:2]
                            h2, w2 = latest_infer_frame.shape[:2]
                            infer_resized = cv2.resize(latest_infer_frame, (w1, h1)) if (h1, w1) != (h2, w2) else latest_infer_frame
                            combined = np.concatenate((orig_for_display, infer_resized), axis=1)
                            cv2.imshow(WINDOW_NAME, combined)
                        else:
                            cv2.imshow(WINDOW_NAME, latest_infer_frame)

                        # FPS 统计与打印
                        now = time.time()
                        frame_count += 1
                        delta = now - last_display_time
                        last_display_time = now
                        instant_fps = 1.0 / delta if delta > 0 else 0.0
                        ema_fps = (alpha * instant_fps + (1 - alpha) * ema_fps) if ema_fps is not None else instant_fps
                        if now - last_print_time >= 1.0:
                            print(f"FPS: {min(ema_fps, 60):.1f} (display) | infer: {infer_time*1000:.0f}ms")
                            last_print_time = now

                    except Exception as e:
                        print(f"[Display warning] failed to display combined frame: {e}")

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User requested exit.")
                        raise KeyboardInterrupt
                    elif key == ord('g'):
                        if latest_original_frame is not None and latest_infer_frame is not None:
                            base_dir = os.path.dirname(os.path.abspath(__file__))
                            dir_original = os.path.join(base_dir, "grab", "original")
                            dir_denoised = os.path.join(base_dir, "grab", "denoised")
                            os.makedirs(dir_original, exist_ok=True)
                            os.makedirs(dir_denoised, exist_ok=True)
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            name = f"grab_{timestamp}.png"
                            path_original = os.path.join(dir_original, name)
                            path_denoised = os.path.join(dir_denoised, name)
                            cv2.imwrite(path_original, latest_original_frame)
                            cv2.imwrite(path_denoised, latest_infer_frame)
                            print("Captured. Processing will start after the real-time denoising program ends.")
                        else:
                            print("No frame to capture.")

                except queue.Empty:
                    break

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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        process_pending_denoised_with_enhance(base_dir)
        print("Cleaned up and exiting.")

if __name__ == "__main__":
    main()
