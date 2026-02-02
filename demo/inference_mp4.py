import os
import time
import cv2
import numpy as np
from tqdm import tqdm
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
import argparse


# -------------------------- 核心配置参数 --------------------------
# 输入视频参数
TARGET_RESOLUTION = (960, 720)       # 需与模型输入匹配

# 推理参数（单设备串行，无需多设备配置）
HEF_PATH = "dncnn_lite_rgb_376x496_alpha0_8.hef"
DEVICE_ID = 0                        # 使用单个Hailo设备（根据实际设备ID调整）

# 输出视频参数
VIDEO_CODEC = cv2.VideoWriter_fourcc(*"mp4v")
VIDEO_EXT = ".mp4"
SPLIT_LINE_COLOR = (0, 255, 0)        # 绿色分割线
SPLIT_LINE_WIDTH = 2
TEXT_COLOR = (255, 255, 255)          # 白色文字
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 0.8
TEXT_THICKNESS = 2


# -------------------------- 帧预处理 --------------------------
def process_frame(frame):
    """处理视频BGR帧为模型输入格式（RGB+float32）"""
    # 调整分辨率
    frame_resized = cv2.resize(
        frame, 
        dsize=TARGET_RESOLUTION, 
        interpolation=cv2.INTER_LANCZOS4
    )
    
    # BGR转RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # 格式转换为float32
    frame_float = frame_rgb.astype(np.float32)
    
    return frame_float, frame_resized  # 返回预处理帧（推理用）和原始调整帧（拼接用）


# -------------------------- 推理结果后处理 --------------------------
def postprocess_infer_result(infer_tensor):
    """将推理输出张量转换为BGR帧（适配OpenCV）"""
    # 移除多余维度
    tensor_squeezed = np.squeeze(infer_tensor)
    
    # CHW → HWC转换
    if tensor_squeezed.shape[0] in [3, 1]:
        frame_hwc = tensor_squeezed.transpose(1, 2, 0)
    else:
        frame_hwc = tensor_squeezed
    
    # 单通道转三通道
    if frame_hwc.shape[-1] == 1:
        frame_hwc = np.repeat(frame_hwc, 3, axis=-1)
    
    # 分辨率校准
    frame_resized = cv2.resize(
        frame_hwc, 
        dsize=TARGET_RESOLUTION, 
        interpolation=cv2.INTER_LANCZOS4
    )
    
    # RGB→BGR
    frame_bgr = cv2.cvtColor(frame_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return frame_bgr


# -------------------------- 帧分割与拼接（适配4分块模型） --------------------------
def split_and_stack(frame_tensor):
    """将[720, 960, 3]的单帧分割为4个[376, 496, 3]子图，堆叠为[4, 376, 496, 3]"""
    if frame_tensor.shape != (720, 960, 3):
        raise ValueError(f"输入形状必须为[720, 960, 3]，实际为{frame_tensor.shape}")
    
    # 分块参数（带16像素padding）
    sub_height = 720 // 2 + 16  # 376
    sub_width = 960 // 2 + 16   # 496
    
    # 分割4个子图
    sub1 = frame_tensor[:sub_height, :sub_width, :]          # 左上
    sub2 = frame_tensor[-sub_height:, :sub_width, :]         # 左下
    sub3 = frame_tensor[:sub_height, -sub_width:, :]         # 右上
    sub4 = frame_tensor[-sub_height:, -sub_width:, :]        # 右下
    
    # 堆叠为批次格式（适配模型输入）
    return np.stack([sub1, sub2, sub3, sub4], axis=0)


def stack_to_original(sub_images):
    """将[4, 376, 496, 3]的子图拼接回原始[720, 960, 3]帧"""
    if sub_images.shape != (4, 376, 496, 3):
        raise ValueError(f"输入形状必须为[4, 376, 496, 3]，实际为{sub_images.shape}")
    
    sub1, sub2, sub3, sub4 = sub_images[0], sub_images[1], sub_images[2], sub_images[3]
    
    # 去除padding
    sub1_crop = sub1[:360, :480, :]
    sub2_crop = sub2[16:, :480, :]
    sub3_crop = sub3[:360, 16:, :]
    sub4_crop = sub4[16:, 16:, :]
    
    # 拼接
    top_row = np.concatenate([sub1_crop, sub3_crop], axis=1)
    bottom_row = np.concatenate([sub2_crop, sub4_crop], axis=1)
    full_image = np.concatenate([top_row, bottom_row], axis=0)
    
    return full_image


# -------------------------- 帧拼接与标注 --------------------------
def stitch_frames(original_frame, infer_frame, process_fps):
    """横向拼接原始帧与推理帧，并添加标注"""
    # 横向拼接
    stitched_frame = cv2.hconcat([original_frame, infer_frame])
    
    # 添加分割线
    split_x = original_frame.shape[1]
    cv2.line(
        stitched_frame, 
        (split_x, 0), 
        (split_x, stitched_frame.shape[0]), 
        SPLIT_LINE_COLOR, 
        SPLIT_LINE_WIDTH
    )
    
    # 添加文字标注
    cv2.putText(stitched_frame, "Original Frame", (20, 40),
                TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS)
    cv2.putText(stitched_frame, "Inferred Frame", (split_x + 20, 40),
                TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS)
    
    return stitched_frame


# -------------------------- Hailo设备初始化（单设备） --------------------------
def init_single_device(hef_path, device_id):
    """初始化单个Hailo设备并加载模型"""
    # 扫描可用设备
    device_ids = Device.scan()
    if len(device_ids) <= device_id:
        raise RuntimeError(f"设备ID {device_id} 不存在，仅检测到 {len(device_ids)} 个Hailo设备")
    
    print(f"初始化设备 {device_id}（硬件ID: {device_ids[device_id]}）...")
    
    # 创建设备参数
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
    
    # 创建输入/输出流参数
    input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
    
    # 获取流信息
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    
    device_info = {
        "target": target,
        "network_group": network_group,
        "network_group_params": network_group_params,
        "input_vstreams_params": input_vstreams_params,
        "output_vstreams_params": output_vstreams_params,
        "input_vstream_info": input_vstream_info,
        "output_vstream_info": output_vstream_info
    }
    
    print(f"设备初始化完成 | 输入形状: {input_vstream_info.shape} | 输出形状: {output_vstream_info.shape}")
    return device_info


# -------------------------- 单帧推理函数 --------------------------
def run_single_frame_inference(device, frame_tensor):
    """对单帧进行推理（含分块与拼接）"""
    # 1. 帧分块（适配4分块模型）
    split_tensor = split_and_stack(frame_tensor)
    
    # 2. 执行推理
    network_group = device["network_group"]
    input_vstreams_params = device["input_vstreams_params"]
    output_vstreams_params = device["output_vstreams_params"]
    network_group_params = device["network_group_params"]
    input_vstream_info = device["input_vstream_info"]
    
    start_time = time.time()
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            input_data = {input_vstream_info.name: split_tensor}
            infer_results = infer_pipeline.infer(input_data)
    infer_time = time.time() - start_time
    
    # 3. 推理结果拼接回原始尺寸
    infer_tensor = infer_results[device["output_vstream_info"].name]
    original_infer_frame = stack_to_original(infer_tensor)
    
    return original_infer_frame, infer_time

def parse_args():
    parser = argparse.ArgumentParser(description="Video inference settings")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input video file (e.g. /path/to/video.mp4)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/inference_videos",
        help="Directory to save inference results (default: output/inference_videos)"
    )
    args = parser.parse_args()
    return args



# -------------------------- 主函数（串行处理逻辑） --------------------------
def main():
    args = parse_args()
    INPUT_VIDEO_PATH = args.input
    SAVE_VIDEO_DIR = args.save_dir
    
    total_start_time = time.time()
    print("="*60)
    print(f"开始串行视频处理 | 目标分辨率: {TARGET_RESOLUTION[0]}×{TARGET_RESOLUTION[1]}")
    print(f"使用设备ID: {DEVICE_ID} | 模型路径: {HEF_PATH}")
    print(f"输入视频: {INPUT_VIDEO_PATH} | 输出目录: {SAVE_VIDEO_DIR}")
    print("="*60)

    # 1. 检查输入视频
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(f"输入视频文件不存在: {INPUT_VIDEO_PATH}")

    # 2. 初始化输出目录
    os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
    
    # 3. 打开输入视频
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入视频: {INPUT_VIDEO_PATH}")
    
    # 4. 获取输入视频信息
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n输入视频信息 | 分辨率: {input_width}×{input_height} | FPS: {input_fps:.1f} | 总帧数: {total_frames}")

    # 5. 初始化视频写入器
    video_filename = f"serial_infer_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}{VIDEO_EXT}"
    video_save_path = os.path.join(SAVE_VIDEO_DIR, video_filename)
    stitched_resolution = (TARGET_RESOLUTION[0] * 2, TARGET_RESOLUTION[1])  # 横向拼接
    
    video_writer = cv2.VideoWriter(
        video_save_path,
        VIDEO_CODEC,
        input_fps,
        stitched_resolution
    )
    
    if not video_writer.isOpened():
        raise RuntimeError(f"无法初始化视频写入器: {video_save_path}")
    print(f"输出视频信息 | 分辨率: {stitched_resolution[0]}×{stitched_resolution[1]} | 保存路径: {video_save_path}")

    # 6. 初始化Hailo设备
    device = init_single_device(HEF_PATH, DEVICE_ID)

    # 7. 初始化统计变量
    processed_frames = 0
    total_infer_time = 0.0
    start_process_time = time.time()

    # 8. 串行处理：读取→推理→写入
    print(f"\n开始串行处理（共{total_frames}帧）...")
    progress = tqdm(total=total_frames, desc="处理进度")
    
    while True:
        # 步骤1：读取一帧
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕
        
        # 步骤2：预处理帧
        frame_processed, frame_original = process_frame(frame)
        
        # 步骤3：单帧推理（含分块与拼接）
        infer_frame_tensor, infer_time = run_single_frame_inference(device, frame_processed)
        total_infer_time += infer_time
        
        # 步骤4：后处理推理结果
        infer_frame = postprocess_infer_result(infer_frame_tensor)
        
        # 步骤5：计算当前处理FPS
        elapsed_time = time.time() - start_process_time
        current_fps = processed_frames / elapsed_time if elapsed_time > 1e-3 else 0.0
        
        # 步骤6：拼接帧并写入视频
        stitched_frame = stitch_frames(frame_original, infer_frame, current_fps)
        video_writer.write(stitched_frame)
        
        # 步骤7：更新统计与进度
        processed_frames += 1
        progress.update(1)

    # 9. 释放资源
    progress.close()
    video_writer.release()
    cap.release()
    device["target"].release()  # 释放Hailo设备
    cv2.destroyAllWindows()

    # 10. 输出统计结果
    total_process_time = time.time() - total_start_time
    avg_infer_time = total_infer_time / processed_frames if processed_frames > 0 else 0.0
    avg_process_fps = processed_frames / total_process_time if total_process_time > 0 else 0.0

    print(f"\n✅ 串行视频处理完成！")
    print("\n" + "="*60)
    print("📊 处理统计结果")
    print("="*60)
    print(f"1. 基础信息")
    print(f"   - 总处理帧数: {processed_frames} 帧")
    print(f"   - 总处理时间: {total_process_time:.2f} 秒")
    print(f"   - 平均处理FPS: {avg_process_fps:.2f} fps")
    print(f"\n2. 推理性能")
    print(f"   - 总推理耗时: {total_infer_time:.2f} 秒")
    print(f"   - 单帧平均推理时间: {avg_infer_time:.3f} 秒")
    print(f"\n3. 输出信息")
    print(f"   - 输出视频路径: {video_save_path}")
    print(f"   - 输出视频分辨率: {stitched_resolution[0]}×{stitched_resolution[1]}")
    print("="*60)


if __name__ == "__main__":
    main()