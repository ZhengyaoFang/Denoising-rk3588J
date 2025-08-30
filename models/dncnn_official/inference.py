import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
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

# -------------------------- 配置参数 --------------------------
noisy_dir = '/home/firefly/Denoising/data/20250113'  # 输入噪声图像目录
output_dir = '/home/firefly/Denoising-rk3588J/output/20250113_output_dncnn_color_blind'  # 输出目录
hef_path = '/home/firefly/Denoising-rk3588J/models/dncnn_official/dncnn_color_blind.hef'  # 模型路径
batch_size = 4  # 批次大小固定为4（对应4个子图像）
model_input_shape = (3, 321, 481)  # 模型输入形状 (channel, height, width)
original_image_shape = (720, 960)  # 原始图像形状 (height, width)
NUM_DEVICES = 2  # 使用的设备数量


# -------------------------- 图像预处理与分块 --------------------------
def load_and_split_image(filepath):
    """加载图像并分割为4块321x481的子图像"""
    start_time = time.time()
    
    # 加载图像并转为RGB
    image = Image.open(filepath).convert('RGB')
    # 确保图像尺寸为720x960
    image = image.resize((original_image_shape[1], original_image_shape[0]), Image.LANCZOS)
    image_np = np.array(image, dtype=np.float32)  # 形状为 (720, 960, 3)
    
    # 分割为4块：(321, 481, 3) x 4
    # 注意：321*2=642 < 720，481*2=962 < 960，会有少量重叠用于后续拼接
    h, w = model_input_shape[1], model_input_shape[2]
    sub_images = []
    
    # 计算分割区域（带重叠）
    sub_images.append(image_np[:h, :w, :])                  # 左上
    sub_images.append(image_np[:h, original_image_shape[1]-w:, :])  # 右上
    sub_images.append(image_np[original_image_shape[0]-h:, :w, :])  # 左下
    sub_images.append(image_np[original_image_shape[0]-h:, original_image_shape[1]-w:, :])  # 右下
    
    # 转换为模型输入格式 (B, C, H, W)
    sub_images = [np.transpose(img, (2, 0, 1)) for img in sub_images]
    batch_tensor = np.stack(sub_images, axis=0)  # 形状为 (4, 3, 321, 481)
    
    load_time = time.time() - start_time
    return batch_tensor, load_time


# -------------------------- 图像拼接与保存 --------------------------
def merge_and_save_image(output_tensor, filepath):
    """将4块推理结果拼接为720x960图像并保存"""
    start_time = time.time()
    
    # 确保输出是4个子图像
    assert output_tensor.shape[0] == 4, f"预期4个子图像，实际得到{output_tensor.shape[0]}个"
    
    # 转换为HWC格式并裁剪到模型输出尺寸
    sub_images = []
    for i in range(4):
        # 如果是CHW格式，转换为HWC
        if output_tensor.shape[1] == 3:
            img = np.transpose(output_tensor[i], (1, 2, 0))
        else:
            img = output_tensor[i]
        # 确保尺寸正确
        img = img[:model_input_shape[1], :model_input_shape[2], :]
        sub_images.append(img)
    
    # 创建空白画布
    merged = np.zeros((original_image_shape[0], original_image_shape[1], 3), dtype=np.float32)
    
    h, w = model_input_shape[1], model_input_shape[2]
    h_original, w_original = original_image_shape
    
    # 放置子图像并处理重叠区域（取平均值）
    # 左上
    merged[:h, :w, :] += sub_images[0]
    # 右上
    merged[:h, w_original-w:, :] += sub_images[1]
    # 左下
    merged[h_original-h:, :w, :] += sub_images[2]
    # 右下
    merged[h_original-h:, w_original-w:, :] += sub_images[3]
    
    # 处理重叠区域的平均值计算
    # 顶部中间重叠
    merged[:h, w:w_original-w, :] /= 1  # 无重叠
    # 左侧中间重叠
    merged[h:h_original-h, :w, :] /= 1  # 无重叠
    # 右侧中间重叠
    merged[h:h_original-h, w_original-w:, :] /= 1  # 无重叠
    # 底部中间重叠
    merged[h_original-h:, w:w_original-w, :] /= 1  # 无重叠
    # 四个角重叠区
    merged[:h, :w, :] /= 1  # 仅左上覆盖
    merged[:h, w_original-w:, :] /= 1  # 仅右上覆盖
    merged[h_original-h:, :w, :] /= 1  # 仅左下覆盖
    merged[h_original-h:, w_original-w:, :] /= 1  # 仅右下覆盖
    
    # 中心区域（如果有）
    center_h_start, center_h_end = h, h_original - h
    center_w_start, center_w_end = w, w_original - w
    if center_h_start < center_h_end and center_w_start < center_w_end:
        merged[center_h_start:center_h_end, center_w_start:center_w_end, :] /= 1  # 无重叠
    
    # 转换为uint8并保存
    merged = np.clip(merged, 0, 255).astype(np.uint8)
    Image.fromarray(merged).save(filepath)
    
    save_time = time.time() - start_time
    return save_time


# -------------------------- 设备初始化 --------------------------
def init_device(hef_path, device_id):
    """初始化单个Hailo设备并加载模型"""
    # 检测可用设备
    device_ids = Device.scan()
    if len(device_ids) <= device_id:
        raise RuntimeError(f"设备ID {device_id} 不存在，仅检测到 {len(device_ids)} 个设备")
    
    print(f"初始化设备 {device_id}（ID: {device_ids[device_id]}）...")
    
    # 创建设备参数
    vdevice_params = VDevice.create_params()
    vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
    vdevice_params.device_ids.append(device_id)
    
    target = VDevice(params=vdevice_params)
    
    # 加载HEF模型
    hef = HEF(hef_path)
    
    # 配置网络组（PCIe接口）
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()
    
    # 创建输入/输出流参数
    input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
    
    # 获取输入/输出流信息
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    
    device_info = {
        'target': target,
        'hef': hef,
        'network_group': network_group,
        'network_group_params': network_group_params,
        'input_vstreams_params': input_vstreams_params,
        'output_vstreams_params': output_vstreams_params,
        'input_vstream_info': input_vstream_info,
        'output_vstream_info': output_vstream_info,
        'device_id': device_id
    }
    
    print(f"设备 {device_id} 初始化完成，输入形状：{input_vstream_info.shape}，输出形状：{output_vstream_info.shape}")
    return device_info


# -------------------------- 推理函数 --------------------------
def run_inference(device, input_batch):
    """在单个设备上运行推理"""
    network_group = device['network_group']
    input_vstreams_params = device['input_vstreams_params']
    output_vstreams_params = device['output_vstreams_params']
    network_group_params = device['network_group_params']
    input_vstream_info = device['input_vstream_info']
    
    # 执行推理并计时
    start_time = time.time()
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            # 输入数据字典
            input_data = {input_vstream_info.name: input_batch}
            # 推理
            infer_results = infer_pipeline.infer(input_data)
    
    inference_time = time.time() - start_time
    output_tensor = infer_results[device['output_vstream_info'].name]
    
    return output_tensor, inference_time


# -------------------------- 工作进程函数 --------------------------
def worker_process(device_id, task_queue, result_queue, hef_path):
    """工作进程，处理分配给特定设备的任务"""
    try:
        # 初始化设备
        device = init_device(hef_path, device_id)
        
        # 处理任务队列中的任务
        while True:
            task = task_queue.get()
            
            # 检查是否是终止信号
            if task is None:
                break
                
            batch_tensor, output_path, batch_index = task
            
            # 执行推理
            output_tensor, infer_time = run_inference(device, batch_tensor)
            
            # 拼接并保存结果
            save_time = merge_and_save_image(output_tensor, output_path)
                
            # 将结果返回给主进程
            result_queue.put((batch_index, 1, infer_time, save_time))
            
    except Exception as e:
        print(f"设备 {device_id} 工作进程出错: {str(e)}")
    finally:
        # 释放设备资源
        if 'device' in locals():
            device['target'].release()
        print(f"设备 {device_id} 工作进程已退出")


# -------------------------- 主流程 --------------------------
def main():
    # 记录总开始时间
    total_start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出结果将保存至：{output_dir}")

    # 获取所有图像文件
    start_time = time.time()
    image_files = [f for f in sorted(os.listdir(noisy_dir)) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = len(image_files)
    print(f"发现 {total_images} 张图像，开始推理（每个设备批次大小：{batch_size}，设备数量：{NUM_DEVICES}）...")

    # 预处理所有图像并分配任务
    device_tasks = [[] for _ in range(NUM_DEVICES)]
    total_load_time = 0
    
    for idx, img_name in enumerate(tqdm(image_files, desc="加载和分割图像")):
        img_path = os.path.join(noisy_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        # 加载并分割图像
        batch_tensor, load_time = load_and_split_image(img_path)
        total_load_time += load_time
        
        # 轮流向设备分配任务
        device_id = idx % NUM_DEVICES
        device_tasks[device_id].append((batch_tensor, output_path))
    
    preprocessing_time = time.time() - start_time
    print(f"图像加载与分割完成，耗时: {preprocessing_time:.4f} 秒")

    # 创建任务队列和结果队列
    task_queues = [mp.Queue() for _ in range(NUM_DEVICES)]
    result_queue = mp.Queue()
    
    # 创建并启动工作进程
    processes = []
    for device_id in range(NUM_DEVICES):
        p = mp.Process(
            target=worker_process,
            args=(device_id, task_queues[device_id], result_queue, hef_path)
        )
        p.start()
        processes.append(p)
        print(f"启动设备 {device_id} 的工作进程，PID: {p.pid}")

    # 记录推理开始时间
    inference_start_time = time.time()
    
    # 分配任务给设备
    batch_index = 0
    for device_id in range(NUM_DEVICES):
        for task in device_tasks[device_id]:
            batch_tensor, output_path = task
            task_queues[device_id].put((batch_tensor, output_path, batch_index))
            batch_index += 1

    # 发送终止信号
    for q in task_queues:
        q.put(None)

    # 收集结果并显示进度
    total_batches = batch_index
    processed_batches = 0
    total_infer_time = 0
    total_save_time = 0
    progress_bar = tqdm(total=total_images, desc="整体进度")
    
    while processed_batches < total_batches:
        batch_index, num_images, infer_time, save_time = result_queue.get()
        total_infer_time += infer_time
        total_save_time += save_time
        processed_batches += 1
        progress_bar.update(num_images)

    progress_bar.close()

    # 等待所有进程完成
    for p in processes:
        p.join()
        print(f"设备进程 {p.pid} 已完成")

    # 计算各阶段时间
    inference_end_time = time.time()
    total_inference_wall_time = inference_end_time - inference_start_time
    total_time = inference_end_time - total_start_time
    
    # 计算设备利用率
    device_utilization = (total_infer_time / (NUM_DEVICES * total_inference_wall_time)) * 100

    # 统计性能指标
    if total_images > 0:
        print(f"\n===== 性能统计 =====")
        print(f"总处理图像数量: {total_images} 张")
        print(f"图像加载与分割时间: {preprocessing_time:.4f} 秒")
        print(f"推理计算总时间（累加）: {total_infer_time:.4f} 秒")
        print(f"推理阶段实际耗时（墙钟时间）: {total_inference_wall_time:.4f} 秒")
        print(f"图像拼接与保存总时间: {total_save_time:.4f} 秒")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"平均单张图像处理时间: {total_time / total_images:.4f} 秒")
        print(f"设备平均利用率: {device_utilization:.2f}%")
        print(f"并行加速比: {(total_infer_time / NUM_DEVICES) / total_inference_wall_time:.2f}x")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
