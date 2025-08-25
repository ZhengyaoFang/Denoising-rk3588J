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
output_dir = '/home/firefly/Denoising-rk3588J/output/20250113_output_dncnn_v0'  # 并行处理输出目录
hef_path = '/home/firefly/Denoising-rk3588J/models/dncnn_v0/dncnn_bs3.hef'  # Hailo模型路径
batch_size = 2  # 每个设备的批次大小
input_shape = (3, 720, 960)  # (channel, height, width)
NUM_DEVICES = 2  # 使用两个加速棒


# -------------------------- 图像预处理 --------------------------
def load_image(filepath):
    """加载并预处理图像"""
    start_time = time.time()
    # 加载图像并转为RGB
    image = Image.open(filepath).convert('RGB')
    # 调整尺寸为(960, 720)
    image = image.resize((input_shape[2], input_shape[1]), Image.LANCZOS)
    # 转为numpy数组（HWC格式，float32类型）
    image_np = np.array(image, dtype=np.float32)
    load_time = time.time() - start_time
    return image_np, load_time


# -------------------------- 图像保存 --------------------------
def save_image(tensor, filepath):
    """保存处理后的图像"""
    start_time = time.time()
    # 移除所有尺寸为1的多余维度
    tensor = np.squeeze(tensor)
    
    # 如果是CHW格式，转换为HWC格式
    if tensor.shape[0] in [3, 1]:
        tensor = tensor.transpose(1, 2, 0)
    
    # 确保通道数正确
    if tensor.shape[-1] == 1:
        tensor = np.repeat(tensor, 3, axis=-1)
    
    # 裁剪或调整尺寸至目标形状
    if tensor.shape[:2] != (720, 960):
        tensor = tensor[:720, :960, :]
    
    # 转换为uint8并保存
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(filepath)
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
                
            batch_tensor, output_paths, batch_index = task
            
            # 执行推理
            output_tensor, infer_time = run_inference(device, batch_tensor)
            
            # 保存结果并统计保存时间
            save_times = []
            for i in range(len(output_paths)):
                save_time = save_image(output_tensor[i], output_paths[i])
                save_times.append(save_time)
            
            avg_save_time = np.mean(save_times) if save_times else 0
                
            # 将结果返回给主进程
            result_queue.put((batch_index, len(output_paths), infer_time, avg_save_time))
            
    except Exception as e:
        print(f"设备 {device_id} 工作进程出错: {str(e)}")
    finally:
        # 释放设备资源
        if 'device' in locals():
            device['target'].release()
        print(f"设备 {device_id} 工作进程已退出")


# -------------------------- 主流程 --------------------------
def main():
    # 记录总开始时间（包括所有阶段）
    total_start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出结果将保存至：{output_dir}")

    # 获取所有图像文件并统计加载时间
    start_time = time.time()
    image_files = [f for f in sorted(os.listdir(noisy_dir)) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = len(image_files)
    print(f"发现 {total_images} 张图像，开始并行推理（每个设备批次大小：{batch_size}，设备数量：{NUM_DEVICES}）...")

    # 预处理所有图像并统计时间
    device_batches = [[] for _ in range(NUM_DEVICES)]
    device_output_paths = [[] for _ in range(NUM_DEVICES)]
    total_load_time = 0
    
    for idx, img_name in enumerate(tqdm(image_files, desc="加载图像")):
        img_path = os.path.join(noisy_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        # 预处理图像并统计时间
        img_np, load_time = load_image(img_path)
        total_load_time += load_time
        
        # 轮流向两个设备分配任务
        device_id = idx % NUM_DEVICES
        device_batches[device_id].append(img_np)
        device_output_paths[device_id].append(output_path)
    
    preprocessing_time = time.time() - start_time
    print(f"图像加载与预处理完成，耗时: {preprocessing_time:.4f} 秒")

    # 为每个设备创建任务队列
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

    # 记录推理处理开始时间
    inference_start_time = time.time()
    
    
    # 分配任务给设备
    batches = []
    batch_index = 0
    print("Start time!")
    for device_id in range(NUM_DEVICES):
        # 分割批次并发送到队列
        for i in range(0, len(device_batches[device_id]), batch_size):
            end_idx = min(i + batch_size, len(device_batches[device_id]))
            batch_imgs = device_batches[device_id][i:end_idx]
            batch_paths = device_output_paths[device_id][i:end_idx]
            
            # 如果批次不足，补零
            if len(batch_imgs) < batch_size:
                pad_size = batch_size - len(batch_imgs)
                batch_imgs += [np.zeros_like(batch_imgs[0]) for _ in range(pad_size)]
            
            batch_tensor = np.stack(batch_imgs, axis=0)
            task_queues[device_id].put((batch_tensor, batch_paths, batch_index))
            batches.append((device_id, batch_index))
            batch_index += 1

    # 发送终止信号
    for q in task_queues:
        q.put(None)

    # 收集结果并显示进度
    total_batches = len(batches)
    processed_batches = 0
    total_infer_time = 0
    total_save_time = 0
    progress_bar = tqdm(total=total_images, desc="整体进度")
    
    while processed_batches < total_batches:
        batch_index, num_images, infer_time, avg_save_time = result_queue.get()
        total_infer_time += infer_time
        total_save_time += avg_save_time * num_images
        processed_batches += 1
        progress_bar.update(num_images)

    progress_bar.close()

    # 等待所有进程完成
    for p in processes:
        p.join()
        print(f"设备进程 {p.pid} 已完成")
    print("End time!")
    # 计算各阶段时间
    inference_end_time = time.time()
    total_inference_wall_time = inference_end_time - inference_start_time  # 推理阶段的实际耗时（墙钟时间）
    total_time = inference_end_time - total_start_time  # 总耗时（包括所有阶段）
    
    # 计算设备利用率
    device_utilization = (total_infer_time / (NUM_DEVICES * total_inference_wall_time)) * 100

    # 统计性能指标
    if total_images > 0:
        print(f"\n===== 性能统计 =====")
        print(f"总处理图像数量: {total_images} 张")
        print(f"图像加载与预处理时间: {preprocessing_time:.4f} 秒")
        print(f"推理计算总时间（累加）: {total_infer_time:.4f} 秒")
        print(f"推理阶段实际耗时（墙钟时间）: {total_inference_wall_time:.4f} 秒")
        print(f"图像保存总时间: {total_save_time:.4f} 秒")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"平均单张图像处理时间: {total_time / total_images:.4f} 秒")
        print(f"设备平均利用率: {device_utilization:.2f}%")
        print(f"并行加速比: {(total_infer_time / NUM_DEVICES) / total_inference_wall_time:.2f}x")


if __name__ == "__main__":
    # 在Windows系统上需要添加这一行
    mp.set_start_method('spawn')
    main()
    