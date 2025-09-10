import os
import subprocess

def check_ffmpeg():
    """检查系统是否安装了ffmpeg"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_video_duration(input_file):
    """获取视频总时长（秒）"""
    try:
        # 使用ffprobe获取视频信息
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return float(result.stdout)
    except Exception as e:
        print(f"获取视频时长失败: {e}")
        return 0

def split_video(input_file, output_dir="output_segments", segment_duration=10):
    """
    使用ffmpeg将视频分割为指定时长的片段
    
    参数:
        input_file: 输入视频文件路径
        output_dir: 输出片段的保存目录
        segment_duration: 每个片段的时长(秒)，默认为10秒
    """
    # 检查ffmpeg是否安装
    if not check_ffmpeg():
        print("错误: 未找到ffmpeg。请先安装ffmpeg并确保它在系统PATH中。")
        return
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频总时长
    video_duration = get_video_duration(input_file)
    if video_duration <= 0:
        print("无法处理视频文件")
        return
        
    print(f"视频总时长: {video_duration:.2f}秒")
    
    # 计算需要分割的片段数量
    num_segments = int(video_duration // segment_duration)
    if video_duration % segment_duration > 0:
        num_segments += 1
    
    print(f"将分割为 {num_segments} 个片段")
    
    # 生成输出文件名基础
    input_filename = os.path.basename(input_file)
    filename, ext = os.path.splitext(input_filename)
    
    # 分割视频并保存
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, video_duration)
        duration = end_time - start_time
        
        # 生成输出路径
        output_filename = f"{filename}_segment_{i+1}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # 构建ffmpeg命令
        # -ss 指定开始时间
        # -t 指定持续时间
        # -c:v 和 -c:a 指定视频和音频编码器，使用copy模式避免重新编码，速度更快
        cmd = [
            "ffmpeg", "-y",  # -y 表示覆盖已存在的文件
            "-ss", str(start_time),
            "-i", input_file,
            "-t", str(duration),
            "-c:v", "copy",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            # 执行ffmpeg命令
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"已保存片段 {i+1}/{num_segments}: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"分割片段 {i+1} 失败: {e.stderr.decode()}")

if __name__ == "__main__":
    # 输入视频文件路径
    input_video = "/home/firefly/Denoising-rk3588J/data/20250703video/WIN_20250703_17_58_53_Pro.mp4"
    output_dir = "/home/firefly/Denoising-rk3588J/data/20250703video"
    
    # 调用分割函数，这里设置为60秒一个片段
    split_video(input_video, output_dir=output_dir, segment_duration=10)
    print("视频分割完成！")
    