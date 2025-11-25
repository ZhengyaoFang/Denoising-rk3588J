#include "../../hailort/hailort/libhailort/include/hailo/hailort.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <cstring>
#include <iostream>

using namespace hailort;

const std::string HEF_PATH = "../../dncnn_80ep_l9_4split_16pad.hef";
const std::string INPUT_IMAGE = "../20250113_0000.jpg";
const std::string OUTPUT_IMAGE = "../output_reconstructed.jpg";
const cv::Size TARGET_RESOLUTION(960, 720);

static inline cv::Rect ClampRect(const cv::Rect &r, int W, int H) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width,  W - x);
    int h = std::min(r.height, H - y);
    return cv::Rect(x, y, w, h);
}

std::vector<cv::Mat> split_into_four(const cv::Mat &image)
{
    const int H = image.rows; // 720
    const int W = image.cols; // 960
    const int sub_h = H / 2;
    const int sub_w = W / 2;

    std::vector<cv::Mat> tiles(4);
    tiles[0] = image(ClampRect(cv::Rect(0, 0, sub_w + 16, sub_h + 16), W, H)).clone();
    tiles[1] = image(ClampRect(cv::Rect(0, sub_h - 16, sub_w + 16, sub_h + 16), W, H)).clone();
    tiles[2] = image(ClampRect(cv::Rect(sub_w - 16, 0, sub_w + 16, sub_h + 16), W, H)).clone();
    tiles[3] = image(ClampRect(cv::Rect(sub_w - 16, sub_h - 16, sub_w + 16, sub_h + 16), W, H)).clone();
    return tiles;
}

cv::Mat reconstruct_from_four(const std::vector<cv::Mat> &subs)
{
    cv::Mat sub1 = subs[0], sub2 = subs[1], sub3 = subs[2], sub4 = subs[3];
    cv::Mat top_row, bottom_row, full_image;

    cv::hconcat(sub1(cv::Rect(0, 0, 480, 360)), sub3(cv::Rect(16, 0, 480, 360)), top_row);
    cv::hconcat(sub2(cv::Rect(0, 16, 480, 360)), sub4(cv::Rect(16, 16, 480, 360)), bottom_row);
    cv::vconcat(top_row, bottom_row, full_image);
    return full_image;
}

cv::Mat infer_single_tile(const cv::Mat &tile, const std::shared_ptr<hailort::ConfiguredNetworkGroup> &network_group)
{
    auto input_vparams = network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32,
                                                                 HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                                 HAILO_DEFAULT_VSTREAM_QUEUE_SIZE, "");
    auto output_vparams = network_group->make_output_vstream_params(true, HAILO_FORMAT_TYPE_UINT8,
                                                                    HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
                                                                    HAILO_DEFAULT_VSTREAM_QUEUE_SIZE, "");

    auto pipeline = InferVStreams::create(*network_group, input_vparams.value(), output_vparams.value());
    auto input_vstreams = pipeline->get_input_vstreams();
    auto output_vstreams = pipeline->get_output_vstreams();

    std::map<std::string, std::vector<uint8_t>> input_data, output_data;
    for (auto &v : input_vstreams)
        input_data[v.get().name()] = std::vector<uint8_t>(v.get().get_frame_size());
    for (auto &v : output_vstreams)
        output_data[v.get().name()] = std::vector<uint8_t>(v.get().get_frame_size());

    const auto &first_input = input_vstreams.front().get();
    auto &input_buf = input_data[first_input.name()];
    std::memcpy(input_buf.data(), tile.data, std::min(input_buf.size(), (size_t)tile.total() * sizeof(float) * tile.channels()));

    std::map<std::string, MemoryView> input_mem, output_mem;
    for (auto &v : input_vstreams)
        input_mem[v.get().name()] = MemoryView(input_data[v.get().name()].data(), input_data[v.get().name()].size());
    for (auto &v : output_vstreams)
        output_mem[v.get().name()] = MemoryView(output_data[v.get().name()].data(), output_data[v.get().name()].size());

    pipeline->infer(input_mem, output_mem, 1);

    const auto &first_output = output_vstreams.front().get();
    const auto &out_info = first_output.get_info();
    int h = out_info.shape.height;
    int w = out_info.shape.width;
    int c = out_info.shape.features;

    cv::Mat out_tile(h, w, CV_8UC3, output_data[first_output.name()].data());
    return out_tile.clone();
}


// =====================================================================
//     封装接口函数：输入图像 + 设备号 + [可选输出宽高] → 输出图像
// =====================================================================
cv::Mat run_inference_on_device(
    const cv::Mat &input_bgr,
    int chosen_index,
    int output_width = 960,      // ✅ 默认输出宽度
    int output_height = 720      // ✅ 默认输出高度
)
{
    // ===== 1. 扫描并选择设备 =====
    auto devices_result = hailort::Device::scan();
    if (!devices_result)
        throw std::runtime_error("Device scan failed");

    auto device_ids = devices_result.value();
    if (device_ids.empty())
        throw std::runtime_error("No Hailo devices found");

    if (chosen_index < 0 || static_cast<size_t>(chosen_index) >= device_ids.size())
        throw std::runtime_error("Invalid device index");

    std::vector<std::string> selected_devices = { device_ids[chosen_index] };
    auto vdevice = hailort::VDevice::create(selected_devices);
    if (!vdevice)
        throw std::runtime_error("Failed to create VDevice for chosen device");

    // ===== 2. 图像预处理（缩放到模型输入尺寸）=====
    const cv::Size model_input_size(960, 720);  // 模型固定输入分辨率
    cv::Mat input;
    cv::resize(input_bgr, input, model_input_size);
    input.convertTo(input, CV_32FC3);

    // ===== 3. 加载 HEF 模型并配置网络 =====
    auto hef = Hef::create(HEF_PATH);
    if (!hef)
        throw std::runtime_error("Failed to load HEF file");

    auto configure_params = vdevice.value()->create_configure_params(hef.value());
    auto network_groups = vdevice.value()->configure(hef.value(), configure_params.value());
    auto network_group = network_groups->at(0);

    // ===== 4. 分块推理 + 拼接 =====
    auto tiles = split_into_four(input);
    std::vector<cv::Mat> outputs;
    for (auto &t : tiles)
        outputs.push_back(infer_single_tile(t, network_group));

    cv::Mat merged = reconstruct_from_four(outputs);

    // ===== 5. 输出后处理：resize 到指定宽高 =====
    if (merged.cols != output_width || merged.rows != output_height) {
        cv::Mat resized;
        cv::resize(merged, resized, cv::Size(output_width, output_height), 0, 0, cv::INTER_AREA);
        return resized;
    }

    return merged;
}



// =====================================================================
//                        main 示例：接口调用
// =====================================================================
int main()
{
    const std::string INPUT_IMAGE = "../20250113_0000.jpg";
    const std::string OUTPUT_IMAGE = "../output_interface.jpg";

    cv::Mat input = cv::imread(INPUT_IMAGE);
    if (input.empty()) {
        std::cerr << "Failed to read input image!" << std::endl;
        return -1;
    }

    int device_index = 0;  // 这里手动选择设备编号
    int width=100;
    int height=100;
    try {
        cv::Mat result = run_inference_on_device(input, device_index, width, height);
        cv::imwrite(OUTPUT_IMAGE, result);
        std::cout << "✅ Output saved to: " << OUTPUT_IMAGE << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
