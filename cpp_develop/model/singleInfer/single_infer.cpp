#include "/home/firefly/Denoising-rk3588J/cpp_develop/hailort/hailort/libhailort/include/hailo/hailort.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <cstring>

using namespace hailort;

const std::string HEF_PATH = "/home/firefly/Denoising-rk3588J/demo/dncnn_80ep_l9_4split_16pad.hef";
const std::string INPUT_IMAGE = "/home/firefly/Denoising-rk3588J/data/20250113/20250113_0000.jpg";
const std::string OUTPUT_IMAGE = "/home/firefly/Denoising-rk3588J/cpp_develop/model/singleInfer/output_reconstructed.jpg";
const cv::Size TARGET_RESOLUTION(960, 720);

static inline cv::Rect ClampRect(const cv::Rect &r, int W, int H) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width,  W - x);
    int h = std::min(r.height, H - y);
    return cv::Rect(x, y, w, h);
}

// ---- 与 Python 完全一致的分块逻辑 ----
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

// ---- 与 Python 完全一致的拼接逻辑 ----
cv::Mat reconstruct_from_four(const std::vector<cv::Mat> &subs)
{
    cv::Mat sub1 = subs[0], sub2 = subs[1], sub3 = subs[2], sub4 = subs[3];
    cv::Mat top_row, bottom_row, full_image;

    cv::hconcat(sub1(cv::Rect(0, 0, 480, 360)), sub3(cv::Rect(16, 0, 480, 360)), top_row);
    cv::hconcat(sub2(cv::Rect(0, 16, 480, 360)), sub4(cv::Rect(16, 16, 480, 360)), bottom_row);
    cv::vconcat(top_row, bottom_row, full_image);
    return full_image;
}

// ---- 单块推理函数 ----
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

    // 准备输入数据
    std::map<std::string, std::vector<uint8_t>> input_data, output_data;
    for (auto &v : input_vstreams)
        input_data[v.get().name()] = std::vector<uint8_t>(v.get().get_frame_size());
    for (auto &v : output_vstreams)
        output_data[v.get().name()] = std::vector<uint8_t>(v.get().get_frame_size());

    // 拷贝输入图像
    const auto &first_input = input_vstreams.front().get();
    auto &input_buf = input_data[first_input.name()];
    std::memcpy(input_buf.data(), tile.data, std::min(input_buf.size(), (size_t)tile.total() * sizeof(float) * tile.channels()));

    // 建立 memoryview
    std::map<std::string, MemoryView> input_mem, output_mem;
    for (auto &v : input_vstreams)
        input_mem[v.get().name()] = MemoryView(input_data[v.get().name()].data(), input_data[v.get().name()].size());
    for (auto &v : output_vstreams)
        output_mem[v.get().name()] = MemoryView(output_data[v.get().name()].data(), output_data[v.get().name()].size());

    pipeline->infer(input_mem, output_mem, 1);

    // 解析输出
    const auto &first_output = output_vstreams.front().get();
    const auto &out_info = first_output.get_info();
    int h = out_info.shape.height;
    int w = out_info.shape.width;
    int c = out_info.shape.features;

    cv::Mat out_tile(h, w, CV_8UC3, output_data[first_output.name()].data());
    return out_tile.clone();
}

int main()
{
    cv::Mat input = cv::imread(INPUT_IMAGE);
    if (input.empty()) return -1;

    cv::resize(input, input, TARGET_RESOLUTION);
    input.convertTo(input, CV_32FC3);

    auto vdevice = VDevice::create();
    if (!vdevice) return -1;

    auto hef = Hef::create(HEF_PATH);
    if (!hef) return -1;

    auto configure_params = vdevice.value()->create_configure_params(hef.value());
    auto network_groups = vdevice.value()->configure(hef.value(), configure_params.value());
    auto network_group = network_groups->at(0);

    // 拆成 4 块
    auto tiles = split_into_four(input);

    // 逐块推理
    std::vector<cv::Mat> outputs;
    for (auto &t : tiles)
        outputs.push_back(infer_single_tile(t, network_group));

    // 拼接回原图
    cv::Mat merged = reconstruct_from_four(outputs);
    cv::imwrite(OUTPUT_IMAGE, merged);

    return 0;
}
