import cv2
import time
from encode import encode_video
from encode_rs import encoding

# ---------------------- 文件参数
# 输入视频路径
video_input_path = 'video_input.mp4'
# 输出视频路径
video_output_path = 'video_demo'  # .avi
# 输入字符串路径
file_input_path = 'data_input.txt'

# ---------------------- 输出视频参数
# 视频分辨率
video_size = (1920, 1080)
# 视频帧率
video_fps = 60
# 视频时长
video_time = 5

# 每一个数据帧的有效数据长度
data_bit_per_frame = 2440


# ---------------------- 函数
# 读取文件到二进制字符串
def read_file(file):
    f = open(file, "rb")
    s = f.read()
    str = ""
    for i in range(len(s)):
        str = str + bin(s[i]).replace('0b', '').zfill(8)
    return str


# 从视频中读取一帧并缩放为[1856,1044]
def get_frame(cap):
    ret, frame = cap.read()
    img_resize = cv2.resize(frame, (1856, 1044))
    return img_resize


print('Start encoding')
time_start = time.time()

# 视频文件
video_input = cv2.VideoCapture(video_input_path)

# 数据文件
s = read_file(file_input_path)

# 生成的帧序列
frames_out = []
# 帧对的数量
frame_pairs = int(video_fps * video_time / 2)

# ---------------------- 视频帧生成
for i in range(frame_pairs):
    frame1 = get_frame(video_input)
    frame2 = get_frame(video_input)

    # ---------------------- 数据输入
    # 处理数据
    seq_num = i + 1  # 每帧中数据的sequence number
    seg_bin = bin(seq_num).replace('0b', '').zfill(16)  # sequence number转换为二进制并填充为16bit
    seg_byte1 = int(seg_bin[0:8], 2)  # seq_num的第一个byte
    seg_byte2 = int(seg_bin[8:16], 2)  # seq_num的第二个byte

    # 获得数据
    s_seg = s[data_bit_per_frame * i: data_bit_per_frame * (i + 1)]  # 有效数据2440bit
    num_bytes = int(len(s_seg) / 8)  # 有效数据305bytes

    # 生成checksum
    checksum = seg_byte1 ^ seg_byte2
    for j in range(num_bytes):
        s_byte = int(s_seg[8 * j:8 * j + 8], 2)
        checksum = checksum ^ s_byte

    # 组装为一个数据帧序列,总共2464bit
    # 16bit sequence number
    # 8bit checksum
    # 2440bit data
    data_input = seg_bin + bin(checksum).replace('0b', '').zfill(8) + s_seg

    # ---------------------- 数据编码
    # 进行RS编码和卷积编码，输出为16788bit
    encoded_len, hidden_info = encoding(data_input, 30, 11, 3, 1, 5, 64)

    # 前导块
    preamble_data = bin(encoded_len).replace('0b', '').zfill(16)

    # ---------------------- 数据嵌入
    frame1_out, frame2_out = encode_video(frame1, frame2, preamble_data, hidden_info)

    frames_out.append(frame1_out)
    frames_out.append(frame2_out)

    print('Encoding ' + str(i + 1) + '/' + str(frame_pairs))

encode_time = time.time() - time_start
time_start = time.time()
print('Encoding over, take time:' + str(encode_time))

# ---------------------- 视频生成
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWrite = cv2.VideoWriter(video_output_path + '.avi', fourcc, video_fps, video_size)

for i in range(len(frames_out)):
    videoWrite.write(frames_out[i])

video_gen_time = time.time() - time_start
print('Video generate, take time:' + str(video_gen_time))

print('Total time:', str(encode_time + video_gen_time))
