import cv2
import time
from encode import encode_video
from encode_rs import encoding

# ---------------------- 文件参数
# 输入源：1-视频，2-图片
input_from = 1
# 输入视频路径
video_input_path = '1.mp4'
# 输入图片路径
pic_input_path = ''
# 输出视频路径
video_output_path = ''  # .avi
# 输入字符串路径
file_input_path = 'data_input.txt'

# 每一个数据帧的有效数据长度
data_bit_per_frame = 2440

# ---------------------- debug用
# 是否输出视频
output_video = False
# 是否生成对比帧
output_right = False


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
def get_frame_from_video(cap):
    ret, frame = cap.read()
    img_resize = cv2.resize(frame, (1856, 1044))
    return img_resize


# 读取图片并缩放为[1856,1044]
def get_frame_from_pic(file):
    frame = cv2.imread(file)
    frame = cv2.resize(frame, (1856, 1044))
    return frame


# 用两帧生成视频
def twoframes2video(img1, img2, video_path, fps=60, time=5):
    imgInfo = img1.shape
    size = (imgInfo[1], imgInfo[0])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWrite = cv2.VideoWriter(video_path + '.avi', fourcc, fps, size)

    for i in range(0, fps * time):
        if i % 2 == 0:
            videoWrite.write(img1)
        else:
            videoWrite.write(img2)

    print('Video generate')


print('Start encoding')
time_start = time.time()

# ---------------------- 图像输入
if input_from == 1:
    # 视频提取帧
    cap = cv2.VideoCapture(video_input_path)
    frame1 = get_frame_from_video(cap)
    frame2 = get_frame_from_video(cap)
else:
    # 图片输入
    frame1 = get_frame_from_pic(pic_input_path)
    frame2 = get_frame_from_pic(pic_input_path)

# ---------------------- 数据输入
f = open(file_input_path)
s = f.read()

# 处理数据
seq_num = 1  # 每帧中数据的sequence number
seg_bin = bin(seq_num).replace('0b', '').zfill(16)  # sequence number转换为二进制并填充为16bit
seg_byte1 = int(seg_bin[0:8], 2)  # seq_num的第一个byte
seg_byte2 = int(seg_bin[8:16], 2)  # seq_num的第二个byte

# 获得数据
s_seg = s[1000:3440]  # 有效数据2440bit
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
preamble_data = '1010011011100110'

# ---------------------- 数据嵌入
frame1_out, frame2_out = encode_video(frame1, frame2, preamble_data, hidden_info)

# ---------------------- 输出
cv2.imwrite('e1.jpg', frame1_out)
cv2.imwrite('e2.jpg', frame2_out)

encode_time = time.time() - time_start
print('Encoding over, take time:' + str(encode_time))

# 用两帧生成视频
if output_video:
    twoframes2video(frame1_out, frame2_out, video_output_path, 60)

# 生成应接收的对比帧
if output_right:
    frame1 = frame1_out[9:1071, 16:1904]
    frame2 = frame2_out[9:1071, 16:1904]
    cv2.imwrite('d1.jpg', frame1)
    cv2.imwrite('d2.jpg', frame2)
