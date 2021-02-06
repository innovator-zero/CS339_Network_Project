import cv2
import time
from detect import detect
from decode_speed import decode_video
from decode_rs import decoding

# ---------------------- 文件参数
# 图片来源：1-视频提取，2-照片提取，3-直接加载
frame_from = 2
# 接收到的视频路径
video_rx_path = ''
# 接受的视频帧率
video_fps = 60
# 视频开头跳过帧数
skip_frame = 0
# 数据输出路径
file_output_path = 'output.txt'
# 解码用临时文件
tmp_file = 'tmp_data.txt'
# 解码参数
length = 16788


# ---------------------- 函数
# 将二进制数据转为字符串，并写入文件中
def get_str(str, file):
    with open(file, "w", encoding='utf-8') as f:
        for i in range(0, len(str), 8):
            # 以每8位为一组二进制，转换为十进制
            byte = int(str[i:i + 8], 2)

            # 将转换后的十进制数视为ascii码，再转换为字符串写入到文件中
            f.write(chr(byte))
            byte = ""


# 从视频中提取一帧，识别外层的黑白框，去掉最外层的黑框，并缩放为[1888,1062]
def get_frame_from_video(cap, file=None):
    ret, frame = cap.read()

    if file != None:
        cv2.imwrite(file, frame)  # 可以讲读取的帧保存下来

    frame_detect = detect(frame)
    frame_resize = cv2.resize(frame_detect, (1888, 1062))
    return frame_resize


# 从照片中提取一帧，识别外层的黑白框，去掉最外层的黑框，并缩放为[1888,1062]
def get_frame_from_pic(file):
    frame = cv2.imread(file)
    frame_detect = detect(frame)
    frame_resize = cv2.resize(frame_detect, (1888, 1062))
    return frame_resize


print('Start decoding')
time_start = time.time()

# ---------------------- 帧提取
if frame_from == 1:
    # 视频
    cap = cv2.VideoCapture(video_rx_path)
    # 可以跳过一些帧
    if skip_frame > 0:
        for i in range(skip_frame):
            cap.read()

    frame1 = get_frame_from_video(cap, 'v1.jpg')
    frame2 = get_frame_from_video(cap, 'v2.jpg')

elif frame_from == 2:
    # 图片
    frame1 = get_frame_from_pic('r1.jpg')
    frame2 = get_frame_from_pic('r2.jpg')
else:
    # 直接读
    frame1 = cv2.imread('d1.jpg')
    frame2 = cv2.imread('d2.jpg')

# 高斯低通滤波
frame1 = cv2.GaussianBlur(frame1, (3, 3), 0)
frame2 = cv2.GaussianBlur(frame2, (3, 3), 0)

# 显示提取的帧
cv2.imshow('f1', frame1)
cv2.imshow('f2', frame2)
cv2.waitKey()

# ---------------------- 解调
# 从两帧中解调出隐藏的数据，为数据序列和前导块
# 数据序列直接写入文件中，以便解码
preamble_data = decode_video(frame1, frame2, tmp_file, length)
print('preamble data:', preamble_data)

# ---------------------- 解码
# RS解码和卷积解码，得到2464bit
data_output = decoding(tmp_file, 30, 11, 3, 1, 5, 216, 184)

# 数据序列分块
seq_num = int(data_output[0:16], 2)  # 16bit sequence number
checksum = int(data_output[16:24], 2)  # 8bit checksum
s = data_output[24:]  # 2440bit data

# 将得到的数据写入文件中
with open(file_output_path, "w", encoding='utf-8') as f:
    f.write(s)
    f.close()

decode_time = time.time() - time_start
print("decode over, take time:" + str(decode_time))
