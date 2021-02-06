# 代码功能：对给定的两帧连续帧图片嵌入图像代码，将01比特位嵌入图片中
# 给定输入：两帧的图片(size=1856*1044)，嵌入数据16800bit，前导块16bit
# 嵌入的data frame总大小为1830*1035，其中有效数据块为42(cell)*25(cell)*16(块)，每个cell大小为10*9
# 最终在处理后图像外层分别填充大小为16*9的白色像素和黑色像素。输出两帧图像大小为1920*1080

import cv2
import math
import numpy as np
from numba import jit
import warnings

warnings.filterwarnings("ignore")


# 计算图像对应像素点的纹理复杂度，进行纹理自适应的数据帧嵌入
# 使用numba模块进行逐像素处理的加速
@jit
def Contrast(m, n, img_lab):
    window_size = 5
    height = 1035  # 嵌入frame的高
    width = 1830  # 嵌入frame的宽

    # 确定滑动窗口的范围
    x_low = n - window_size // 2
    x_low = (x_low + abs(x_low)) // 2  # 为负数则令顶点x=0
    y_low = m - window_size // 2
    y_low = (y_low + abs(y_low)) // 2  # 为负数则令顶点y=0
    x_high = n + window_size // 2
    if x_high >= width:
        x_high = width - 1
    y_high = m + window_size // 2
    if y_high >= height:
        y_high = height - 1

    # 计算滑动窗口内的四近邻亮度差的平方和
    contrast_sum = 0
    for i in range(y_low + 1, y_high):
        for j in range(x_low + 1, x_high):
            contrast_sum = contrast_sum + pow(int(img_lab[i - 1][j][0]) - int(img_lab[i][j][0]), 2) + pow(
                int(img_lab[i + 1][j][0]) - int(img_lab[i][j][0]), 2) + pow(
                int(img_lab[i][j - 1][0]) - int(img_lab[i][j][0]), 2) + pow(
                int(img_lab[i][j + 1][0]) - int(img_lab[i][j][0]), 2)

    for i in range(y_low + 1, y_high):
        contrast_sum = contrast_sum + pow(int(img_lab[i - 1][x_low][0]) - int(img_lab[i][x_low][0]), 2) + pow(
            int(img_lab[i + 1][x_low][0]) - int(img_lab[i][x_low][0]), 2) + pow(
            int(img_lab[i][x_low + 1][0]) - int(img_lab[i][x_low][0]), 2)
        contrast_sum = contrast_sum + pow(int(img_lab[i - 1][x_high][0]) - int(img_lab[i][x_high][0]), 2) + pow(
            int(img_lab[i + 1][x_high][0]) - int(img_lab[i][x_high][0]), 2) + pow(
            int(img_lab[i][x_high - 1][0]) - int(img_lab[i][x_high][0]), 2)

    for j in range(x_low + 1, x_high):
        contrast_sum = contrast_sum + pow(int(img_lab[y_low][j - 1][0]) - int(img_lab[y_low][j][0]), 2) + pow(
            int(img_lab[y_low][j + 1][0]) - int(img_lab[y_low][j][0]), 2) + pow(
            int(img_lab[y_low + 1][j][0]) - int(img_lab[y_low][j][0]), 2)
        contrast_sum = contrast_sum + pow(int(img_lab[y_high][j - 1][0]) - int(img_lab[y_high][j][0]), 2) + pow(
            int(img_lab[y_high][j + 1][0]) - int(img_lab[y_high][j][0]), 2) + pow(
            int(img_lab[y_high - 1][j][0]) - int(img_lab[y_high][j][0]), 2)

    # 4个顶点单独考虑
    contrast_sum = contrast_sum + pow(int(img_lab[y_low + 1][x_low][0]) - int(img_lab[y_low][x_low][0]), 2) + pow(
        int(img_lab[y_low][x_low + 1][0]) - int(img_lab[y_low][x_low][0]), 2)
    contrast_sum = contrast_sum + pow(int(img_lab[y_low + 1][x_high][0]) - int(img_lab[y_low][x_high][0]), 2) + pow(
        int(img_lab[y_low][x_high - 1][0]) - int(img_lab[y_low][x_high][0]), 2)
    contrast_sum = contrast_sum + pow(int(img_lab[y_high - 1][x_low][0]) - int(img_lab[y_high][x_low][0]), 2) + pow(
        int(img_lab[y_high][x_low + 1][0]) - int(img_lab[y_high][x_low][0]), 2)
    contrast_sum = contrast_sum + pow(int(img_lab[y_high - 1][x_high][0]) - int(img_lab[y_high][x_high][0]), 2) + pow(
        int(img_lab[y_high][x_high - 1][0]) - int(img_lab[y_high][x_high][0]), 2)

    # 计算窗口有效面积
    diff_x = x_high - x_low + 1
    diff_y = y_high - y_low + 1
    num = 8 + (diff_x - 2 + diff_y - 2) * 6 + (diff_x - 2) * (diff_y - 2) * 4  # 总共多少平方项，用以计算特定像素差的概率
    return contrast_sum / num / (diff_x * diff_y)


# 计算最大的纹理对比度
@jit
def calculate_maxContrast(img_lab):
    height = 1035  # 嵌入frame的高
    width = 1830  # 嵌入frame的宽
    max_contrast = 0
    contrast = [[0] * width for i in range(height)]

    for i in range(0, height):
        for j in range(0, width):
            contrast[i][j] = Contrast(i, j, img_lab)
            if contrast[i][j] > max_contrast:
                max_contrast = contrast[i][j]

    return max_contrast


# 通过改变两帧的亮度，将图像代码(包括特定边框，前导块，隐藏的数据hidden_info等)嵌入帧中。
@jit
def encode_info(img1_lab, img2_lab, preamble_data, hidden_info, max_contrast1, max_contrast2):
    height = 1035  # 嵌入frame的高
    width = 1830  # 嵌入frame的宽
    size_x = 10  # 每个单元(cell)的像素个数为size_x*size_y
    size_y = 9
    delta_E00 = 4  # CIEDE2000公式中的参数选择
    k = 0.5

    tmp_img1 = img1_lab
    tmp_img2 = img2_lab

    # 为简化变量名，下面所有与x和y相关的变量均以cell为单位
    block_xNum = (width // size_x - 15) // 4  # 每个block的x方向cell数量
    block_yNum = (height // size_y - 15) // 4

    for i in range(0, height):
        for j in range(0, width):
            x = j // size_x
            y = i // size_y

            # 亮度自适应嵌入
            power = math.pow(img1_lab[i][j][0] - 50, 2)
            delta_l1 = (1 + 0.015 * power / (math.sqrt(20 + power))) * delta_E00
            # 根据纹理复杂度对亮度差添加权重
            contrast = Contrast(i, j, tmp_img1)
            if max_contrast1 == 0:
                max_contrast1 = 1
            alpha = contrast / max_contrast1 * (1 - k) + k
            delta_l1 = delta_l1 * alpha

            power = math.pow(img2_lab[i][j][0] - 50, 2)
            delta_l2 = (1 + 0.015 * power / (math.sqrt(20 + power))) * delta_E00
            # 根据纹理复杂度对亮度差添加权重
            contrast = Contrast(i, j, tmp_img2)
            if max_contrast2 == 0:
                max_contrast2 = 1
            alpha = contrast / max_contrast2 * (1 - k) + k
            delta_l2 = delta_l2 * alpha

            # (1)外层black border
            if x == 0 or x == width // size_x - 1 or y == 0 or y == height // size_y - 1:
                info = '1'  # info为像素点中需要隐藏的信息

            # (2)第二层black-and-white-lines
            elif x == 1 or x == width // size_x - 2 or y == 1 or y == height // size_y - 2:
                if (x + y) % 2 == 0:
                    info = '1'
                else:
                    info = '0'

            # (3)Data block之间的black-and-white-lines
            elif x == 6 + block_xNum or x == 7 + block_xNum * 2 or x == 8 + block_xNum * 3 or y == 6 + block_yNum or y == 7 + block_yNum * 2 or y == 8 + block_yNum * 3:
                if (x + y) % 2 == 0:
                    info = '1'
                else:
                    info = '0'

            # (4)Code preamble blocks
            elif x >= 2 and x <= 5 or x >= width // size_x - 6 and x <= width // size_x - 3:  # 纵向两列
                if y < 9 + 3 * block_yNum:  # 需要单独考虑左下角和右下角的情况，这里剔除
                    block_y = abs(y - 6) // (block_yNum + 1)  # y方向上与第几个data block平行;取绝对值是考虑左上角和右上角
                    y_index = (y - 2 - (block_yNum + 1) * block_y) % 4  # 在4*4block内的y值
                    if x >= 2 and x <= 5:
                        x_index = x - 2
                    else:
                        x_index = (x - 1) % 4
                elif x >= 2 and x <= 5:  # 左下角
                    y_index = (y - 1) % 4
                    x_index = x - 2
                else:  # 右下角
                    y_index = (y - 1) % 4
                    x_index = (x - 1) % 4
                info = preamble_data[x_index + y_index * 4]  # 获取preamble block内数据

            elif y >= 2 and y <= 5 or y >= height // size_y - 6 and y <= height // size_y - 3:  # 横向两行，此时不需要考虑四个角
                block_x = (x - 6) // (block_xNum + 1)
                x_index = (x - 2 - (block_xNum + 1) * block_x) % 4
                if y >= 2 and y <= 5:
                    y_index = y - 2
                else:
                    y_index = (y - 1) % 4
                info = preamble_data[x_index + y_index * 4]

            # (5)Data block部分。分别计算cell属于哪个数据块，以及在数据块内的相对位置。
            else:
                block_x = (x - 6) // (block_xNum + 1)  # x方向上block的下标
                block_y = (y - 6) // (block_yNum + 1)  # y方向上block的下标
                block_index = block_x * 4 + block_y  # block下标，介于0-15(纵向排列)
                cell_index = (x - 6 - block_x * (block_xNum + 1)) + (
                            y - 6 - block_y * (block_yNum + 1)) * block_xNum  # cell在block内的序号
                bit_index = block_index + cell_index * 16  # 获取数据流的比特位序号
                if bit_index < len(hidden_info):
                    info = hidden_info[bit_index]
                else:
                    info = '2'

            if info == '1':  # 对应的pixel隐藏数据1，第一帧亮度增加delta_l，第二帧亮度减少delta_l
                if int(img1_lab[i][j][0]) + delta_l1 > 255:  # 进行亮度修正，当亮度上溢时，设置为亮度最大值
                    img1_lab[i][j][0] = 255
                else:
                    img1_lab[i][j][0] += delta_l1
                if int(img2_lab[i][j][0]) - delta_l2 < 0:  # 进行亮度修正，当亮度下溢时，设置为亮度最小值
                    img2_lab[i][j][0] = 0
                else:
                    img2_lab[i][j][0] -= delta_l2

            elif info == '0':  # 对应的pixel隐藏数据0，第一帧亮度减少delta_l，第二帧亮度增加delta_l
                if int(img1_lab[i][j][0]) - delta_l1 < 0:
                    img1_lab[i][j][0] = 0
                else:
                    img1_lab[i][j][0] -= delta_l1
                if int(img2_lab[i][j][0]) + delta_l2 > 255:
                    img2_lab[i][j][0] = 255
                else:
                    img2_lab[i][j][0] += delta_l2

    return img1_lab, img2_lab


# 填充黑白边框，使图像大小为1920*1080
@jit
def fill(img1_rgb, img2_rgb, img1_out, img2_out):
    video_height = 1044
    video_width = 1856
    extra_x = 16  # 外围填充的黑白像素区域大小
    extra_y = 9

    # 添加黑色像素
    for i in range(0, video_height + 4 * extra_y):
        for j in range(0, extra_x):
            img1_out[i][j][0] = 0
            img1_out[i][j][1] = 0
            img1_out[i][j][2] = 0
            img2_out[i][j][0] = 0
            img2_out[i][j][1] = 0
            img2_out[i][j][2] = 0

    for i in range(0, video_height + 4 * extra_y):
        for j in range(video_width + 3 * extra_x, video_width + 4 * extra_x):
            img1_out[i][j][0] = 0
            img1_out[i][j][1] = 0
            img1_out[i][j][2] = 0
            img2_out[i][j][0] = 0
            img2_out[i][j][1] = 0
            img2_out[i][j][2] = 0

    for i in range(0, extra_y):
        for j in range(0, video_width + 4 * extra_x):
            img1_out[i][j][0] = 0
            img1_out[i][j][1] = 0
            img1_out[i][j][2] = 0
            img2_out[i][j][0] = 0
            img2_out[i][j][1] = 0
            img2_out[i][j][2] = 0

    for i in range(video_height + 3 * extra_y, video_height + 4 * extra_y):
        for j in range(0, video_width + 4 * extra_x):
            img1_out[i][j][0] = 0
            img1_out[i][j][1] = 0
            img1_out[i][j][2] = 0
            img2_out[i][j][0] = 0
            img2_out[i][j][1] = 0
            img2_out[i][j][2] = 0

            # 添加白色像素
    for i in range(extra_y, video_height + 3 * extra_y):
        for j in range(extra_x, 2 * extra_x):
            img1_out[i][j][0] = 255
            img1_out[i][j][1] = 255
            img1_out[i][j][2] = 255
            img2_out[i][j][0] = 255
            img2_out[i][j][1] = 255
            img2_out[i][j][2] = 255

    for i in range(extra_y, video_height + 3 * extra_y):
        for j in range(video_width + 2 * extra_x, video_width + 3 * extra_x):
            img1_out[i][j][0] = 255
            img1_out[i][j][1] = 255
            img1_out[i][j][2] = 255
            img2_out[i][j][0] = 255
            img2_out[i][j][1] = 255
            img2_out[i][j][2] = 255

    for i in range(extra_y, 2 * extra_y):
        for j in range(extra_x, video_width + 3 * extra_x):
            img1_out[i][j][0] = 255
            img1_out[i][j][1] = 255
            img1_out[i][j][2] = 255
            img2_out[i][j][0] = 255
            img2_out[i][j][1] = 255
            img2_out[i][j][2] = 255

    for i in range(video_height + 2 * extra_y, video_height + 3 * extra_y):
        for j in range(extra_x, video_width + 3 * extra_x):
            img1_out[i][j][0] = 255
            img1_out[i][j][1] = 255
            img1_out[i][j][2] = 255
            img2_out[i][j][0] = 255
            img2_out[i][j][1] = 255
            img2_out[i][j][2] = 255

    # 将原图复制到扩大后的对应像素位置
    for i in range(2 * extra_y, video_height + 2 * extra_y):
        for j in range(2 * extra_x, video_width + 2 * extra_x):
            img1_out[i][j][0] = img1_rgb[i - 2 * extra_y][j - 2 * extra_x][0]
            img1_out[i][j][1] = img1_rgb[i - 2 * extra_y][j - 2 * extra_x][1]
            img1_out[i][j][2] = img1_rgb[i - 2 * extra_y][j - 2 * extra_x][2]
            img2_out[i][j][0] = img2_rgb[i - 2 * extra_y][j - 2 * extra_x][0]
            img2_out[i][j][1] = img2_rgb[i - 2 * extra_y][j - 2 * extra_x][1]
            img2_out[i][j][2] = img2_rgb[i - 2 * extra_y][j - 2 * extra_x][2]

    return img1_out, img2_out


# 在图像中填充图像代码信息，preamble_data的大小为16位二进制数,hidden_info的大小为16800位二进制数,img1和img2为互补帧
def encode_video(img1, img2, preamble_data, hidden_info):
    video_height = 1044  # 原始视频图像大小
    video_width = 1856
    extra_x = 16  # 外围填充的黑白像素区域大小
    extra_y = 9

    # 将BGR格式图像转化为LAB格式，方便进行亮度转化
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    array1 = np.asarray(img1_lab)  # 将转换为矩阵以进行numba模块加速
    array2 = np.asarray(img2_lab)

    # 计算图像中对比度最大的值
    max_contrast1 = calculate_maxContrast(img1_lab)
    max_contrast2 = calculate_maxContrast(img2_lab)

    # 对LAB格式图像进行像素级处理；先计算像素点对应的cell位置，再根据数据帧的划分方式找到需要隐藏的信息，从而进行亮度改变
    img1_lab, img2_lab = encode_info(array1, array2, preamble_data, hidden_info, max_contrast1, max_contrast2)

    # 将处理好的LAB图像转换回BGR
    img1_rgb = cv2.cvtColor(img1_lab, cv2.COLOR_Lab2BGR)
    img2_rgb = cv2.cvtColor(img2_lab, cv2.COLOR_Lab2BGR)
    array1 = np.asarray(img1_rgb)
    array2 = np.asarray(img2_rgb)

    # 创建大小为1920*1080的图像，作为填充边框后图像的预备
    tmp_img1 = cv2.resize(img1_rgb, None, fx=(video_width + 4 * extra_x) / video_width,
                          fy=(video_height + 4 * extra_y) / video_height, interpolation=cv2.INTER_AREA)
    tmp_img2 = cv2.resize(img1_rgb, None, fx=(video_width + 4 * extra_x) / video_width,
                          fy=(video_height + 4 * extra_y) / video_height, interpolation=cv2.INTER_AREA)

    img1_out = np.asarray(tmp_img1)
    img2_out = np.asarray(tmp_img2)

    img1_out, img2_out = fill(array1, array2, img1_out, img2_out)

    return img1_out, img2_out
