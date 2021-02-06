# 代码功能：从拍摄的连续两帧画面中提取有效数据信息
# 输入为拍摄的连续两帧画面，大小为1888*1062
# 输出为16788比特的数据流

import cv2
from numba import jit
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# 对给定的连续两帧图像进行解码，根据亮度信息差获取隐藏在帧中的数据
@jit
def soft_decide(img1_lab, img2_lab):
    height = 1035  # 嵌入frame的高
    width = 1830  # 嵌入frame的宽
    size_x = 10  # 每个单元(cell)的像素个数为size_x*size_y
    size_y = 9
    extra_x = 16  # 外围填充的黑白像素区域大小
    extra_y = 9
    x_cells = width // size_x
    y_cells = height // size_y

    block_xNum = (width // size_x - 15) // 4  # 每个block的x方向cell数量
    block_yNum = (height // size_y - 15) // 4  # 每个block的y方向cell数量

    black = []
    white = []
    sum_blackborder = 0  # 用最外层的黑框判断是否两帧相反

    # 亮度变化，以cell为单位
    delta_l = np.zeros((y_cells, x_cells))

    for i in range(extra_y, height + extra_y):
        for j in range(extra_x, width + extra_x):
            x = (j - extra_x) // size_x
            y = (i - extra_y) // size_y

            if img1_lab[i][j][0] > img2_lab[i][j][0]:
                delta_l[y][x] = delta_l[y][x] + int(img1_lab[i][j][0]) - int(img2_lab[i][j][0])
            else:
                delta_l[y][x] = delta_l[y][x] + (int(img2_lab[i][j][0]) - int(img1_lab[i][j][0])) * (
                    -1)  # jit模块，不写成(*-1)会有问题

    for y in range(y_cells):
        for x in range(x_cells):
            delta_l[y][x] = float(delta_l[y][x] / (size_x * size_y))  # 一个cell中的亮度变化取均值

    # 对亮度变化图的每个cell进行处理，根据两帧的像素亮度变化确定隐藏的比特位，从而进行01字符串解码
    for y in range(y_cells):
        for x in range(x_cells):

            # (1)外层black border
            if x == 0 or x == width // size_x - 1 or y == 0 or y == height // size_y - 1:
                sum_blackborder = sum_blackborder + delta_l[y][x]
            # (2)第二层black-and-white-lines
            elif x == 1 or x == width // size_x - 2 or y == 1 or y == height // size_y - 2:
                if (x + y) % 2 == 0:
                    black.append(delta_l[y][x])
                else:
                    white.append(delta_l[y][x])
            # (3)Data block之间的black-and-white-lines
            elif x == 6 + block_xNum or x == 7 + block_xNum * 2 or x == 8 + block_xNum * 3 or y == 6 + block_yNum or y == 7 + block_yNum * 2 or y == 8 + block_yNum * 3:
                if (x + y) % 2 == 0:
                    black.append(delta_l[y][x])
                else:
                    white.append(delta_l[y][x])
            # (4)Code preamble blocks
            elif x >= 2 and x <= 5 or x >= width // size_x - 6 and x <= width // size_x - 3:  # 纵向两列
                continue
            elif y >= 2 and y <= 5 or y >= height // size_y - 6 and y <= height // size_y - 3:  # 横向两行
                continue
            else:
                continue

    if sum_blackborder < 0:
        for i in range(0, y_cells):
            for j in range(0, x_cells):
                delta_l[i][j] = delta_l[i][j] * (-1)

        for i in range(0, len(black)):
            black[i] = black[i] * (-1)
        for i in range(0, len(white)):
            white[i] = white[i] * (-1)

    return black, white, delta_l


def decode_video(img1, img2, tmp_file, length):
    # 数据帧初始定义
    height = 1035  # 嵌入frame的高
    width = 1830  # 嵌入frame的宽
    size_x = 10  # 每个单元(cell)的像素个数为size_x*size_y
    size_y = 9
    x_cells = width // size_x
    y_cells = height // size_y

    block_height = 25  # 每个data block的高，单位为cell
    block_width = 42  # 每个data block的宽，单位为cell

    sum_data = np.zeros(block_height * block_width * 16)  # 存储一个cell内隐藏数据之和，最终做均值处理

    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)

    # 为简化变量名，下面所有与x和y相关的变量均以cell为单位
    block_xNum = (width // size_x - 15) // 4  # 每个block的x方向cell数量
    block_yNum = (height // size_y - 15) // 4

    # 获取亮度变化
    black, white, delta_l = soft_decide(img1_lab, img2_lab)

    # out_win = "dl"
    # cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow(out_win, delta_l)
    # cv2.waitKey()

    # 归一化
    black.sort()
    R1 = black[int(len(black) // 2)]
    white.sort()
    R0 = white[int(len(white) // 2)]
    print('R1=' + str(R1))
    print('R0=' + str(R0))

    preamble_count = [0 for i in range(0, 16)]
    tmp_data = [0 for i in range(0, 16)]
    preamble_data = ''

    for y in range(y_cells):
        for x in range(x_cells):
            if delta_l[y][x] > 0:  # 判断隐藏信息为1或0，用于前导块提取
                info = 1
            else:
                info = 0

            # (1)外层black border
            if x == 0 or x == width // size_x - 1 or y == 0 or y == height // size_y - 1:
                continue
            # (2)第二层black-and-white-lines
            elif x == 1 or x == width // size_x - 2 or y == 1 or y == height // size_y - 2:
                continue
            # (3)Data block之间的black-and-white-lines
            elif x == 6 + block_xNum or x == 7 + block_xNum * 2 or x == 8 + block_xNum * 3 or y == 6 + block_yNum or y == 7 + block_yNum * 2 or y == 8 + block_yNum * 3:
                continue
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
                tmp_data[x_index + y_index * 4] += info
                preamble_count[x_index + y_index * 4] += 1
            elif y >= 2 and y <= 5 or y >= height // size_y - 6 and y <= height // size_y - 3:  # 横向两行
                block_x = (x - 6) // (block_xNum + 1)
                x_index = (x - 2 - (block_xNum + 1) * block_x) % 4
                if y >= 2 and y <= 5:
                    y_index = y - 2
                else:
                    y_index = (y - 1) % 4
                tmp_data[x_index + y_index * 4] += info
                preamble_count[x_index + y_index * 4] += 1
            # (5)Data block部分。分别计算cell属于哪个数据块，以及在数据块内的相对位置。
            else:
                block_x = (x - 6) // (block_xNum + 1)  # x方向上block的下标
                block_y = (y - 6) // (block_yNum + 1)  # y方向上block的下标
                block_index = block_x * 4 + block_y  # block下标，介于0-15(纵向排列)
                cell_index = (x - 6 - block_x * (block_xNum + 1)) + (
                        y - 6 - block_y * (block_yNum + 1)) * block_xNum  # cell在block内的序号
                bit_index = block_index + cell_index * 16  # 获取数据流的比特位序号

                sum_data[bit_index] = delta_l[y][x]

    # 为方便解码，将数据序列写入一个txt文件中，每行一个数据
    f = open(tmp_file, 'w')
    for i in range(0, length):
        v = (sum_data[i] - R0) / (R1 - R0)
        if v > 1:
            f.write('1\n')
        elif v < 0:
            f.write('0\n')
        else:
            f.write(str(v) + '\n')
    f.close()

    for i in range(0, 16):
        if tmp_data[i] > preamble_count[i] * 0.5:
            preamble_data = preamble_data + '1'
        else:
            preamble_data = preamble_data + '0'

    return preamble_data
