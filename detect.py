import cv2
import numpy as np
import math


# 配置数据
class Config:
    def __init__(self):
        pass

    min_area = 1000000
    min_contours = 30
    epsilon_start = 10
    epsilon_step = 10


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# 求两点间的距离
def point_distance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


# 找出外接四边形, c是轮廓的坐标数组
def boundingBox(idx, c):
    if len(c) < Config.min_contours:
        return None
    epsilon = Config.epsilon_start
    while True:
        approxBox = cv2.approxPolyDP(c, epsilon, True)

        # 求出拟合得到的多边形的面积
        theArea = math.fabs(cv2.contourArea(approxBox))
        # 输出拟合信息
        # print("contour idx: %d ,contour_len: %d ,epsilon: %d ,approx_len: %d ,approx_area: %s" % (
        # idx, len(c), epsilon, len(approxBox), theArea))
        if (len(approxBox) < 4):
            return None
        if theArea > Config.min_area:
            if (len(approxBox) > 4):
                # epsilon 增长一个步长值
                epsilon += Config.epsilon_step
                continue
            else:  # approx的长度为4，表明已经拟合成矩形了
                # 转换成4*2的数组
                approxBox = approxBox.reshape((4, 2))
                return approxBox
        else:
            # print("failed to find boundingBox,idx = %d area=%f" % (idx, theArea))
            return None


# -----------------------------------------------------------------------------------------------

def detect_one(image, thres, file_name='', debug=False):
    # 转成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 中值滤波平滑，消除噪声
    binary = cv2.medianBlur(gray, 3)

    if debug:
        # 显示转换后的二值图像
        cv2.imshow("gray", binary)
        cv2.waitKey()

    # 转换为二值图像
    ret, binary = cv2.threshold(binary, thres, 255, cv2.THRESH_BINARY)
    if debug:
        # 显示转换后的二值图像
        cv2.imshow("binary", binary)
        cv2.waitKey()

    # 进行2次腐蚀操作（erosion）
    # 腐蚀操作将会腐蚀图像中白色像素，可以将断开的线段连接起来
    # binary = cv2.dilate(binary, None, iterations=2)
    # if debug:
    #     # 显示腐蚀后的图像
    #     cv2.imshow("dilate", binary)
    #     cv2.waitKey()

    # canny 边缘检测
    binary = cv2.Canny(binary, 0, 255, apertureSize=3)
    if debug:
        # 显示边缘检测的结果
        cv2.imshow("Canny", binary)
        cv2.waitKey()

    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        # 输出轮廓数目
        print("the count of contours is  %d" % (len(contours)))

    find = False
    detected = []
    # 针对每个轮廓，拟合外接四边形,如果成功，则将该区域切割出来，作透视变换，并保存为图片文件
    for idx, c in enumerate(contours):
        if debug:
            img = image.copy()
            cv2.drawContours(img, c, -1, (0, 0, 255), 3)
            cv2.imshow('cx', img)
            cv2.waitKey()

        approxBox = boundingBox(idx, c)
        if approxBox is None:
            # print("box none")
            continue

        # 获取最小矩形包络
        rect = cv2.minAreaRect(approxBox)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = box.reshape(4, 2)
        box = order_points(box)

        # 待切割区域的原始位置，
        # approxPolygon 点重排序, [top-left, top-right, bottom-right, bottom-left]
        src_rect = order_points(approxBox)

        w, h = point_distance(box[0], box[1]), point_distance(box[1], box[2])

        # 生成透视变换矩阵
        dst_rect = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]],
            dtype="float32")

        # 透视变换
        M = cv2.getPerspectiveTransform(src_rect, dst_rect)

        # 得到透视变换后的图像
        warped = cv2.warpPerspective(image, M, (w, h))

        if not debug:
            return warped
        else:
            # 将变换后的结果图像写入png文件
            cv2.imwrite(file_name + "piece%d.png" % idx, warped, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            find = True
            detected.append(idx)
    zero = np.zeros(1)
    return zero



def detect(img):
    for i in range(50, 250, 10):
        d = detect_one(img, i)
        if d.shape[0] > 100:
            #print(i)
            return d
    print('Fail to detect')

