from reedsolo import RSCodec
import subprocess


# rs编码
def rs_encode(rs_n, rs_k, info):
    # 表示能容忍(rs_n-rs_k)/2位错误
    rsc = RSCodec(rs_n - rs_k)
    midinfo = []
    for i in range(len(info)):
        midinfo.append(ord(info[i]))
    encodedinfo = rsc.encode(midinfo)

    # 将编码后的bytearray数据转为二进制
    out_code = ''.join(format(x, '08b') for x in encodedinfo)
    # print(out_code)
    return out_code


def conv_encode(n, p1, p2, info):
    ccode = "viterbi_main"
    # 调用命令行执行卷积码编码的c++程序
    out_code = subprocess.getstatusoutput(
        ccode + ' ' + '--encode' + ' ' + str(n) + ' ' + str(p1) + ' ' + str(p2) + ' ' + str(info))[1]
    return out_code


# data是输入数据，类型为字符串，内容是二进制的
# rs_n和rs_k是rs编码的两个参数
# conv_p123是卷积码编码译码的参数
# unit_l是把原数据进行编码时的分组长度

# 需要encode后的len为16800
def convert(data):
    string = ''

    for i in range(len(data)):
        if i % 8 == 0:
            bin_to_dec = int(data[i:i + 8], 2)
            # print(bin_to_dec)
            dec_to_asc = chr(bin_to_dec)
            string += dec_to_asc
            # print("iiiii,", i)
    # print(string)
    return string


def encoding(data, rs_n, rs_k, conv_p1, conv_p2, conv_p3, unit_l):
    # rs编码
    i = 0
    rs_data = ''
    rs_len1 = 0  # rs编码后,一个unit的长度
    rs_len2 = 0  # rs 编码后，最后一个unit的长度
    while i * unit_l < len(data):
        i = i + 1
        if i * unit_l < len(data):
            dec_to_asc = convert(data[(i - 1) * unit_l:i * unit_l])
            # dec_to_asc = data[(i - 1) * unit_l:i * unit_l]
            # print(len(dec_to_asc))
            rs_data = rs_data + rs_encode(rs_n, rs_k, dec_to_asc)
            if i == 1:
                rs_len1 = len(rs_encode(rs_n, rs_k, dec_to_asc))
        else:
            dec_to_asc = convert(data[(i - 1) * unit_l:])
            rs_data = rs_data + rs_encode(rs_n, rs_k, dec_to_asc)
            rs_len2 = len(rs_encode(rs_n, rs_k, dec_to_asc))

    # 把rs编码后的数据存入txt
    with open("middata.txt", "w", encoding='utf-8') as f:
        f.write(rs_data)

    # 卷积码编码
    encoded = conv_encode(conv_p1, conv_p2, conv_p3, "middata.txt")
    # 输出编码后数据长度
    #print('len_of_encoded=', len(encoded))

    return len(encoded), encoded
