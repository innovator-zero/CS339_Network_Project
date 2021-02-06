from reedsolo import RSCodec
import subprocess
import os


# rs解码
def rs_decode(rs_n, rs_k, info):
    # 表示能容忍(rs_n-rs_k)/2位错误
    rsc = RSCodec(rs_n - rs_k)
    bytes_code = bytes(int(info[i:i + 8], 2) for i in range(0, len(info), 8))

    decodedinfo = rsc.decode(bytes_code)
    out_code = ''.join(format(x, '08b') for x in decodedinfo[0])

    return str(out_code)

# 数据经过viterbi解码之后还要处理parity位
def vite_decode(n, p1, p2, info):
    ccode = "viterbi_main"
    # 调用命令行执行卷积码编码的c++程序
    out_code = subprocess.getstatusoutput(ccode + ' ' + str(n) + ' ' + str(p1) + ' ' + str(p2) + ' ' + str(info))[1]
    return out_code

# 这里的rs_len1指的是一个unit经过rs编码之后的长度, rs_len2是最后一个unit经过rs编码之后的长度
# 因为viterbi译码不知道为啥最后会多出现几位的0， 所以要输入rs_len1和rs_len2， 截掉最后多余的0
def decoding(filename, rs_n, rs_k, conv_p1, conv_p2, conv_p3, rs_len1, rs_len2):
    # viterbi 解码
    vite_data = vite_decode(conv_p1, conv_p2, conv_p3, filename)
    # rs解码
    i = 0
    rs_data = ''
    while i * rs_len1 < len(vite_data):
        i = i + 1
        if i * rs_len1 < len(vite_data):
            rs_data = rs_data + rs_decode(rs_n, rs_k, vite_data[(i - 1) * rs_len1:i * rs_len1])
        else:
            rs_data = rs_data + rs_decode(rs_n, rs_k, vite_data[(i - 1) * rs_len1:(i - 1) * rs_len1+rs_len2])

    return rs_data

