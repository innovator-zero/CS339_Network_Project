# CS399_Network_Project

**SJTU 2020Fall CS399 计算机网络（D类） 课程大作业** 

**基于像素亮度变化的屏幕-相机通信系统**

对论文*Chromacode: A fully imperceptible screen-camera communication system*的复现尝试（并不完美）

**Usage：**

```top_send_video.py``` 将字符串隐藏到视频```video_input.mp4```中，并生成视频```video_demo.avi```，由于输入的视频为30帧，输出60帧的视频后视频会加速成两倍（输入60帧视频不会加速）

```top_send.py ```发送方将字符串隐藏在两张图片中，输入的字符串为二进制的```data_input.txt```的前2440位，输出两张编码后的图片```e1.jpg```和```e2.jpg```，也可以只用这两张图片生成一个视频（即重复这两帧）

```top_rx.py ```接收方读入两张相机拍摄的图片```r1.jpg```和```r2.jpg```，解码得到隐藏的字符串，并输出到```output.txt```

```judge.py``` 可以判断发送的字符串和接收到的字符串是否一致