# opencv-facenetdetect

opencv基于 resnet_ssd深度学习人脸检测模型
使用方法：

图片中人脸的检测：
python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel


视频中人脸的检测：
python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel


参考博客：https://blog.csdn.net/u014365862/article/details/79655657
