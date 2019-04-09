from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import PIL.Image as Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
print(args)
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
# time.sleep(2.0)
# video_path = r"程潇.mp4"
video_path = r"蔡依林.mp4"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')                     # 视频存储的编码的格式
cap_fps = cap.get(cv2.CAP_PROP_FPS)                          # 获取读取视屏的帧率
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # 获取读取视屏的宽度
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))         # 获取读取视屏的高度
cap_total_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 获取读取视频的总帧数
print("视频帧率，总帧数，尺寸:W,H", cap_fps, cap_total_Frames, cap_width, cap_height)
# out = cv2.VideoWriter(r"程潇_test_result.avi",fourcc, cap_fps,(cap_width,cap_height),isColor=True)
out = cv2.VideoWriter(r"蔡依林_test_result.avi",fourcc, cap_fps,(cap_width,cap_height),isColor=True)
count = 0
start = time.time()  #单位/秒
while cap.isOpened():
    ret,frame = cap.read()
    # print(frame.shape)
    if ret==True:
        count += 1
        frame = imutils.resize(frame, width=400)
        print(frame.shape)
        # print(type(frame))
        # exit()
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output frame
        out.write(frame)
        cv2.imshow("frame", frame)
        # print(type(frame))
        print(frame.shape)
        # exit()
        key = cv2.waitKey(5) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # out.write(frame)
        # # img.show()
        # cv2.imshow('frame', frame)
        # # if len(boxes) != 0:  # 保存图片
        # #     cv2.imwrite(r'/home/tensorflow01/oneday/celeba/程潇detect_img/{}.jpg'.format(count), frame)  # 保存图片
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     break
        print('第 {} 张'.format(count), end='  ')
        print("FPS of the video is {:5.2f}".format(count / (time.time() - start)))
    else:
        break
        # Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


"""蔡依林 (225, 400, 3)  FPS = 21.37"""
"""程潇  (225, 400, 3) FPS== 24.43"""