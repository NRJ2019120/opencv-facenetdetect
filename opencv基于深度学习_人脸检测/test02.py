import cv2
import numpy as np
import PIL.Image as Image

# image = Image.open("person.jpg")
image =cv2.imread("person.jpg")
print(image.size)
# image.show()
image_data = np.array(image)
print(image_data.shape)
# img_data = image_data[]
# image.show()
cv2.imshow("img",image_data)
cv2.waitKey(0)


# img = Image.fromarray(image)
# print(img.shape)
# cv2.imshow("frame", img)

# from PIL import Image
# import numpy as np
# im = Image.open("/home/lw/a.jpg")
# im.show()
# img = np.array(im)      # image类 转 numpy
# img = img[:,:,0]        #第1通道
# im=Image.fromarray(img) # numpy 转 image类
# im.show()
