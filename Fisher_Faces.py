import cv2
import numpy as np
import os
source_dir = "F:\PythonProject\Face_Decetor\Fisher_imgs\samples\\"
my_dir = "F:\PythonProject\Face_Decetor\Fisher_imgs\\"
pic_names = ["Mark","Bill"]

class Fisher_Faces:
    #用于学习的图片
    images=[]
    #标签类型，两个人分别对应0和1
    labels = [0,0,0,0,0,0,1,1,1,1,1,1]
    #类初始化，把图片加载到该类中来
    def detect(self,img_names,t_img):
        #初次加载探测器，读取每一个图片，用于标记与学习
        if len(self.images) == 0:
            for i in img_names:
                print(i)
                self.images.append(cv2.imread(i,cv2.IMREAD_GRAYSCALE))
        #创建EigenFace识别器
        recognizer = cv2.face.FisherFaceRecognizer_create()
        #训练
        recognizer.train(self.images,np.array(self.labels))
        #加载级联分类器,标识出图像中的人物
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        #调用detectMultiScale
        faces = faceCascade.detectMultiScale(
            t_img,  #检测图片
            scaleFactor = 1.15, #前后两次相继的扫描中，搜索窗口的缩放比例
            minNeighbors = 5,   #标识构成检测目标的相邻矩阵的最小个数，默认为3，当存在3个以上的标记，才认为人脸存在，数值越大精度越高
            minSize = (5,5)     #目标的最小尺寸 maxSize为最大尺寸 小于最小尺寸，大于最大尺寸的目标将被忽略
        )
        #对每一个人脸标注矩形，并放置文字
        for (x,y,w,h) in faces:
            #分析待检测图片
            predict_image = t_img
            #得到label和confidence
            label,confidence = recognizer.predict(predict_image) #此时t_img和predict_image都是灰度图
            print("标记(label) = ",label)
            confidence = round(confidence,2)
            print("可信度(confidence) = ",confidence)
            cv2.rectangle(t_img,(x,y),(x+w,y+w),(0,255,0),2)
            cv2.putText(t_img,pic_names[label],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
            cv2.putText(t_img,str(confidence),(x,y+w),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        cv2.imshow("result",t_img)
        cv2.waitKey(0)
    
    def sizeCvt(self,img_names):
        #标准宽高
        s_img_h = 320
        s_img_w = 320
        target_dir = my_dir
        counter = 0

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        for i in img_names:
            counter += 1
            source_img = cv2.imread(i)
            s_img = cv2.resize(source_img,(s_img_w,s_img_h),0,0,cv2.INTER_LINEAR) #修改尺寸
            if counter == 13 or counter == 14:
                cv2.imwrite(target_dir + "t" + str(counter-12)+".png",s_img)
            else:
                cv2.imwrite(target_dir + "a" + str(counter)+".png",s_img)
        print("批处理图像完成")     

if __name__ == "__main__":
    #图片路径初始化
    #先将所有的图片处理成同一种尺寸
    images = ['a1.png','a2.png','a3.png','a4.png','a5.png','a6.png','a7.png','a8.png','a9.png','a10.png','a11.png','a12.png','a13.png','a14.png']
    source_images = ['a1.png','a2.png','a3.png','a4.png','a5.png','a6.png','a7.png','a8.png','a9.png','a10.png','a11.png','a12.png','a13.png','a14.png']
    for i in range (0,14):
        source_images[i] = source_dir + source_images[i]
    #检测器初始化
    FF = Fisher_Faces()
    FF.sizeCvt(source_images)
    standard_imges = ['1','2','3','4','5','6','7','8','9','10','11','12']
    for i in range (0,12):
        standard_imges[i] = my_dir + images[i]
    #目标图像读取
    t_img1 = cv2.imread(my_dir+"t1.png",cv2.IMREAD_GRAYSCALE)
    t_img2 = cv2.imread(my_dir+"t2.png",cv2.IMREAD_GRAYSCALE)

    #对两张目标图片进行检测
    FF.detect(standard_imges,t_img1)
    FF.detect(standard_imges,t_img2)