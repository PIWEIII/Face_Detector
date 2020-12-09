import cv2
import numpy as np
my_dir = "F:\PythonProject\Face_Decetor\LBPH_imgs\\"
pic_names = ["Trump","Biden"]

class LBPH_Method:
    #用于学习的图片
    images=[]
    #标签类型，两个人分别对应0和1
    labels = [0,0,1,1]
    #类初始化，把图片加载到该类中来
    def detect(self,img_names,t_img):
        #读取每一个图片，用于标记与学习
        if len(self.images) == 0:
            for i in img_names:
                self.images.append(cv2.imread(i,cv2.IMREAD_GRAYSCALE))
        #创建LBPH识别器
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        #训练
        recognizer.train(self.images,np.array(self.labels))
        #分析待检测图片
        predict_image = t_img
        #得到label和confidence
        label,confidence = recognizer.predict(predict_image)
        print("标记(label) = ",label)
        print("可信度(confidence) = ",confidence)
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
            cv2.rectangle(t_img,(x,y),(x+w,y+w),(0,255,0),2)
            cv2.putText(t_img,pic_names[label],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),5)
        cv2.imshow("result",t_img)
        cv2.waitKey(0)
        
                

if __name__ == "__main__":
    #图片路径初始化
    images = ['a1.png','a2.png','a3.png','a4.png']
    for i in range (0,4):
        images[i] = my_dir + images[i]
    #目标图像读取
    t_img1 = cv2.imread(my_dir+"t1.png",cv2.IMREAD_GRAYSCALE)
    t_img2 = cv2.imread(my_dir+"t2.png",cv2.IMREAD_GRAYSCALE)
    #检测器初始化
    LM = LBPH_Method()
    #对两张目标图片进行检测
    LM.detect(images,t_img1)
    LM.detect(images,t_img2)




  
        
        
