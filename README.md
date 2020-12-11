# 一个简单的人脸识别项目

年级专业：18级计算机2班

学号：1806010222

姓名：刘生伟

---

## 1、介绍

本项目采用了三种方式来进行简单的机器学习，并人脸识别。这三种方式分别是

* LBPH算法
* Eigen_Faces算法
* Fisher_Faces算法

通过对应文件夹的图片，即可进行人脸识别。下面介绍三种算法的异同



## 2、算法介绍

### (1).LBPH算法

基本原理是：将像素点A的值，与其最邻近的8个像素点值逐一比较，

如果A的像素值大于其邻近点的像素值，则得到0；

如果A的像素值小于等于其邻近点的像素值，则得到1；

最后将这8个像素点串联起来，得到一个二进制序列，转换为十进制数作为点A的LBP值。

<img src="C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211175442533.png" alt="image-20201211175442533" style="zoom:50%;" />

对图像的每个像素都进行如此的处理，就得到了LBP特征图像，成为LBPH或LBP直方图。根据此直方图，为对应的结果打上label，即可判断输入的图像与已学习的图像差距，得到置信度Confidence.

***代码如下***

```python
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
            label,confidence = recognizer.predict(predict_image)
            print("标记(label) = ",label)
            confidence = round(confidence,2)
            print("可信度(confidence) = ",confidence)
            cv2.rectangle(t_img,(x,y),(x+w,y+w),(0,255,0),2)
            cv2.putText(t_img,pic_names[label],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),5)
            cv2.putText(t_img,str(confidence),(x,y+w),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
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
```

### (2).EigenFaces人脸识别

该算法的思想在于，去除一张图片中的冗余数据，对相应的图片降维，利用主成分信息（PCA），来人脸识别。

<img src="C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211175451130.png" alt="image-20201211175451130" style="zoom:50%;" />

```python
import cv2
import numpy as np
import os
source_dir = "F:\PythonProject\Face_Decetor\EF_imgs\samples\\"
my_dir = "F:\PythonProject\Face_Decetor\EF_imgs\\"
pic_names = ["EDC","Jin Chengwu"]

class Eigen_Faces:
    #用于学习的图片
    images=[]
    #标签类型，两个人分别对应0和1
    labels = [0,0,0,1,1,1]
    #类初始化，把图片加载到该类中来
    def detect(self,img_names,t_img):
        #初次加载探测器，读取每一个图片，用于标记与学习
        if len(self.images) == 0:
            for i in img_names:
                print(i)
                self.images.append(cv2.imread(i,cv2.IMREAD_GRAYSCALE))
        #创建EigenFace识别器
        recognizer = cv2.face.EigenFaceRecognizer_create()
        #训练
        recognizer.train(self.images,np.array(self.labels))
        #加载级联分类器,标识出图像中的人物
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        #调用detectMultiScale,找出人脸
        faces = faceCascade.detectMultiScale(
            t_img,  #检测图片
            scaleFactor = 1.15, #前后两次相继的扫描中，搜索窗口的缩放比例
            minNeighbors = 5,   #标识构成检测目标的相邻矩阵的最小个数，默认为3，当存在3个以上的标记，才认为人脸存在，数值越								大精度越高
            minSize = (5,5)     #目标的最小尺寸 maxSize为最大尺寸 小于最小尺寸，大于最大尺寸的目标将被忽略
        )
        #对每一个人脸检查，并标注矩形，放置文字
        for (x,y,w,h) in faces:
             #分析待检测图片
            predict_image = t_img
            #得到label和confidence
            label,confidence = recognizer.predict(predict_image) #此时t_img和predict_image都是灰度图
            print("标记(label) = ",label)
            confidence = round(confidence,2)
            print("可信度(confidence) = ",confidence)
            cv2.rectangle(t_img,(x,y),(x+w,y+w),(0,255,0),2)
            if(confidence < 13000.0):
                cv2.putText(t_img,pic_names[label],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
                cv2.putText(t_img,str(confidence),(x,y+w),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            else:
                cv2.putText(t_img,"Unknow",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
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
            #规定最后两张图片为待检测图片
            if counter == 7 or counter == 8:
                cv2.imwrite(target_dir + "t" + str(counter-6)+".png",s_img)
            else:
                cv2.imwrite(target_dir + "a" + str(counter)+".png",s_img)
        print("批处理图像完成")     

if __name__ == "__main__":
    #图片路径初始化
    #注意，在Eigen_Faces中，所有的输入样本，尺寸必须一样
    images = ['a1.png','a2.png','a3.png','a4.png','a5.png','a6.png','a7.png','a8.png']
    source_images = ['a1.png','a2.png','a3.png','a4.png','a5.png','a6.png','a7.png','a8.png']
    for i in range (0,8):
        source_images[i] = source_dir + source_images[i]
    #检测器初始化
    EF = Eigen_Faces()
    EF.sizeCvt(source_images)
    standard_imges = ['1','2','3','4','5','6']
    for i in range (0,6):
        standard_imges[i] = my_dir + images[i]
    #目标图像读取
    t_img1 = cv2.imread(my_dir+"t1.png",cv2.IMREAD_GRAYSCALE)
    t_img2 = cv2.imread(my_dir+"t2.png",cv2.IMREAD_GRAYSCALE)

    #对两张目标图片进行检测
    EF.detect(standard_imges,t_img1)
    EF.detect(standard_imges,t_img2)
```

EigenFaces算法对于训练素材的要求多了一步：每一个训练的图片都应该尺寸相等。所以在Eigen_Faces类中，我额外定义了一个图片尺寸清洗的方法，可以将目标文件夹中的不同尺寸图片洗成统一尺寸，同时规定，输入的最后两张图片为待检测图片。

### (3).FisherFaces人脸识别

PCA方法是EigenFaces方法的核心，它找到了最大化数据总方差特征的线性组合。不可否认它很有用，但是它存在着一定的弊端，他的操作过程中，我们会损失一定的信息，如果这些信息在分类中也有关键作用（例如：根据脚的尺码可以计算出身高，但是身高仍然用作区分高矮人群体），必然会导致无法分类（或分类速度降低，因为要进行重复运算）。

FisherFaces采用LDA，线性判别分析，实现人脸识别。LDA是一种经典的线性方法。

LDA在降为的同时考虑类别信息，其思路是：在低维下，相同类型的信息应该紧密的聚集在一起，不同类别的应尽可能地分散开来。

* 类别间差距尽量“大”
* 类别内差距尽量“小”

做完LDA，首先将训练样本集投影到直线A上，让投影点满足：

* 同类间的点尽可能靠近
* 异类间的点尽可能原理

<img src="C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211181048484.png" alt="image-20201211181048484" style="zoom:50%;" />

```python
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
```

### (4).级联分类器人脸检测

为了项目更加的用户友好，我使用了级联分类起来标注出图片中的人脸，然后打上标记，标记即根据label值，取得最开始的人名数组。

```python
#加载级联分类器,标识出图像中的人物
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#调用detectMultiScale
faces = faceCascade.detectMultiScale(
    t_img,  #检测图片
    scaleFactor = 1.15, #前后两次相继的扫描中，搜索窗口的缩放比例
    minNeighbors = 5,   #标识构成检测目标的相邻矩阵的最小个数，默认为3，当存在3个以上的标记，才认为人脸存在，数值越								大精度越高
    minSize = (5,5)     #目标的最小尺寸 maxSize为最大尺寸 小于最小尺寸，大于最大尺寸的目标将被忽略
)
#对每一个人脸标注矩形，并放置文字
for (x,y,w,h) in faces:
    cv2.rectangle(t_img,(x,y),(x+w,y+w),(0,255,0),2)
    cv2.putText(t_img,pic_names[label],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cv2.imshow("result",t_img)
    cv2.waitKey(0)
   
```



## 3、项目过程中遇到的问题

### (1).训练集

找训练集是一个麻烦的过程，有很多图片在测试中根本不合格，所以需要筛选图片，这一部分需要人工完成，效率很低。但是项目比较简单，应用网络的训练集效益不大。同时要识别自己认识的人，我认为数据集应该自己规定。

### (2).规格处理

在应用EigenFaces时，它总是报错，经过阅读错误信息，发现在这个算法下，需要将所有的训练数据调整到同等大小，很简单，利用OpenCV的resize函数即可。

### (3).路径操作

将数据集复制来复制去是比较麻烦的，我们可以通过os库中的一些函数，来帮我们进行文件操作。



## 4、更好的预期

日后我会继续花时间优化这个项目，目前正在考虑的是Tensorflow建立自己的神经网络，然后通过更好的机器学习，训练属于自己的人脸识别模型。



## 5、个人总结

在本次开发中，我进一步的熟悉了OpenCV这个强大的开源库，同时也感受到了层层抽象的魅力。如果徒手从0开始写一个CV库函数，我怕我可能要花2个月左右。借助Python，我们可以凭借着简单的代码知识，轻易做到几千几万行C++才能做到的事情。这是值得我们反思的。我们在额外拓展学习的过程中，不能成为一个仅仅会利用其他人所写的代码库的人，应该参与到“造轮子”的工作当中，一方面是为开源平台做贡献，另一方面更是强化自己的学习技巧。

对于一个程序员来说，抽象思维是重要的，它可以指导你去如何将一个数学模型，变成一行行代码。同时，数学功底，决定了一个程序员的上限，只有提出模型，建立模型的人，才能真正的为国家高新科技产业做出贡献，今后我将更加努力。



## 附：代码运行效果

### LBPH:

![image-20201211215101689](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211215101689.png)

![image-20201211215115144](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211215115144.png)

### EigenFaces:

![image-20201211182259111](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211182259111.png)

![image-20201211215138134](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211215138134.png)

![image-20201211182307169](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211182307169.png)

### FisherFaces:

![image-20201211215204195](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211215204195.png)

![image-20201211182344518](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211182344518.png)

![image-20201211215217966](C:\Users\Desktop\AppData\Roaming\Typora\typora-user-images\image-20201211215217966.png)