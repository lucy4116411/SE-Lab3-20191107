import cv2
import numpy as np
import random


# 開啟鏡頭
cap = cv2.VideoCapture(0)


oriImg = cv2.imread("fly.png",cv2.IMREAD_UNCHANGED)

#通道分離 merge出透明背景
b=cv2.split(oriImg)[0]
g=cv2.split(oriImg)[1]
r=cv2.split(oriImg)[2]
img=cv2.merge([b,g,r])


#把圖片設置成適當大小
cimg = cv2.resize(img,(80,50))

# 設定影像尺寸
width = 640
height = 480

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# 初始化平均影像
ret, frame = cap.read()

avg = cv2.blur(frame, (6, 6))
avg_float = np.float32(avg)

#設定控制參數
count=0
score=0
show=0
speed=60
show_count=0



while(cap.isOpened()):
 
  # 讀取一幅影格
  ret, flip_frame = cap.read()
  
  #翻轉影像 與人動作同步
  frame=cv2.flip(flip_frame,1)
  
  cv2.putText(frame,"Move any part of your body to get rid of the fly!",(20,60),cv2.FONT_HERSHEY_DUPLEX,0.5,(36,28,237),1,cv2.LINE_AA)
  if ret == False:
    break

  # 模糊處理
  blur = cv2.blur(frame, (6, 6))

  # 計算目前影格與平均影像的差異值
  different = cv2.absdiff(avg, blur)

  # 將圖片轉為灰階
  gray = cv2.cvtColor(different, cv2.COLOR_BGR2GRAY)

  # 篩選出變動程度大於門檻值的區域
  ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

  # 使用型態轉換函數去除雜訊
  kernel = np.ones((5, 5), np.uint8)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

  # 產生移動部分的等高線
  cntImg, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  print("speed:"+str(speed))
  #以speed控制圖片出現速度
  if(count%speed==0):
      #隨機取出現位置的x,y
      imgx=random.randint(0,490)
      imgy=random.randint(0,380)
      show=1
      show_count+=1
      scoreCount=0
      #逐漸加快速度
      if(show_count%5==0):
          if(speed>10):
              speed-=4
          elif(speed>5):
              speed-=1
          else:
              speed=5
              

  #圖片貼上去 (bug)
  if(show==1): 
      frame[imgy:imgy+50,imgx:imgx+80]+=cimg    
  scoreCount=0    
  for c in cnts:
      
    # 忽略太小的區域
    if cv2.contourArea(c) < 4000:
      continue
     
    #忽略大區域
    if cv2.contourArea(c)> 9000:
      continue

    
    # 計算等高線的外框範圍
    (x, y, w, h) = cv2.boundingRect(c)
    
    #若等高線範圍在img內，即為觸碰到
    if( x<imgx<x+w and y<imgy<y+h and scoreCount==0):
        score+=1
        show=0
        scoreCount=1
        imgx=0
        imgy=0
    
    # 畫出移動量位於區域大小內的區塊(綠色)
    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
  # 畫出有位移的等高線 (黃色) (此為確認有偵測到那些移動的部分)
  #cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
  
  if(show==0):
      cv2.putText(frame,"You did it!",(240,200),cv2.FONT_HERSHEY_DUPLEX,1.5,(0,255,255),1,cv2.LINE_AA)
  count+=1
  text="Score:"
  cv2.putText(frame,text+str(score),(10,40),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,cv2.LINE_AA)
  
  # 顯示結果影像
  cv2.imshow('frame',frame)

  #若按esc鍵 跳出
  if cv2.waitKey(1)==27:
    break

  # 更新平均影像
  cv2.accumulateWeighted(blur, avg_float, 0.5)
  avg = cv2.convertScaleAbs(avg_float)

cap.release()
cv2.destroyAllWindows()