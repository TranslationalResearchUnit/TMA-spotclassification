from openslide import open_slide
import cv2
from matplotlib import pyplot as plt
import skimage.measure
from skimage.transform import rescale, rotate
import numpy as np

from keras.models import load_model
from Utils.capsnetModel.model import Model
from Utils.predict import predictImage

import os,sys,time



def getSpotContour(conf,outFolder):
        
        img = open_slide(conf.dataName)

        img.getLevelDimensions(1)
                   
        level = 7
        levelDimensions = img.level_dimensions[level]
        patch = img.read_region((0, 0), level, (levelDimensions[0],levelDimensions[1]))

        plt.imsave(outFolder+"\level"+str(level)+".png", patch)

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)        

        cv2.imwrite(outFolder + "\\gray.png",gray_blur)

        
        ret3,thresh = cv2.threshold(gray_blur,16,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        thresh = ~thresh

        cont_img = thresh.copy()

        cv2.imwrite(outFolder + "\\contour.png",cont_img)
        
        _,contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return img,patch,contours

def displayOutput(patchesInfoResult,outFolder):
        img = cv2.imread(outFolder + "\\level7.png")
        for cnt in patchesInfoResult:
                classType, surePar, x,y,w,h = cnt[0]
                if classType>=0:
                        if classType == 0:
                                color = (0,32,255)
                                
                        if classType == 1:
                                color = (0,255,0)

                        if classType == 2:
                                color = (0,0,0)                                        
                        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                        
                        if surePar == False:
                                color = (255,0,0)
                                cv2.rectangle(img,(x+2,y+2),(x+w-2,y+h-2),color,2)
        cv2.imwrite(outFolder + "\\Final.png",img)
        
def delete3header(patch,contours,conf,outFolder):
        ymin = 9999999
        countBoxMin = 0

        ymax = 0
        countBoxMax = 0
        
        for cnt in contours:
                
            area = cv2.contourArea(cnt)
            
            if area < conf.minAreaTMASpot or area > conf.maxAreaTMASpot:
                continue
            
            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            #cv2.ellipse(img, ellipse, (0,255,0), 2)
            
            x,y,w,h = cv2.boundingRect(cnt)
            if (y+h>ymax):
                    ymax = y +h            
            if  (y<ymin) :
                    ymin = y
                    
            cv2.rectangle(patch,(x,y),(x+w,y+h),(0,255,0),2)

        for cnt in contours:
                area = cv2.contourArea(cnt)
            
                if area < conf.minAreaTMASpot or area > conf.maxAreaTMASpot:
                        continue
                
                x,y,w,h = cv2.boundingRect(cnt)
                if y < ymin + 20:
                        countBoxMin = countBoxMin +1
                        cv2.rectangle(patch,(x,y),(x+w,y+h),(255,255,0),2)
                        
                if y + h > ymax - 20:
                        countBoxMax = countBoxMax +1
                        cv2.rectangle(patch,(x,y),(x+w,y+h),(0,255,255),2)
 
        plt.imsave(outFolder + "\\roi.png", patch)

        if (countBoxMax ==3):                
                return ymax,0
                
        else:
                if countBoxMin ==3:
                        return ymin,1
                else:
                        if countBoxMax<3 and countBoxMin<3 :
                                if countBoxMax> countBoxMin:
                                        return ymax,0
                                else:
                                        return ymin,1
                        if countBoxMax<3 and countBoxMin>3 :
                                return ymax,0
                        
                        if countBoxMax>3 and countBoxMin<3 :        
                                return ymin,1
                        if countBoxMax>3 and countBoxMin>3 :
                                return -1,-1

def is_purple_dot(r, g, b):

	rb_avg = (r+b)/2

	if r > g - 10 and b > g - 10 and rb_avg > g + 20:

		return True

	return False

	

#this is actually a better method than is whitespace, but only if your images are purple lols

def is_purple(crop):

	pooled = skimage.measure.block_reduce(crop, (int(crop.shape[0]/15), int(crop.shape[1]/15), 1), np.average)

	num_purple_squares = 0

	for x in range(pooled.shape[0]):

		for y in range(pooled.shape[1]):

			r = pooled[x, y, 0]*255

			g = pooled[x, y, 1]*255

			b = pooled[x, y, 2]*255

			if is_purple_dot(r, g, b):

				num_purple_squares += 1

	if num_purple_squares > 100: 

		return True

	return False


#zero padding for really small crops

def zero_pad(image, patch_size):



	x = image.shape[0] #get current x and y of image

	y = image.shape[1]

	if x >= patch_size and y >= patch_size:

		return image #if its already big enough, then do nothing



	x_new = max(x, patch_size)

	y_new = max(y, patch_size)

	new_image = np.zeros((x_new, y_new, 3)) #otherwise, make a new image

	x_start = int(x_new/2 - x/2)

	y_start = int(y_new/2 - y/2) #find where to place the old image

	new_image[x_start:x_start+x, y_start:y_start+y, :] = image #place the old image



	return new_image #return the padded image

def patchProcess(modelCNN, modelCapsNet, image_path, patch_size, inverse_overlap_factor):


        outputed_windows_per_subfolder = 0

        
        type_histopath = True


        image = plt.imread(image_path+".png")

        image = zero_pad(image, patch_size)

        x_max = image.shape[0] #width of image
        y_max = image.shape[1] #height of image


        
        x_steps = int((x_max-patch_size) / patch_size * inverse_overlap_factor) #number of x starting points
        y_steps = int((y_max-patch_size) / patch_size * inverse_overlap_factor) #number of y starting points

        step_size = int(patch_size / inverse_overlap_factor) #step size, same for x and y


	#loop through the entire big image
        classCount1 = [0,0,0]
        classCount2 = [0,0,0]
        for i in range(x_steps+1):
                for j in range(y_steps+1):
                        #get a patch

                        x_start = i*step_size
                        x_end = x_start + patch_size

                        y_start = j*step_size
                        y_end = y_start + patch_size

                        assert x_start >= 0; assert y_start >= 0; assert x_end <= x_max; assert y_end <= y_max

                        patch = image[x_start:x_end, y_start:y_end, :]
                        
                        #assert patch.shape == (patch_size, patch_size, 3)

                        out_path =  image_path +"_"+str(x_start)+"_"+str(y_start)+".png"

                        if type_histopath: #do you want to check for white space?
                                if is_purple(patch): #if its purple (histopathology images)
                                        plt.imsave(out_path, patch)
                                        outputed_windows_per_subfolder += 1
                                        patchType = predictImage(modelCNN,out_path, "CNN")
                                        classCount1[patchType] = classCount1[patchType]+1

                                        patchType = predictImage(modelCapsNet,out_path, "CapsNet")
                                        classCount2[patchType] = classCount2[patchType]+1
                                        
                        else:
                                plt.imsave(out_path, patch)
                                outputed_windows_per_subfolder += 1
                                predictImage(modelCNN,out_path)
                                classCount1[patchType] = classCount1[patchType]+1

                                predictImage(modelCapsNet,out_path)
                                classCount2[patchType] = classCount2[patchType]+1
                                
                                        
        return(classCount1,classCount2)



def produce_patches(conf,img,contours,yheader,top,Folder):
        j = 0
        idx = 1
        patchesInfoResult = []
        
        

        #"HistoPath"
        modelCapsNet = Model(conf.modelName, output_folder=None,  nb_label= conf.nb_label)
        modelCapsNet.load(conf.modelCapsNetLocation)


        modelCNN = load_model(conf.modelCNNLocation)

        
        for cnt in contours:     
            area = cv2.contourArea(cnt)
            patchInfo = []
            x,y,w,h = cv2.boundingRect(cnt)
            
            if area < conf.minAreaTMASpot or area > conf.maxAreaTMASpot:
                    patchInfo.append([-1,x,y,w,h])
                    continue
            if len(cnt) < 5:
                    patchInfo.append([-1,x,y,w,h])
                    continue            
            if top!=-1:
                    if top==0:
                            if y + h > yheader - 20:
                                continue
                    else:
                            if y < yheader + 20:
                                continue
            else:
                    print("Can not find the header")
            
            factor= img.getLevelDownsample(7)
            patch = img.getUCharPatch(int(x*factor), int(y*factor),int(w*factor/img.getLevelDownsample(conf.level)),int(h*factor/img.getLevelDownsample(conf.level)),conf.level)            
            outFolder = Folder+"\\"+str(idx-1)+"_"+os.path.splitext(os.path.basename(conf.dataName))[0]+"_"+str(x)+"_"+str(y)
            try:
                    os.stat(outFolder)
            except:
                    os.mkdir(outFolder)
                
            spotImage = outFolder+"\\"+os.path.splitext(os.path.basename(os.path.basename(conf.dataName)))[0]+"_"+str(x)+"_"+str(y)
            plt.imsave(spotImage+".png", patch)

            start_time = time.time()
            
            print("----Processing for spot ", idx, " at: ",x, "and",y )
            filePath = os.path.splitext(os.path.basename(os.path.basename(conf.dataName)))[0]+"_"+str(x)+"_"+str(y)

            classCount, classCountCapsNet= patchProcess(modelCNN,modelCapsNet,spotImage,224,3)
            print(classCount, classCountCapsNet)
            maxProb,surePar = decisionTree(classCount,classCountCapsNet,conf.priority)                           
            
            score = [x /sum(classCount)*100 for x in classCount]  
            
            spot = cv2.imread(spotImage+".png")
            
            label1 = "{}: {:.2f}%".format("Outlier", score[0] )
            label2 = "{}: {:.2f}%".format("Not Tumor", score[1] )
            label3 = "{}: {:.2f}%".format("Tumor", score[2])
            
            cv2.putText(spot, "CNN predict", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,32,255), 2)
            
            cv2.putText(spot, label1, (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.putText(spot, label2, (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.putText(spot, label3, (10, 85),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)


            score2 = [x /sum(classCountCapsNet)*100 for x in classCountCapsNet]

            cv2.putText(spot, "CapsNet predict", (10, 105),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,32,255), 2)
            
            label1 = "{}: {:.2f}%".format("Outlier", score2[0] )
            label2 = "{}: {:.2f}%".format("Not Tumor", score2[1] )
            label3 = "{}: {:.2f}%".format("Tumor", score2[2])
            
            cv2.putText(spot, label1, (10, 125),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.putText(spot, label2, (10, 145),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.putText(spot, label3, (10, 165),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)


            
            if maxProb== 1:
                    color = (0,255,0)
                    print("Class: NotTumor")
            
            else:
                    if maxProb== 2:
                            color = (0,0,0)
                            print("Class: Tumor")
                    else:
                            color = (0,32,255)
                            print("Class: Outlier")
                                        
            cv2.rectangle(spot,(0,0),(np.size(spot, 1),np.size(spot, 0)),color,10)
            if surePar == False:
                                color = (255,0,0)
                                cv2.rectangle(spot,(5,5),(np.size(spot, 1)-5,np.size(spot, 0)-5),color,5)
                                
            cv2.imwrite(spotImage+".png",spot)
            

            try:
                    os.stat(Folder+"\\0\\")
                    os.stat(Folder+"\\1\\")
                    os.stat(Folder+"\\2\\")
                    
            except:
                    os.mkdir(Folder+"\\0\\")
                    os.mkdir(Folder+"\\1\\")
                    os.mkdir(Folder+"\\2\\")
                    
            if (conf.priority> -1):               
                    if maxProb==0:
                            cv2.imwrite(Folder+"\\0\\" +str(j)+".png",spot)
                    if maxProb==1:
                            cv2.imwrite(Folder+"\\1\\" +str(j)+".png",spot)
                    if maxProb==2:
                            cv2.imwrite(Folder+"\\2\\" +str(j)+".png",spot)
                    j = j + 1
                    
            elapsed_time = time.time() - start_time
            print("time:", elapsed_time)
            print("----")
            patchInfo.append([maxProb,surePar,x,y,w,h])
            idx = idx + 1
            patchesInfoResult.append(patchInfo)
            
        return patchesInfoResult

            
def decisionTree(classCount,classCountCapsNet,p):
        
        
        decisionCNN = classCount.index(max(classCount))
        scoreCNN = [x /sum(classCount)*100 for x in classCount]

        
                        
        scoreCapsNet  = [x /sum(classCountCapsNet)*100 for x in classCountCapsNet]        
        decisionCapsNet = classCountCapsNet.index(max(classCountCapsNet))
        

        decision = 2
        surePar = True
        
        if (decisionCNN==decisionCapsNet):                
                decision = decisionCNN
                if (p != decisionCNN):                                               
                        if (scoreCNN[2]>10) or (scoreCapsNet[2]>10):
                                if (decision!=2):
                                        surePar = False
                                if (p==2):
                                        decision = 2                                                                
        else:
                surePar = False
                if (decisionCNN==p) or (decisionCapsNet==p):
                        decision = p
                        
                else:                        
                        if (p==2):
                                decision = 1
                        else:
                                decision = 2                                                        
        return decision, surePar
            
