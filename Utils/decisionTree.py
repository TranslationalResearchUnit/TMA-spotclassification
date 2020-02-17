
def decisionTree(classCount, classCountCapsNet, p):
        
        maxProb = classCount.index(max(classCount))        

        score = [x /sum(classCount)*100 for x in classCount]

        scoreCapsNet  = [x /sum(classCountCapsNet)*100 for x in classCountCapsNet]

        
        print(score)
        surePar = True
        checkagain = False
        if (maxProb==0):
                if (score[0]>=75):
                        maxProb = 0
                        if (classCount[2]>=2) and (score[0]<90):
                                maxProb = 2
                                if score[2]<20:
                                        checkagain = True                                        
                else:
                        if (p<1):
                                if score[1]>score[2]:
                                        maxProb = 1
                                        checkagain = True
                                else:
                                        maxProb = 2
                                        checkagain = True
                                
                        if (p==1):
                                if (score[2]/(score[1]+score[2])*100)>20 :
                                        maxProb = 2
                                        if score[2]<20:
                                                checkagain = True
                                else:
                                        maxProb = 1                                        
                        if (p==2):
                                if (score[2]/(score[1]+score[2])*100)>5 :
                                        maxProb = 2                                        
                                else:
                                        maxProb = 1
                
                if checkagain == True:
                        if (maxProb==2):
                                if (scoreCapsNet[2]>=20):
                                        maxProb = 2
                                
                                if (scoreCapsNet[2]>=10) and (scoreCapsNet[2]<20):
                                        maxProb = 2
                                        surePar = False
                                if (scoreCapsNet[2]<10):                                        
                                        if (score[0]+ scoreCapsNet[0]> score[1]+ scoreCapsNet[1]):
                                                maxProb = 0
                                                surePar = False
                                        else:
                                                maxProb = 1
                                                surePar = False
                        
        else:                
                if (maxProb==1):                        
                        if (p==1):
                                if ((score[2]>20) and (score[1]<40)) or (score[2]>30):
                                        maxProb = 2
                                        surePar = False
                                else:
                                        maxProb = 1
                        if (p==2):            
                                if (score[2]<5) :
                                        maxProb = 1
                                else:
                                        maxProb = 2
                                        if score[1]+scoreCapsNet[1]>75 :
                                                surePar = False
                else:
                        if (p==1):
                                if (score[1]<50):
                                        maxProb = 2
                                        if(score[1]<30)and(score[1]+scoreCapsNet[1]>score[2]+ scoreCapsNet[2]) and(score[2]<30):
                                                maxProb = 1
                                                surePar = False
                                        if (score[1]+scoreCapsNet[1]>50):
                                                surePar = False
                                else:
                                        maxProb = 1
                                if maxProb == 2:
                                        if score[1]+scoreCapsNet[1]>100 :
                                                maxProb = 1
                                                surePar = False
                        if(p==2):
                                if score[1]+scoreCapsNet[1]>75 :
                                        surePar = False
                        
        return maxProb,score,surePar

            
            
