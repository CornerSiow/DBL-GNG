#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:05:09 2024

@author: CORNER SIOW
"""

import numpy as np
import cv2

ERROR_INDEX = -1
        


def displayGraph(data, nodeList, edgeList):
   
    padding = 2
    img_size = 512
    minX = data[:,0].min() 
    maxX = data[:,0].max()
    
    minY = data[:,1].min() 
    maxY = data[:,1].max() 
    
    lenX = maxX - minX + padding
    lenY = maxY - minY + padding
    
    scaleX = img_size/lenX
    scaleY = img_size/lenY
    
    #Print GNG Result
    image = np.ones((img_size,img_size,3),dtype=np.uint8) * 255
    # Normal Data
    for x,y,c in data:
        image = cv2.circle(image, (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY)) , 5,  [200,200,200] , -1) 
        
    W = nodeList
    E = edgeList      
   
    for e in E:
        x, y = W[e[0]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        start_point = p
        x, y = W[e[1]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        end_point =  p
        image = cv2.line(image, start_point, end_point, [50,50,50],3)     
        

        

    for i, d in enumerate(W):
        
        p = [int((d[0]-minX+padding//2)*scaleX), int((d[1]-minY+padding//2)*scaleY)]
        image = cv2.circle(image, p , 7,  [50,50,50] , -1)  
        
     
        
        
    for e in E:
        x, y = W[e[0]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        start_point = p
        x, y = W[e[1]][:2]
        p = (int((x-minX+padding//2)*scaleX), int((y-minY+padding//2)*scaleY))
        end_point =  p
        
        
     
    cv2.imshow("window", image)  
    
        
    cv2.waitKey(2)    
    return image

class StandardGNG:
    def __init__(self, feature_number = 2, maxNodeLength = 68, L1 = 0.33, L2 = 0.05, 
                 newNodeFreq = 50, maxAge = 15, newNodeFactor = 0.05, errorNodeFactor = 0.5):
       
        self.W = []    
        self.c = []
        
        self.M = maxNodeLength
        self.alpha = L1
        self.beta = L2
        self.gamma = newNodeFreq        
        self.theta = maxAge
        self.rho = newNodeFactor
        self.delta = errorNodeFactor
        self.newNodeCount = 0
        
    
    def getConnectedNodes(self, index):
        connectedNodes = np.unique(np.concatenate((self.c[self.c[:,0] == index,1], self.c[self.c[:,1] == index,0])))
        return connectedNodes
    
    def pushData(self, x):
        if len(self.W) < 2:
            # append error
            self.W.append(np.append(x,[0]))   
            if len(self.W) == 2:
                self.W = np.asarray(self.W, dtype=np.float32)
                #append age
                self.c = [[0,1,0]]
                self.c = np.asarray(self.c)  
            return 0
                                   
        # obtains the distance to each node
        diff = x - self.W[:,:ERROR_INDEX] 
        dist = np.linalg.norm(diff, axis=1)    
        
            
        # find the first and second winer node
        s1 = np.argmin(dist)
        temp = dist[s1]
        dist[s1] = 9999999
        s2 = np.argmin(dist)
        dist[s1] = temp
        
        
        
        # check connection, if connected then set age to 0, else add new age
        c1 = np.logical_and(self.c[:,0] == s1, self.c[:,1] == s2)
        c2 = np.logical_and(self.c[:,0] == s2, self.c[:,1] == s1)
        c = np.logical_or(c1,c2)       
        if True in c:               
            self.c[c==True, 2] = 0
        else:
            self.c = np.vstack((self.c,[s1,s2,0]))
        
        #increase age
        self.c[np.logical_or(self.c[:,0] == s1, self.c[:,1] == s1),2] += 1               
        
        
        # increase winner node error
        self.W[s1,ERROR_INDEX] += self.alpha * dist[s1]
        
        
        # move the winner node        
        self.W[s1,:ERROR_INDEX] += self.alpha * (x - self.W[s1,:ERROR_INDEX])
        
        
        # find the neighbor nodes
        connectedNodes = self.getConnectedNodes(s1)
        
        
        # move neighbor node            
        self.W[connectedNodes,:ERROR_INDEX] += self.beta * (x - self.W[connectedNodes,:ERROR_INDEX])
        
       
        
        # delete edge if more then age.
        nodeToDelete = self.c[self.c[:,2] >= self.theta][:,:2]
        self.c = self.c[self.c[:,2] < self.theta]        
        nodeToDelete = np.unique(nodeToDelete)              
        
        #check node required to delete
        finalDelete = []
        for v in nodeToDelete:            
            if v not in self.c[:,:2]:               
                finalDelete.append(v)
        
        if len(finalDelete) > 1:
            finalDelete.sort(reverse=True)   
            
        for v in finalDelete:                  
            #delete local layer node
            self.c = self.c[ np.logical_not(np.logical_or(self.c[:,0] == v, self.c[:,1] == v))]        
            self.c[self.c[:,0] > v,0] -= 1
            self.c[self.c[:,1] > v,1] -= 1           
            self.W = np.delete(self.W, v, axis=0)
           
            
               
              
        # discount Error of all nodes
        self.W[:,ERROR_INDEX] *= self.delta
        
        self.newNodeCount += 1
        if self.newNodeCount >= self.gamma and len(self.W) < self.M:
            self.newNodeCount = 0
            
            # get the maximum error
            q1 = np.argmax(self.W[:,ERROR_INDEX])        
            # get the connected nodes
            connectedNodes = self.getConnectedNodes(q1)
           
            # get the maximum error of neighbors
            q2 = connectedNodes[np.argmax(self.W[connectedNodes, ERROR_INDEX])]
            
            # insert new node between q1 and q2
            self.W[q1, ERROR_INDEX] *= self.rho
            self.W[q2, ERROR_INDEX] *= self.rho            
            q3 = len(self.W)
            new_w = (self.W[q1] + self.W[q2]) * 0.5
            self.W = np.vstack((self.W,new_w))        
            
            
                       
            #remove the original edge
            self.c = self.c[ np.logical_not(np.logical_and(self.c[:,0] == q1, self.c[:,1] == q2))]
            self.c = self.c[ np.logical_not(np.logical_and(self.c[:,0] == q2, self.c[:,1] == q1))]
            #add the edge
            self.c = np.vstack((self.c,[q1,q3,0]))
            self.c = np.vstack((self.c,[q2,q3,0]))
            
          
           
data = []
f = open("../dataset/Aggregation.txt", "r")
for x in f:
    data.append(np.asanyarray(x.split(), dtype=np.float32))
data = np.vstack(data)    


gng = StandardGNG()
for epoch in range(10):
    np.random.shuffle(data)
    for x in data:
        gng.pushData(x[:2])
        displayGraph(data, gng.W, gng.c)

cv2.waitKey(0)
cv2.destroyAllWindows()
