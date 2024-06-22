#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:51:33 2024

@author: Corner
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


        
class FCMBLGNG:
  
    def __init__(self, maximumNode):
     
        self.maximumNode = maximumNode
        self.eps = 0.0001
       
        
        
    def initializeNodes(self, data, number):
        nodes = []
        selected = np.random.choice(len(data), number)
        
        nodes = data[selected]
        
                
        W = np.asarray(nodes)    
        c = np.empty((0,3),dtype=int)
        for d in data:
            diff = d[:2] - W[:,:ERROR_INDEX] 
            dist = np.linalg.norm(diff, axis=1)    
           
            idx = np.argsort(dist)        
            s1 = idx[0]
            s2 = idx[1]
            cond = np.logical_or(np.logical_and(c[:,0] == s1, c[:,1] == s2), np.logical_and(c[:,0] == s2, c[:,1] == s1))
            
            if cond.sum() == 0:
                c = np.append(c, np.asarray([[s1,s2, 0]]),axis=0)
        
        W[:, ERROR_INDEX] = 0
        self.W = W
        self.c = c
            
             
    def newNode(self):    
        if len(self.W) >= self.maximumNode:
            return
        q1 = np.argmax(self.W[:,ERROR_INDEX])       
       
        # get the connected nodes
        connectedNodes = self.getConnectedNodes(q1)
        
        if len(connectedNodes) == 0:            
            self.nodeList[q1,ERROR_INDEX] = 0
            print("no connected nodes", connectedNodes)
            return
       
        # get the maximum error of neighbors
        q2 = connectedNodes[np.argmax(self.W[connectedNodes, ERROR_INDEX])]
        
        # insert new node between q1 and q2
        q3 = len(self.W)
        new_w = (self.W[q1] + self.W[q2]) * 0.5
        self.W = np.vstack((self.W,new_w))        
       
        #remove the original edge
        self.c = self.c[ np.logical_not(np.logical_and(self.c[:,0] == q1, self.c[:,1] == q2))]
        self.c = self.c[ np.logical_not(np.logical_and(self.c[:,0] == q2, self.c[:,1] == q1))]
        #add the edge
        self.c = np.vstack((self.c,[q1,q3,0]))
        self.c = np.vstack((self.c,[q2,q3,0]))
        
        
        
    def computeMembership(self, x, selectedNodes):
        W = self.W[selectedNodes,:ERROR_INDEX]
        d = x - W
        dist = np.linalg.norm(d, axis=1)    
        
        
        k = len(selectedNodes)
        membership_mat =np.zeros(k)
        for j in range(k):
            den =  sum([dist[j] / (dist[c] + self.eps) for c in range(k)])
            membership_mat[j] = 1/(den + self.eps)
            
        
        return membership_mat
        
    
    def getConnectedNodes(self, index):
        connectedNodes = np.unique(np.concatenate((self.c[self.c[:,0] == index,1], self.c[self.c[:,1] == index,0])))
        return connectedNodes
    
    def localLearning(self, data):
       
        g = np.empty((0,3),dtype=int)
        error = np.zeros(len(self.W))
        
                
        h1 = np.zeros((len(self.W),2))
        h2 = np.zeros(len(self.W))
               
        for i, v in enumerate(data):
            #get s1 and s2
            d = v[:2] - self.W[:,:ERROR_INDEX]
            dist = np.linalg.norm(d, axis=1)    
            
            idx = np.argsort(dist)      
            s1 = idx[0]
            s2 = idx[1]
            
            connectedNodes = self.getConnectedNodes(s1)
            
            temp = np.append(connectedNodes, s1) 
           
            membership = self.computeMembership(data[i, :2], temp)
            for index, j in enumerate(temp):
                  h1[j] += membership[index] ** 2 * v[:2]
                  h2[j] += membership[index] ** 2
            
            # Connection
            cond = np.logical_or(np.logical_and(g[:,0] == s1, g[:,1] == s2), np.logical_and(g[:,0] == s2, g[:,1] == s1))
            if cond.sum() == 0:
                g = np.append(g, np.asarray([[s1,s2, 0]]),axis=0)
            # Error
            error[s1] += dist[s1]**2
           
            
        for j in range(len(self.W)):
            self.W[j,:ERROR_INDEX] = h1[j] / (h2[j] + self.eps)
        
        self.W[:,ERROR_INDEX] = error
        self.c = g
      
      
        
      
data = []
f = open("../dataset/Aggregation.txt", "r")
for x in f:
    data.append(np.asanyarray(x.split(), dtype=np.float32))
data = np.vstack(data)    


gng = FCMBLGNG(68)
gng.initializeNodes(data, 2)
for epoch in range(100):
    gng.localLearning(data)
    gng.newNode()

    displayGraph(data, gng.W, gng.c)
cv2.waitKey(0)
cv2.destroyAllWindows()

