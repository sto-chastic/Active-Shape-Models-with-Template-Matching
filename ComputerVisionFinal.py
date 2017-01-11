# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:52:04 2016

Copyright (c) 2016, David Jimenez
@author: David Jimenez and Julien Keung
 """

import csv
import cv2
import numpy as np
from numpy import linalg as LA
import fnmatch
import os
import matplotlib.pyplot as plt
import scipy.fftpack
import time
 
def butt(nrows,ncols,f, n=2, pxd=1):
    """Designs an n-th order lowpass 2D Butterworth filter with cutoff
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
   
    pxd = float(pxd)
    M = nrows
    N = ncols
    x = np.linspace(-1, 1, N)  * N / pxd
    y = np.linspace(-1, 1, M)  * M / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = 1 / (1.0 + (f/ radius)**(2*n))
    return filt
    
def gaus(nrows,ncols, sigma):

    # Create Gaussian mask of sigma = 10
    M = nrows
    N = ncols
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
    filt = np.exp(-gaussianNumerator / (2*sigma*sigma))
    
    return filt 
    
def nothing(x):
    pass
    
def preproc(img):
    [rows, cols] = img.shape
    img=np.array(img, dtype='float');
    nrows = rows 
    ncols = cols 
#    nrows = cv2.getOptimalDFTSize(nrows)
#    ncols = cv2.getOptimalDFTSize(ncols)
    
    #img = cv2.equalizeHist(img.astype(np.uint8))
        
    cv2.namedWindow('can')
    # create trackbars for color change
    cv2.createTrackbar('Max','can',0,255, nothing)
    cv2.createTrackbar('Min','can',0,255, nothing)
    cv2.createTrackbar('sigma','can',0,255, nothing)
    cv2.createTrackbar('cut','can',0,255, nothing)    
    
    Ihmf = img;
    #Ihmf = cv2.medianBlur(Ihmf.astype(np.uint8),5)
    Ihmf = cv2.GaussianBlur(img,(5,5),0)
    #Ihmf = cv2.equalizeHist(Ihmf.astype(np.uint8))
    while(1):
        cv2.imshow('can',Ihmf)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # get current positions of four trackbars
        r = cv2.getTrackbarPos('Max','can')
        g = cv2.getTrackbarPos('Min','can')
        s = cv2.getTrackbarPos('sigma','can')
        c = cv2.getTrackbarPos('cut','can')
        
        # Convert image to 0 to 1, then do log(1 + I)
        
        
        # Low pass and high pass filters
        Hlow =0.5 + butt(nrows,ncols,0.5*s)
        Hhigh = butt(nrows,ncols,c*0.01)
        #Hhigh2 = gaus(nrows,ncols,r/2) 
        # Move origin of filters so that it's at the top left corner to
        # match with the input image
        
    
        #If = np.fft.fft2(np.float32(img.copy()))
        
#         # create a mask first, center square is 1, remaining all zeros
#        rows, cols = img.shape
#
        # apply mask and inverse DFT
        HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
        HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
        #HhighShift2 = scipy.fftpack.ifftshift(Hhigh2.copy())
        # Filter the image and crop
        If = np.fft.fft2(np.float32(img.copy()))
        Iouthigh = np.abs(np.fft.ifft2(If.copy() * HhighShift))
#        
        #Iouthigh = img;
        imgLog = np.log1p(np.array(Iouthigh, dtype="float") / 255)
        If2 = np.fft.fft2(imgLog.copy());
        Ioutlow = np.abs(np.fft.ifft2(If2.copy() * HlowShift))



        # Anti-log then rescale to [0,1]
        Ihmf = np.expm1(Ioutlow)
        #Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
        Ihmf = np.array(255*Ihmf, dtype="uint8")
        Ihmf2=np.array(Iouthigh,dtype="uint8");
        Ihf2 = Ihmf;
        Ihmf = cv2.equalizeHist(Ihmf)
        #Ihmf = cv2.Sobel(Ihmf,cv2.CV_64F,1,0,ksize=3) 
        #Ihmf = cv2.Laplacian(Ihmf,cv2.CV_64F,ksize=3);
#       Ihf = np.expm1(Ioutlow)
#       Ihf = (Ihf - np.min(Ihf)) / (np.max(Ihf) - np.min(Ihf))
#       Ihf2 = np.array(255*Ihf, dtype="uint8")
        #Ihmf= cv2.Canny(Ihmf,g,r)
        
    cv2.destroyAllWindows()

   

    # Show all images
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.imshow('Homomorphic Filtered Result 1', cv2.equalizeHist(Ihmf.astype(np.uint8)))
    cv2.waitKey(0)
    cv2.imshow('Homomorphic Filtered Result', Ihf2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print r,g,s,c
    
    return r,g,s,c

def preproc_real(img,r,g,s,c):
    [rows, cols] = img.shape
    #img = cv2.medianBlur(img.astype(np.uint8),5)
    img=cv2.GaussianBlur(img,(5,5),0)
#    nrows = cv2.getOptimalDFTSize(rows)
#    ncols = cv2.getOptimalDFTSize(cols)
    nrows=rows;
    ncols=cols;
        
    # Low pass and high pass filters
    Hlow =0.5 + butt(nrows,ncols,0.5*s)
    Hhigh = butt(nrows,ncols,c*0.01)

    # apply mask and inverse DFT
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = np.fft.fft2(np.float32(img.copy()),(nrows,ncols))
    Iouthigh = np.abs(np.fft.ifft2(If.copy() * HhighShift,(rows,cols)))
        
    #Iouthigh = img;
    imgLog = np.log1p(np.array(Iouthigh, dtype="float") / 255)
    If2 = np.fft.fft2(imgLog.copy(),(nrows,ncols));
    Ioutlow = np.abs(np.fft.ifft2(If2.copy() * HlowShift,(rows,cols)))
    
    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Ioutlow)
    Ihmf = np.array(255*Ihmf, dtype="uint8")
    Ihmf2=np.array(Iouthigh,dtype="uint8");
    Ihf2 = Ihmf;
    Ihmf = cv2.equalizeHist(Ihmf)
    return Ihmf
    
def test_imagefilter(img):
    #img = cv2.imread("C:\\Users\\Julien\\Google Drive\\KULeuven\\Computer vision\\Nieuwe map\\_Data\\Radiographs\\01.tif",0);
    #img = cv2.imread("Project_Data/_Data/Radiographs/01.tif",0);
    
    rows, cols = img.shape
    print rows,cols
    img2 = img[400:1300, 1000:1900]
    #img2 = img[100:1500,60:2930]
    #r,g,s,c = preproc(img2)
    #preproc_real(img2,r,g,s,c)
    cv2.imshow('Transformed Image', preproc_real(img2,29, 42, 100, 12))
    #(0, 0, 47, 6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rescale(A):
    [c,r] = A.shape;
    mean = A[0,:];#Assign the first image as the MEAN image
    error = 10;
    
    old = np.zeros((1,numberOfPoints))
    j=0
    scale1 = np.sqrt(np.sum(np.square(mean)))
    while error>0.00001:
        scale = np.sqrt(np.sum(np.square(mean)))
        A = np.divide(A,scale)
        mean = np.divide(mean,scale)
        old[:] = mean[:];
        
        for i in range(c):#For for calculating the rotation of the matrices.
            #plt.plot(A[i,::2],A[i,1::2])
            #plt.plot(mean[::2],mean[1::2])
            
            s, a, T, A[i,:]= transform(A[i,:], mean);
            #plt.plot(A[i,::2],A[i,1::2])
            A[i,:] = project_tangent(A[i,:], mean); #Rotating the matrix X coordinate
            #plt.plot(A[i,::2],A[i,1::2])
            #plt.show()
            
        mean[:] = np.mean(A,0)#Create a new mean matrix based on the mean of the rotated and scaled shapes.
        scale = np.sqrt(np.sum(np.square(mean)));
        mean[:] = mean /scale;
        error = np.linalg.norm(mean-old);
        #print j
        #print error
        
    mean = mean;
    
    return mean,A,error,scale1


def project_tangent(A, T):
    '''
    Project onto tangent space
    @param A:               
    @param T:                            
    @return: s = scaling, alpha = angle, T = transformation matrix
    '''
    tangent = np.dot(A, T);
    A_new = A/tangent;
    return A_new


def transform(A, T):
    '''
    Calculate scaling and theta angle of image A to target T
    @param A:               
    @param T:                            
    @return: s = scaling, alpha = angle, Tr = transformation matrix
    '''
    Ax, Ay = split(A)
    Tx, Ty = split(T)
    #b2 = (np.dot(Ax, Ty)-np.dot(Ay, Tx))/np.power(np.dot(A, A),2)
    #a2 = np.dot(T, A)/np.power(np.dot(A, A),2)
    
    b2 = (np.dot(Ax, Ty)-np.dot(Ay, Tx))/np.dot(A.T, A)
    a2 = np.dot(T, A)/np.dot(A.T, A)
        
    alpha = np.arctan(b2/a2) #Optimal angle of rotation is found.
    Tr = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    s =np.sqrt(np.power(a2,2) + np.power(b2,2))
    
    result = np.dot(s*Tr, np.vstack((Ax,Ay)));
    #plt.plot(result[0,:],result[1,:])
    new_A = merge(result);
    Tr = Tr
    
    return s, alpha, Tr, new_A
    
def split (A):
    x = A[::2];#Divide in X coordinates
    y = A[1::2];#Divide in Y coordinates
    return x,y

def merge(XY):
    A = np.zeros((1,XY.shape[1]*2))
    A[0,::2] = XY[0, :] ;
    A[0,1::2] = XY[1, :];
    return A

def PCA(X,Variation):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param Variance:         Proportion of the total variation desired.                        
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape 
    Xm = np.mean(X, axis=0)
    x = np.zeros((n,d))
    x = X - Xm
    x = x.T
    Xc = np.dot(x.T,x)
    [L,V] = LA.eig(Xc)
    [ne] = L.shape
    index = np.argsort(-np.absolute(L))
    Li = L[index]
    varTot = np.sum(np.absolute(Li))
    varSum = 0;
    for numEig in range(0,ne):
        varSum = varSum + np.absolute(Li[numEig])
        if varSum/varTot >= Variation:
            print 'Number of Eigenvectors after PCA ='
            print numEig
            break
    Vi = np.dot(x,V)
    
    Vii = Vi[:,index]
    Viii = Vii[:,:numEig]
    Liii = Li[:numEig]
    
    VI = np.dot(Viii.T,Viii)

    VV= np.divide(Viii,np.sqrt(np.diagonal(VI)))
    print("Eigs on top")
    return [Liii,VV,Xm]
    
def pyramid(initialPossition,scale,b,eigVals,eigVecs,mean,testImage,training):

        R = np.array([[1,0],[0,1]])        
        
        testImage = np.uint8(testImage.reshape(height,width))
        lowerRes2 = np.uint8(cv2.pyrDown(testImage))        
        lowerRes3 = np.uint8(cv2.pyrDown(lowerRes2))
        
        reader2 = np.divide(reader,2)
        reader3 = np.divide(reader2,2)

        x = initialPossition[0]/4
        y = initialPossition[1]/4      
        
#        test = lowerRes3.copy()
#        mean3 = mean*scale/4
#        for i in range(numberOfPoints/2):
#
#            test[mean3[i*2+1]+y,mean3[i*2]+x]=255;
#        
#        cv2.imshow('test', test.astype(np.uint8));
#        cv2.waitKey(0);
#        cv2.destroyAllWindows();
        
        he,wi = testImage.shape
        
        readerTemp = reader
        
        train2,he2,wi2 = down(training,he,wi)
        train3,he3,wi3 = down(train2,he2,wi2)
        
        global reader
        
        reader = reader3
        scale = scale/4
        T3,R3,s3,b3, XinM1 = Matching_Real(train3,np.divide(initialPossition,4), R, scale, b,eigVals,eigVecs,mean,lowerRes3,he3,wi3)
        reader = reader2
        T2,R2,s2,b2, XinM2 = Matching_Real(train2,T3*2, R3, s3*2, b3,eigVals,eigVecs,mean,lowerRes2,he2,wi2)
        reader = readerTemp
        T1,R1,s1,b1, XinM3 = Matching_Real(training,T2*2, R2, s2*2, b2, eigVals,eigVecs,mean,testImage,he,wi)
        
        return XinM3, XinM2, XinM1 

def down(train,he,wi) :
    img = train[0].reshape(he,wi)
    hep,wip = cv2.pyrDown(img).shape
    output = np.zeros((trainN,hep*wip))
    for i,row in enumerate(train):
        img = np.uint8(row.reshape(he,wi))
        cv2.pyrDown(img).reshape(1,hep*wip)  
        output[i] = cv2.pyrDown(img).reshape(1,hep*wip)  
        
    
    return output,hep,wip
    
def Matching_Real(train,initialPossition, R, scale, b, eigVals,eigVecs,mean,testImage,he,wi):
    '''We do the PCA and the model "creation" using the form [X1,Y1,X2,Y2...], not sepparated
    initialPossition = Initial Possition for the model as [Xt,Yt]
    
    '''
    Tr = R
    s = scale
    print("Matching-----------")
    error = 1000;
    repetitions = 0;
    while repetitions<5:#error > 0.0001:
        repetitions = repetitions +1;
        print(repetitions)
        
        
        X = np.add(mean, np.dot(eigVecs,b).T)
        Xs = np.dot(s*Tr, np.vstack(split(X[0])))
        
        Xin = (Xs.T + initialPossition).T
        XinM = merge(Xin)
        
        test = testImage.copy()
        test = test.reshape(he,wi)
        for i in range(numberOfPoints/2):

            test[XinM[0,i*2+1],XinM[0,i*2]]=255;
        
#        cv2.imshow('test', test.astype(np.uint8));
#        cv2.waitKey()
#        cv2.destroyAllWindows();

        
        # Get new Target points
        Xrec,fracConv = mahalanobisMatching(train,XinM[0],testImage,he,wi)

        XT = np.vstack((split(Xrec)))
        XI = np.vstack((split(X[0])))
        meanT = np.mean(XT,axis=1)
        meanI = np.mean(XI,axis=1)
        
        initialPossition = np.subtract(meanT, meanI )
#        print meanI
#        print meanT
        print initialPossition
        
        s, a, Tr, xFin = transform(X[0], merge((XT.T-meanT).T)[0])
        
        Tinv = np.linalg.inv(Tr)
        
        y = np.dot(Tinv,np.vstack((np.subtract(np.vstack((split(Xrec)))[0],initialPossition[0]),np.subtract(np.vstack(split(Xrec))[1],initialPossition[1]))))/s

        
        y = y/ np.dot(merge(y)[0], mean) 
        
#        y = project_tangent(merge(y)[0], mean)
#        y = np.vstack(split(y))
#        error = np.sqrt(np.sum(np.power(np.subtract(np.vstack(split(X[0])),y),2), axis =0))
        #print error
        # print(np.power(np.subtract(X-merge(y)));
#        fig = plt.figure()
#        plt.hist(error, 20)
#        plt.show()
#            
        plt.plot(merge(y)[0,::2], merge(y)[0,1::2])
        plt.plot(mean[::2], mean[1::2])
        plt.show()
        bn = np.dot(eigVecs.T,np.subtract(merge(y),mean).T)
        print bn
        print eigVals
        a = 100000
        for i in range(bn.shape[0]):
            if bn[i] >= 3*np.sqrt(eigVals[i]*a)/a:
                b[i] = 3*np.sqrt(eigVals[i]*a)/a
            elif bn[i] <= -3*np.sqrt(eigVals[i]*a)/a:
                b[i] = -3*np.sqrt(eigVals[i]*a)/a
            else:
                b[i] = bn[i]
        test = testImage.copy();
        temp = np.add(mean, np.dot(eigVecs,b).T)
        x = initialPossition[0]
        yy = initialPossition[1]   
#        for i in range(numberOfPoints/2):
#            test[temp[0,i*2+1]+yy,temp[0,i*2]+x]=255;
#        resized = cv2.resize(test.astype(np.uint8), (1000,600), interpolation = cv2.INTER_AREA)
#        cv2.imshow('test', resized);
#        cv2.waitKey(0);
#        cv2.destroyAllWindows();
#        Xf = np.add(mean, np.dot(eigVecs,b).T)
#        Xfs = np.dot(s*Tr, np.vstack(split(Xf[0])))
#        
#        Xfin = (Xfs.T + initialPossition).T
#        XfinM = merge(Xfin)
#        error = np.linalg.norm(XinM - XfinM)#not calculated on the right place, but might work, should be on the image space
#        print error
        Xf = np.add(mean, np.dot(eigVecs,b).T)
        Xfs = np.dot(s*Tr, np.vstack(split(Xf[0])))
        
        Xfin = (Xfs.T + initialPossition).T
        XfinM = merge(Xfin)
        
        if fracConv >= 89:
            break;


    
    cv2.destroyAllWindows()
    return  initialPossition, Tr, s, b, XfinM

def extract_Features(A,n,imgR,he,wi):
    """A should only be one vector, e.g. reader[0]
        n is the profile size, number of pixels in the lines
        imgR is the corresponding image to extract features in vector form.
    """
    img = imgR.reshape(he,wi);

    vec = np.zeros((numberOfPoints,(2*n+1)));
    vecExtr = np.zeros(((vec.shape[1]),numberOfPoints/2));

    m = n+1;
    vec2 =  np.zeros((numberOfPoints,(2*m+1)));
    vec = np.zeros((numberOfPoints,(2*n+1)));
    vecExtr2 = np.zeros(((vec2.shape[1]),numberOfPoints/2));
    vecExtr = np.zeros(((vec.shape[1]),numberOfPoints/2));

    for j in range(numberOfPoints/2):
        x1=  A[np.mod(j*2-2,numberOfPoints)];
        y1 = A[np.mod(j*2-1, numberOfPoints)];
        x2 = A[np.mod(j*2,numberOfPoints)];
        y2 = A[np.mod(j*2+1,numberOfPoints)];
        x3 = A[np.mod(j*2+2,numberOfPoints)];
        y3 = A[np.mod(j*2+3,numberOfPoints)];
        
        dx = x3-x1;
        dy = y3-y1;
        
        mag = np.sqrt(dx*dx+dy*dy);
         
        dx = dx/mag;
        dy = dy/mag;
        
        x = x2;
        y = y2;
        
        #print x,y
        
        nx = -dy;
        ny = dx;
        length2 = np.linspace(-m,m,2*m+1);
        length = np.linspace(-n,n,2*n+1);
        
        vec2[2*j,:] = x + length2 * nx;
        vec2[2*j+1,:] = y + length2 * ny;
        vec[2*j,:] = x + length * nx;
        vec[2*j+1,:] = y + length * ny;
        for i in range(2*m+1):
            vecExtr2[i,j] = img[vec2[2*j+1,i],vec2[2*j,i]];
            #img[vec2[2*j+1,i],vec2[2*j,i]] = 255
        for i in range(2*n+1):
            vecExtr[i,j] = vecExtr2[i+2,j]-vecExtr2[i,j]
            #img[vec[2*j+1,i],vec[2*j,i]] = 255
    #cv2.imshow('test', img.astype(np.uint8));
    #cv2.waitKey(0);
    #cv2.destroyAllWindows()
    return vec,vecExtr
            
def distribution_Training(train,profileSize,he,wi):
    
    vec,_ = extract_Features(combine4Landmarks(0,landm,reader)[0],profileSize,train[0],he,wi);
    A = np.zeros((train.shape[0],vec.shape[1]*vec.shape[0]/2))
    for i in range(train.shape[0]):
        #vecExtr = np.zeros(((vec.shape[1]),numberOfPoints));
        img = train[i,:].reshape(he,wi);
#        lapl = img
        lapl = cv2.equalizeHist(np.uint8(img))
#        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
#        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
#        sobelx_8u = np.power(sobelx,2)
#        sobely_8u = np.power(sobely,2)
#        lapl=np.uint8(np.sqrt(sobelx_8u+sobely_8u));
        
#        sobelx_8u = np.abs(sobelx)
#        sobely_8u = np.abs(sobely)
#        lapl=np.uint8(sobelx_8u+sobely_8u);

        
#        lapl = cv2.Laplacian(img,cv2.CV_64F,ksize=3);

        vec,vecExtrP = extract_Features(combine4Landmarks(0,landm,reader)[i],profileSize,lapl,he,wi);
#        cv2.imshow('test', lapl.astype(np.uint8));
#        cv2.waitKey(0);
#        cv2.destroyAllWindows()
#        print(np.sum(vec))
#        for j in range(numberOfPoints):
#            for k in range(vec.shape[1]):
#                vecExtr[k,j] = lapl[vec[2*j,k],vec[2*j+1,k]];
        for l in range(vecExtrP.shape[1]):
            A[i,vecExtrP.shape[0]*l:vecExtrP.shape[0]*(l+1)] = vecExtrP[:,l]/np.sum(np.abs(vecExtrP[:,l]));
            
#    print(vecExtrP.shape[0])
#    print(A.shape)
    return A   

    
def mahalanobisMatching(train,testModel,testImage,he,wi):
    '''
    Calculates the best point locations based on mahalanobis distance.
    @param testModel: Test shape
    @param testImage: Image to test the shape on
    '''
    
    #testImage = training[0]
    
    profileSize = 20
    comparedProfileSize = 10
    
    sampledProfile = distribution_Training(train,comparedProfileSize,he,wi)
    img = testImage
#    lapl =img

    lapl = cv2.equalizeHist(np.uint8(img))
#    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
#    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
#    sobelx_8u = np.power(sobelx,2)
#    sobely_8u = np.power(sobely,2)
#    lapl=np.uint8(np.sqrt(sobelx_8u+sobely_8u));
        
#    sobelx_8u = np.abs(sobelx)
#    sobely_8u = np.abs(sobely)
#    lapl=np.uint8(sobelx_8u+sobely_8u);


#    lapl = cv2.Laplacian(img,cv2.CV_64F,ksize=3);
    
    vec,vecExtrP = extract_Features(testModel,profileSize,lapl,he,wi)
    subProfile = np.zeros((sampledProfile.shape[0],(2*comparedProfileSize+1)))
    #print(subProfile.shape)
    newPointCoords = np.zeros((numberOfPoints))
    convergencePoints = 0
    for k in range(numberOfPoints/2):
        
        subProfile =  sampledProfile[:,k*(2*comparedProfileSize+1):(k+1)*(2*comparedProfileSize+1)]

        index = 0;
        prevMaha = 1000000;
        for i in range(2*profileSize+1 - 2*comparedProfileSize+1-1):
            norm = np.sum(np.abs(vecExtrP[i:i+(2*comparedProfileSize+1),k]))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ind = np.arange(2*comparedProfileSize+1)    
            ind2 = np.arange(41)               # the x locations for the groups
            wiidth = 0.35                      # the width of the bars## the bars
            rects1 = ax.bar(ind2, vecExtrP[:,k]/norm, wiidth,
                        color='blue')
            Xm = np.mean(subProfile,axis=0)
            rects2 = ax.bar(ind+wiidth, Xm, wiidth,
                        color='red')
            
            m = mahalanobis(subProfile,vecExtrP[i:i+(2*comparedProfileSize+1),k]/norm)
            if (m<prevMaha):

                prevMaha = m;
                norm1 = norm

                index = i
        
        if np.absolute(profileSize-comparedProfileSize-index) < convValue:
            
            
            convergencePoints = convergencePoints +1;
           
        
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ind = np.arange(2*comparedProfileSize+1)                # the x locations for the groups
#        wiidth = 0.35                      # the width of the bars## the bars
#        rects1 = ax.bar(ind, vecExtrP[index:index+(2*comparedProfileSize+1),k]/norm1, wiidth,
#                    color='black')
#        rects2 = ax.bar(ind+wiidth, Xm, wiidth,
#                    color='red')
#        plt.show()
        newPointCoords[2*k] = vec[2*k,comparedProfileSize+index]
        newPointCoords[2*k+1] = vec[2*k+1,comparedProfileSize+index]
   
    fracConv = convergencePoints*100/(numberOfPoints/2)
    print("Fraction of Converging points")
    print(fracConv)
    return newPointCoords,fracConv
    
def mahalanobis(X, g):
    """"Calculate the Mahalanobis distance based on 
    @param X   matrix of training data and shape = (nb samples, nb dimensions of each sample)
    @param g   the current samplepoints around the model  
    
    [rows, cols] = X.shape with,
    Rows: Instance
    Cols: pixel intensity
    
    @param return Mahalanamobis distance
    """
    
    [rows, cols] = X.shape 
    Xm = np.mean(X, axis=0)
    
    #[n,d] = X.shape 
    #Xm = np.mean(X, axis=0)
    #x = np.zeros((n,d))
    #x = X - Xm
    #x = x.T
    #Xc = np.dot(x.T,x)
    
    x = np.zeros((rows, cols))
    x = X-Xm;
    #xcov = np.cov(Xp)
    
    x = x.T
    xcov = np.dot(x,x.T)
    
    i_xcov = np.linalg.pinv(xcov)
    g = g-Xm
    dist = (g).dot(i_xcov).dot(g.T);
    
    return dist    
    
def test_profile_grad( A):
    
#    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) 
#    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    for j in range(numberOfPoints):
        x1 = A[0,np.mod(j*2, numberOfPoints)]
        y1= A[0,np.mod(j*2+1,numberOfPoints)];
        x2 = A[0,np.mod(j*2+2,numberOfPoints)];
        y2 = A[0,np.mod(j*2+3,numberOfPoints)];
        
        #print x1, y1, x2,y2;
        
#        x_l = np.linspace(-10, 1, 10)*sobelx[x,y]/np.sqrt(np.power(sobelx[x,y],2)+np.power(sobely[x,y],2))
#        y_l = np.linspace(-10, 1, 10)*sobely[x,y]/np.sqrt(np.power(sobelx[x,y],2)+np.power(sobely[x,y],2))
#        print x
#        print y
#        print np.rint(gx)
#        print np.rint(gy)
    
    cv2.imshow('test', img);
    cv2.waitKey(0);
    cv2.destroyAllWindows()


def combine4Landmarks(ini,fin,vect):
    comb = vect[ini::8,:]
    for i in range(ini+1,fin):
        comb = np.hstack((comb,vect[i::8,:]))
    
    return comb

def template(training,reader,t,num_tooth,(wh,ww)):
    
    rows, cols = training.shape
    cropLocal = np.zeros((4,num_tooth))
    cropGlobal = np.zeros((4,t))
    for i, row in enumerate(training[0:t]):
        for j in range(num_tooth):
            rx = np.max(reader[j+i*8,::2])
            ry = np.max(reader[j+i*8,1::2])
            lx = np.min(reader[j+i*8,::2])
            ly = np.min(reader[j+i*8,1::2])
            cropLocal[:,j]=np.array([rx,ry,lx,ly])
            
        rx = np.max(cropLocal[0])
        ry = np.max(cropLocal[1])
        lx = np.min(cropLocal[2])
        ly = np.min(cropLocal[3])
        center = np.mean(split(combine4Landmarks(0,num_tooth,reader)[i]),axis=1);
#        cx=(rx+lx)/2
#        cy=(ry+ly)/2
#        print center
#        print cx,cy
        cx = center[0]
        cy = center[1]

        h = ry-ly;
        w = rx-lx;
        cropGlobal[:,i] = np.array([cx,cy,h,w]);
        
    h = np.max(cropGlobal[2])+wh
    w = np.max(cropGlobal[3])+ww     

    vSize = h*w;
    out = np.zeros((t,vSize),dtype=np.uint8)
    
    for i,row in enumerate(training[0:t]):
        img = row
        img = np.uint8(img.reshape(height, width))
        img = cv2.GaussianBlur(img,(3,3),0)
        img = cv2.equalizeHist(img);
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
        sobelx_8u = np.uint8(np.absolute(sobelx))
        sobely_8u = np.uint8(np.absolute(sobely))
        
        img2 = (img+0.3*sobelx_8u+0.7*sobely_8u );
        cx = cropGlobal[0,i];
        cy = cropGlobal[1,i]; 
#       lx = cropGlobal[2,i];
#       ly = cropGlobal[3,i];
#        cv2.rectangle(img2,(int(lx),int(ly)), (int(rx),int(ry)), 255, 5)
        img2 = img2 [cy-h/2:cy+h/2, cx-w/2:cx+w/2]
#        img2 = cv2.resize(img2,(int(h), int(w)), interpolation = cv2.INTER_LINEAR)
        img2 = np.uint8(img2 )
#        cv2.imshow('output', sobely)
#        cv2.waitKey(0)
        #ret3,th = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        cv2.imshow('output', sobely_8u)
#        cv2.waitKey(0)
#        cv2.imshow('output', img2.astype(np.uint8))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows();
        
        out[i] = img2.reshape(1,vSize)
        
    output = np.mean(out,axis=0)
    output = np.uint8(output)
    output = output.reshape(h,w)
#    cv2.imshow('output', output)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows();
    return out,output,h,w;
    
def template_matching(template,img,methods):
    
#     img = cv2.GaussianBlur(img,(3,3),0)
     img2 = cv2.equalizeHist(img)
     w, h = template.shape[::-1]
     # All the 6 methods for comparison in a list
#     methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#                 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
     #methods = ['cv2.TM_CCOEFF_NORMED']
     for meth in methods:
         img = img2.copy()
         method = eval(meth)
         # Apply template Matching
         res = cv2.matchTemplate(img,template,method)
         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
         # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
         if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
             top_left = min_loc
         else:
             top_left = max_loc
         bottom_right = (top_left[0] + w, top_left[1] + h)
    
         
         img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
         cv2.rectangle(img,top_left, bottom_right, 255, 5)
#         fig = plt.figure(figsize=(8, 6)) 
#         plt.subplot(131),plt.imshow(res,cmap = 'gray')
#         plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#         plt.subplot(132),plt.imshow(img,cmap = 'gray')
#         plt.title('Detected Points'), plt.xticks([]), plt.yticks([])
#         plt.subplot(133),plt.imshow(img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]],cmap = 'gray')
#         plt.title('Found patch'), plt.xticks([]), plt.yticks([])
#         plt.suptitle(meth)
#         plt.show()
         fig = plt.figure(figsize=(8, 6)) 
         plt.imshow(img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]],cmap = 'gray')
         plt.title('Found patch'), plt.xticks([]), plt.yticks([])
         plt.show()
     return top_left, bottom_right;

def model(Tdata, Rdata,RdataP, Num):
    ind = Num -1;
    indices  = range(ind*8,(ind+1)*8,)
    train = np.delete(Tdata, ind, axis = 0)
    treader = np.delete(Rdata,indices, axis = 0)
    treaderP = np.delete(RdataP,indices, axis = 0)
    gr = split(combine4Landmarks(0,landm,Rdata[indices])[0])
    initialPossition = np.mean(gr,axis=1)
    
    global reader 
    reader = treader.copy()
    
    out,temp,h,w = template(train,treader,trainN,8,(150,800))
    out,temp2,h,w = template(train,treader,trainN,8,(150,20))
    out,temp3,h,w = template(train,treader,trainN,4,(20,20))

    
    img2 = np.uint8(Tdata[ind].reshape(height,width))
    tl1,br1 = template_matching(temp,img2,['cv2.TM_CCOEFF_NORMED'])
    tl2,br2 = template_matching(temp2,img2[tl1[1]:br1[1],tl1[0]:br1[0]], [ 'cv2.TM_CCOEFF_NORMED'])
    tl3=np.add(tl1,tl2)
    br3=np.add(tl1,br2)
    tl,br = template_matching(temp3,img2[tl3[1]:br3[1],tl3[0]:br3[0]], [ 'cv2.TM_CCOEFF_NORMED'])
    
    
    print "image", ind
    #initialPossition = np.zeros((1,2))
    #initialPossition= (np.add(tl3,tl) + np.add(tl3,br))/2
    shape, A,error,scale = rescale(treaderP);
    
    print initialPossition
    [eigVals,eigVecs,mean] = PCA(A*10,0.98)
    
    eig1 = np.vstack(split(eigVecs[:,0]))
    eig2 = np.vstack(split(eigVecs[:,1]))
    eig3 = np.vstack(split(eigVecs[:,2]))
    eig = np.vstack(split(shape))
    fig = plt.figure(figsize=(20, 8)) 
    a=0.1 *np.sqrt(eigVals[0])
    b=0.1 *np.sqrt(eigVals[1])
    c=0.1 *np.sqrt(eigVals[2])
    plt.subplot(141), plt.plot(eig[0],eig[1])
    plt.title('Mean Shape')
    plt.subplot(142), plt.plot(eig1[0],eig1[1],'ro')
    plt.title('First Principle Component')
    plt.subplot(143), plt.plot(eig2[0],eig2[1],'bo')
    plt.title('Second Principle Component')
    plt.subplot(144), plt.plot(eig3[0],eig3[1],'ko')
    plt.title('Third Principle Component')
    plt.show()
    fig = plt.figure(figsize=(20, 8)) 
    c=-3 *np.sqrt(eigVals[2]*100)/100
    a=0
    b=0
    plt.subplot(131), plt.plot(eig[0]+a*eig1[0]+b*eig2[0]+c*eig3[0],eig[1]+a*eig1[1]+b*eig2[1]+c*eig3[1],'ro')
    b = 0
    a = 0
    c = 0
    plt.subplot(132), plt.plot(eig[0]+a*eig1[0]+b*eig2[0]+c*eig3[0],eig[1]+a*eig1[1]+b*eig2[1]+c*eig3[1],'bo') 
    c=3 *np.sqrt(eigVals[2]*100)/100
    a=0
    b=0
    plt.subplot(133), plt.plot(eig[0]+a*eig1[0]+b*eig2[0]+c*eig3[0],eig[1]+a*eig1[1]+b*eig2[1]+c*eig3[1],'ko') 
    plt.show()
    
    Tr = np.identity(2)
    Xs = np.dot(scale*Tr, np.vstack(split(mean)))
    Xin = (Xs.T + initialPossition).T
    
    b = np.zeros((eigVecs.shape[1],1))

    ground = np.vstack(split(combine4Landmarks(0,landm,Rdata[indices])[0]))

    XinM3, XinM2, XinM1 = pyramid(initialPossition,scale,b,eigVals,eigVecs,shape,Tdata[ind],train)
    
    Tr = np.identity(2)
    Xf1 = np.dot(4*Tr, np.vstack(split(XinM1[0])))
    Xf2 = np.dot(2*Tr, np.vstack(split(XinM2[0])))
    Xf3 = np.dot(Tr, np.vstack(split(XinM3[0])))


    final = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    for i in range(160):
#        cv2.circle(final,(np.int(Xin[0,i]),np.int(Xin[1,i])), 2, (0,0,255), -1)
#        cv2.circle(final,(np.int(Xf1[0,i]),np.int(Xf1[1,i])), 2, (0,255,0), -1)
#        cv2.circle(final,(np.int(Xf2[0,i]),np.int(Xf2[1,i])), 2, (255,0,0), -1)
        cv2.circle(final,(np.int(Xf3[0,i]),np.int(Xf3[1,i])), 2, (0,0,0), -1)
        cv2.circle(final,(np.int(ground[0,i]),np.int(ground[1,i])), 2, (255,255,255), -1)
    fig = plt.figure(figsize=(8, 6)) 
    plt.imshow(final[400:1100,1100:1800],cmap ='gray')
    plt.show()
    
    t = ground - Xf3;
    h = np.sqrt(np.power(t[0],2) + np.power(t[1],2));
    
    error = 0;    
    
    return error,h; 


if __name__ == '__main__':
    
    global height 
    height = 1400;
    global width 
    width = 2870;
    global trainN
    trainN = 13;
    global reader
    reader = np.zeros([112,80]);
    global numberOfPoints
    numberOfPoints = 320
    global landm
    landm = 4
    global convValue
    convValue = 8
    
    readerP = np.zeros([112,80])
    i=0;
    directory = "C:\Users\David\Google Drive\KULeuven\Computer vision\Nieuwe map\\_Data/"
    #directory = "Project_Data/_Data/"
    for filename in fnmatch.filter(os.listdir(directory + "Landmarks/original/"),'*.txt'):
        reader[i,:] = np.loadtxt(open(directory+ "Landmarks/original/"+filename,"rb"),delimiter=",",skiprows=0)
        reader[i,::2]  = reader[i,::2]-60 #-np.mean(reader[i,::2]);#Zero-mean of the X axis
        reader[i,1::2] = reader[i,1::2]-100#-np.mean(reader[i,1::2]);#Zero-mean of the Y axis
        
        #readerP[i,:] = reader[i,:];
        #readerP[i,::2]  = reader[i,::2]-np.mean(reader[i,::2]);#Zero-mean of the X axis
        #readerP[i,1::2] = reader[i,1::2]-np.mean(reader[i,1::2]);#Zero-mean of the Y axis
        
        i+=1;
        
    
    vSize = height*width;
    training = np.zeros((14,vSize))#, dtype=np.int)
    i = 0;
    for filename in fnmatch.filter(os.listdir(directory + "Radiographs/"),'*.tif'):
        img = cv2.imread(directory + "Radiographs/" + filename,0)
        
        img2 = img.copy()
        #img2 = preproc_real(img2,0,0,46,110);
        result = img2[100:1500,60:2930]
        
        result = np.asarray(result)

        imgT = np.zeros((1,vSize), dtype=np.int)
        imgT = result.reshape(1,vSize)
        training[i] = imgT
#        break;
        i+=1;
        
    readerP = combine4Landmarks(0,landm,reader).copy()
    for i in range(14):
        readerP[i,::2]  = readerP[i,::2]-np.mean(readerP[i,::2]);#Zero-mean of the X axis
        readerP[i,1::2] = readerP[i,1::2]-np.mean(readerP[i,1::2]);#Zero-mean of the Y axis
#
#        
    oldreader = reader.copy()
    oldreaderP = readerP.copy()
#    
#    for i in range(14):
#        num = i+1
#        error = model(training, oldreader, oldreaderP, num)
#    errors = np.zeros((14,160))
#    for i in range(14):
    
ind = 1
error, h = model(training, oldreader, oldreaderP, ind)
#errors[i] = h
fig = plt.figure()
plt.hist(h, 20, range=[0, 200])
plt.xlabel('Norm between groundtruth and result', fontsize=16)
plt.ylabel('Number of pixels', fontsize=16)
plt.show()

#    sca = 0.6
#    alpha = np.pi/4
#    Ax, Ay = split(readerP[7]);
#    Tr = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
#    result = np.dot(sca*Tr, np.vstack((Ax,Ay)));
#    readerP[7] = merge(result);
#        
#    shape, A,errors,scale= rescale(readerP);
#    plt.plot(reader[0,::2],reader[0,1::2])
#    
#    plt.plot(shape[::2],shape[1::2])
#    plt.plot(readerP[0,::2], readerP[0,1::2])
#    plt.plot(readerP[1,::2], readerP[1,1::2])
#    plt.plot(readerP[2,::2], readerP[2,1::2])
#
#    plt.show()
#   
#    plt.plot(A[0,::2], A[0,1::2])
#    plt.plot(A[1,::2], A[1,1::2])
#    plt.plot(A[2,::2], A[2,1::2])
#    plt.show()
#    
#    
#    
#    for i in range(6):
#        j = i + 6
#        plt.plot(shape[::2]*scale,shape[1::2]*scale)
#        plt.plot(readerP[j,::2], readerP[j,1::2])
#        plt.plot(A[j,::2]*scale, A[j,1::2]*scale)
#        plt.show()


#    


    

