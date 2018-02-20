import numpy as np
from numpy.linalg import norm
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imshow
np.set_printoptions(threshold=np.inf)
from EM import CutOut
from scipy import ndimage
import xlsxwriter

def ExtractReg(angles,radius,n):
    fit = np.polyfit(angles,radius,n)
    fitfn = np.poly1d(fit)
    ang = np.linspace(0,360,360)
    rad = fitfn(ang) 
    error = abs(radius - fitfn(angles))
    return ang,rad,fit,fitfn,error

def Loco(nnz):
    N = 20
    
    
    
    x = nnz[1]
    y = nnz[0]

    K = x.size
    
#    workbook = xlsxwriter.Workbook('leafoutliner.xlsx')
#    worksheet = workbook.add_worksheet()
#    
#    for i in range(K):
#        worksheet.write(i, 0, x[i])
#        worksheet.write(i, 1, y[i])
#        
#    workbook.close()
        
    
    deltax = np.zeros(K)
    deltay = np.zeros(K)
    
    for i in range(1,K):
        deltax[i] = x[i] - x[i-1]
        deltay[i] = y[i] - y[i-1]
#    deltax = x[1:-1] - x[0:-2]
#    deltay = y[1:-1] - y[0:-2]
    
    
    deltat = (deltax**2 + deltay**2)**0.5
    
    t = np.zeros(K)
    
    for i in range(1,K):
        t[i] = t[i-1] + deltat[i]
    
    T=t[-1]
    
    xi = np.zeros(K)
    epsilon = xi
    
    sumdeltaxj = xi
    sumdeltayj = xi
    
    
    for i in range(2,K):
        sumdeltaxj[i] = sumdeltaxj[i-1] + deltax[i-1] 
        sumdeltayj[i] = sumdeltayj[i-1] + deltay[i-1] 
#        if deltat[i] == 0:
#            print(i)
        xi[i] = sumdeltaxj[i]-deltax[i]/deltat[i]*t[i-1]
        epsilon[i] = sumdeltayj[i]-deltay[i]/deltat[i]*t[i-1]
        
    alpha = np.zeros(N)
    beta = alpha
    gamma = alpha
    delta = alpha
    
    for i in range(1,K):
        alpha[0]=alpha[0]+(deltax[i]/(2*deltat[i])*(t[i]**2-t[i-1]**2)+xi[i]*(t[i]-t[i-1]))/T
        gamma[0]=gamma[0]+(deltay[i]/(2*deltat[i])*(t[i]**2-t[i-1]**2)+epsilon[i]*(t[i]-t[i-1]))/T
        
        for j in range(1,N):
            alpha[j]=alpha[j] + deltax[i]/deltat[i]*(np.cos(2*j*np.pi*t[i]/T)-np.cos(2*j*np.pi*t[i-1]/T))
            beta[j]=beta[j] + deltax[i]/deltat[i]*(np.sin(2*j*np.pi*t[i]/T)-np.sin(2*j*np.pi*t[i-1]/T))
            gamma[j]=gamma[j] + deltay[i]/deltat[i]*(np.cos(2*j*np.pi*t[i]/T)-np.cos(2*j*np.pi*t[i-1]/T))
            delta[j]=delta[j] + deltay[i]/deltat[i]*(np.sin(2*j*np.pi*t[i]/T)-np.sin(2*j*np.pi*t[i-1]/T))
            
    for i in range(1,N):
        alpha[j] = alpha[j] * T/(2*(j*np.pi)**2)
        beta[j] = beta[j] * T/(2*(j*np.pi)**2)
        gamma[j] = gamma[j] * T/(2*(j*np.pi)**2)
        delta[j] = delta[j] * T/(2*(j*np.pi)**2)

    tau = 0.5* np.arctan2(2*(alpha[1]*beta[1]+gamma[1]*delta[1]),alpha[1]**2+gamma[1]**2-beta[1]**2-delta[1]**2)
    
    alphaprime=alpha[1]*np.cos(tau)+beta[1]*np.sin(tau)
    gammaprime=gamma[1]*np.cos(tau)+delta[1]*np.sin(tau)
    
    rho=np.arctan2(gammaprime,alphaprime)
    
    if rho<0:
        tau = tau+np.pi
        
    alphastar = np.zeros(N)
    betastar = alphastar
    gammastar = alphastar
    deltastar = alphastar
    
    for i in range(1,N):
        alphastar[i] = alpha[i]*np.cos(i*tau) + beta[i]*np.sin(i*tau)
        betastar[i] = -alpha[i]*np.sin(i*tau) + beta[i]*np.cos(i*tau)
        gammastar[i] = gamma[i]*np.cos(i*tau) + delta[i]*np.sin(i*tau)
        deltastar[i] = -gamma[i]*np.sin(i*tau) + delta[i]*np.cos(i*tau)
    
    r = alpha[1]*delta[1]-beta[1]*gamma[1]
    
    if r<0:
        betastar = -betastar
        deltastar = -deltastar
        
    a = alphastar
    b = betastar
    c = gammastar
    d = deltastar
    
    phi = 0.5*np.arctan2(2*(a*b+c*d),a**2+c**2-b**2-d**2)
    
    aprime=a*np.cos(phi)+b*np.sin(phi)
    bprime=a*np.sin(phi)+b*np.cos(phi)
    cprime=c*np.cos(phi)+d*np.sin(phi)
    dprime=c*np.sin(phi)+d*np.cos(phi)
    
    theta=np.arctan2(cprime,aprime)
    
    lambda1=np.cos(theta)*aprime+np.sin(theta)*cprime
    lambda12=np.cos(theta)*bprime+np.sin(theta)*dprime
    lambda21=-np.sin(theta)*aprime+np.cos(theta)*cprime
    lambda2=-np.sin(theta)*bprime+np.cos(theta)*dprime
    
    lambdaplus=(lambda1+lambda2)/2
    lambdaminus=(lambda1-lambda2)/2
    
    zetaplus=theta-phi
    zetaminus=-theta-phi
    
    locooffseta = a[0]
    locooffsetc = c[0]
    
    locolambdaplus = np.zeros(N-1)
    locozetaplus = np.zeros(N-1)
    locolambdaminus = np.zeros(N-1)
    locozetaminus = np.zeros(N-1)
    
    locolambdaplus[0] = lambdaplus[2]
    locozetaplus[0] = zetaplus[2]
    
    locolambdaplus[1] = lambdaplus[1]
    locozetaplus[1] = zetaplus[1] 
    
    
    locolambdaplus[2:-1] = lambdaplus[3:-1]
    locozetaplus[2:-1] = zetaplus[3:-1]
    
    locolambdaminus[2:-1] = lambdaminus[1:-3]
    locozetaminus[2:-1] = zetaminus[1:-3]
    
    locoL=(locolambdaplus*locolambdaplus+locolambdaminus*locolambdaminus+ \
           2*locolambdaplus*locolambdaminus* \
           np.cos(locozetaplus-locozetaminus-2*locozetaplus))**0.5
    
    return locoL

def contour(thresh):
    # Contour of leaf on plain background, thickness 10


    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea) # draws largest contour

    leaf_contour = cv2.drawContours(np.zeros(thresh.shape), [c], 0, (255, 255, 255), 8)
    #note, argument = -1 draws all contours

    return c, leaf_contour

def main():
    
    img=cv2.imread('weedcolour.jpg')
    x1,s1,x2,s2 = CutOut(img)
    thresh = s2.astype(np.uint8)
 #   kernel = np.ones((3,3),np.uint8)
 #   edges = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
 

    c,leafcontour = contour(thresh)
    
    edges = np.zeros(thresh.shape)
    
    for i in range(c.shape[0]):
        edges[c[i][0][1]][c[i][0][0]] = 1
    
 #   edges  = cv2.Laplacian(thresh, 2)
    nnz=np.nonzero(edges)
 #   nnz2 = np.nonzero(edges2)
    for i in range(c.shape[0]):
        nnz[0][i] = c[i][0][0]
        nnz[1][i] = c[i][0][1]
        
    
  
    z=len(nnz[0])
#    plt.imshow(thresh)
#    plt.show()    
    plt.imshow(edges)
    plt.show()
    #FIND CENTROID
    '''
    important: centroid is done on original image (before edge detection is  
    made), otherwise centroid is shifted.
    '''
    sourceIm,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,\
                                                    cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])       
    
    ##normalised vector
    ax=cx-20   #a point with same row but different col
    ay=cy
    ux=ax-cx
    uy=ay-cy
    u=[ux,uy]
    
    
    #edge vector
    vx = nnz[1] - cx
    vy = nnz[0] - cy
    
    #find angle and radius of each edge point
    angles=np.zeros(z)
    radius=np.zeros(z)
    for i in range(0,z):
        #find angle
        if nnz[0][i]>cy:#if edge point is under norm vector, add 180 deg
             angles[i]=np.arccos(np.dot(u,[vx[i],vy[i]])/(norm(u)*norm([vx[i] \
                   ,vy[i]])))*(180/np.pi)+180
             
        else:
             angles[i]=np.arccos(np.dot(u,[vx[i],vy[i]])/(norm(u)*norm([vx[i] \
                   ,vy[i]])))*(180/np.pi)
        
        #find radius
        radius[i]=norm([vx[i],vy[i]])
        
        
        #plot radius vs angle
        
    print(radius)
        
    plt.scatter(angles, radius,0.1,'b')
    
    ##################################
    n=20
    ang,rad,fit,fitfn,error = ExtractReg(angles,radius,n)
#    print(fit)
#    print(np.mean(error))
  #  plt.plot(ang, rad, 'r')
   # plt.scatter(angles,error,0.1,'k')
    ###################################
    
#    for i in range()
    
    
#    
#    locoL = Loco(nnz)
#    
 #   print(locoL)
    
    
    plt.show()
    
    return fit,np.mean(error)
    
    
if __name__ == "__main__":
    main()