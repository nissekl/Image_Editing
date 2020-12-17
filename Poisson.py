import cv2 as cv
import numpy as np
import scipy.sparse.linalg
import scipy.sparse as sp
import scipy.optimize as op



def possion_blending(fore_img, back_img, mask_img):
    
    h, w = fore_img.shape[0], fore_img.shape[1]
    #reshape the image into a (h*w,3) shape matrix, each row is a pixel that has 3 channel color
    fore_color_val = fore_img.reshape(h*w,3)
    back_color_val = back_img.reshape(h*w,3)
    mask_val = mask_img.reshape(h*w,1)
    
    
    points_number = h*w
    #create the sparse matirx and b channels (3 color)
    A = sp.lil_matrix((points_number, points_number))
    b_r = np.zeros((points_number,1))
    b_g = np.zeros((points_number,1))
    b_b = np.zeros((points_number,1))

    
    for i in range(points_number):
        if mask_val[i]==0:#outside the mask, just assign the value
            A[i, i] = 1
            b_r[i] = back_color_val[i,2]
            b_g[i] = back_color_val[i,1]
            b_b[i] = back_color_val[i,0]
        
        elif mask_val[i]>0:#inside the mask, calculate the laplacain value and construct the sparse matrix
            rem = i%h
            res = i//h
            l , r , u , d = i-1, i+1, rem+(res-1)*h, rem+(res+1)*h 
            
            A[i, i] = 4
            A[i, l] = -1
            A[i, r] = -1
            A[i, u] = -1
            A[i, d] = -1
            b_r[i] = 4*fore_color_val[i,2] - fore_color_val[r,2] - fore_color_val[l,2] -  fore_color_val[u,2] - fore_color_val[d,2]
            b_g[i] = 4*fore_color_val[i,1] - fore_color_val[r,1] - fore_color_val[l,1] -  fore_color_val[u,1] - fore_color_val[d,1]
            b_b[i] = 4*fore_color_val[i,0] - fore_color_val[r,0] - fore_color_val[l,0] -  fore_color_val[u,0] - fore_color_val[d,0]
    
    #flat the matrix for the LS solution (the function requred)
    b_r = b_r.ravel()
    b_g = b_g.ravel()
    b_b = b_b.ravel()
    
    xr = sp.linalg.lsmr(A, b_r)
    xg = sp.linalg.lsmr(A, b_g)
    xb = sp.linalg.lsmr(A, b_b)
    return xr, xg, xb









'''Mask Dilation'''
mask_img = cv.imread('bear_mask.png',cv.IMREAD_GRAYSCALE)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
mask_dilation = cv.dilate(mask_img, kernel, iterations=1)


'''Import image and reshape into the (h*w,3) formation, each row represent the color'''
fore_img = cv.imread('bear_foreground.jpg')
back_img = cv.imread('land_background.jpg')



#reshape to make sure all the picture are align
h, w = fore_img.shape[0], fore_img.shape[1]
back_img = cv.resize(back_img, (w, h))
mask_dilation = cv.resize(mask_dilation, (w, h))


#Calulate the result
res_r, res_g, res_b = possion_blending(fore_img, back_img, mask_dilation)
res_r = res_r[0].reshape((h,w))
res_g = res_g[0].reshape((h,w))
res_b = res_b[0].reshape((h,w))

#Create a new matrix and connect with correspinding result
res_img = np.zeros(fore_img.shape)
res_img[:,:,0] = res_b
res_img[:,:,1] = res_g
res_img[:,:,2] = res_r

cv.imwrite('flying5.png', res_img)