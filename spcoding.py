
# coding: utf-8

# In[1]:

import numpy as np
from __future__ import division
import scipy.io
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

our_images = scipy.io.loadmat('IMAGES.mat')
images = our_images['IMAGES']


# In[3]:

def fakel2norm(m):
    num = np.shape(m)[1]
    total = np.sum(m*m)
    total = total/num
    return total
        
# In[4]:

def fakel1norm(m):
    num = np.shape(m)[1]
    mabs = np.abs(m)
    total = np.sum(mabs)
    total = total/num
    return total

# In[5]:

def energy(I, a, phi, lam):
    recon_error = I - a.dot(phi)
    term1 = fakel2norm(recon_error)
    term2 = lam*fakel1norm(a)
    ans = term1 + term2
    return ans

# In[6]:

def phigradient(I, a, phi, r, s):
    total = 0
    phit = np.transpose(phi)
    for i in range(np.shape(a)[0]):
        total = total + -2*a[i][r-1]*(I[i][s-1] - np.dot(a[i], phit[s-1]))
    total = total/np.shape(a)[0]
    return total

def agradient(I, a, phi, r, s, lam):
    total = 0
    phit = np.transpose(phi)
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(phi)[1]):
            total = total + -2*(I[i][j] - np.dot(a[i], phit[j]))*phi[s-1][j]
    total = total/np.shape(a)[0]
    total = total + (lam/np.shape(phi)[0])*np.sign(a[r-1][s-1])
    return total

# In[7]:

def normalize(m):
    for i in range(np.shape(m)[0]):
        m[i] = np.true_divide(m[i], np.linalg.norm(m[i]))
    return m


# In[8]:

def random_patch(library, width):
    im = library[:,:,np.random.randint(10)]
    x = np.random.randint(np.shape(im)[0] - width)
    y = np.random.randint(np.shape(im)[1] - width)
    patch = im[x:x+width,y:y+width]
    return patch

    


# In[9]:

def make_batch(library, num, width):
    I = np.zeros((num, width*width))
    count = 0
    while count < num:
        im = random_patch(library, width)
        im_vec = im.reshape(1, width*width)
        im_vec = np.true_divide(im_vec, np.linalg.norm(im_vec))
        I[count] = im_vec
        count = count + 1
    return I


# In[10]:

def init_phis(num, size):
    phis = np.random.rand(num, size)
    return normalize(phis)


# In[4]:

I = make_batch(images, 8, 3)
phis = init_phis(20, 9)
a = np.random.rand(8, 20)
eps = 1.5


# In[8]:

energy(I, a, phis, 1)


# In[9]:

for time in range(600):
    a = np.random.rand(8, 20)
    I = make_batch(images, 8, 3)
    for ai in range(100):
        for i in range(np.shape(a)[0]):
                for j in range(np.shape(a)[1]):
                    a[i][j] = a[i][j] - 0.001*agradient(I, a, phis, i, j, 1)
    for i in range(np.shape(phis)[0]):
                for j in range(np.shape(phis)[1]):
                    phis[i][j] = phis[i][j] - 0.001*phigradient(I, a, phis, i, j)
    


# In[10]:

energy(I, a, phis, 1)


# In[ ]:



