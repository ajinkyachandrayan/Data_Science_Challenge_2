
# coding: utf-8

# In[1]:

from PIL import Image
import os, sys


# In[2]:

path = "images_35_35/"
dirs = os.listdir( path )


# In[3]:

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((35,35), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)


# In[4]:

resize()


# In[ ]:



