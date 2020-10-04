from PIL import Image
import cv2
import numpy as np

def resize():
	
	im1 = Image.open(r'media/snapshot.png')
	im1 = im1.resize((1465, 600))
	im1.save('media/small_snapshot.png')

#resize()
