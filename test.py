from PIL import Image, ImageChops
import cv2
import numpy as np
from resize import resize
import myfunx

def fun():
	#resize()
	x = myfunx.run()
	x = np.array(x)
	data = x
	image = Image.fromarray(data, 'RGB')
	new_size = (1465, 600)
	image = image.resize(new_size)
	#image = ImageChops.invert(image)
	image.save('media/new.png')
	#image.show()
