import os
from PIL import Image
#from resizeimage import resizeimage
i=0
for ss in os.listdir("./train2014"):
	if "COCO_train2014_" in ss:
		with open("./train2014/"+ss, 'r+b') as f:
			with Image.open(f) as image:
				cover = image.resize((256,256), Image.BICUBIC)
				#cover = resizeimage.resize_cover(image, [256, 256])
				cover.save("/home/shivsundram/train2014_resized/"+ss, image.format)
				#print "saving", "train2014_const/"+ss

		i+=1
	if i%100==0:
		perc = float(i)/float(82783)
		print(str(100*perc)+"% done")
	#if i>5:
	#	break
