# this code is used to take photos from a specified folder and
# divide them into multiple folders as we defined below
# the division is according to the image's name, so be sure
# to make all the image names has the same format and in an appropriate sequence



# organize imports
import os
import glob
import datetime

# print start time
print ("[INFO] program started on - " + str(datetime.datetime.now))

# get the input and output path of your dataset
# in order for the code to work appropriately, the defined path must be correct
# and the images should be directly inside the input_path folder
input_path  = "D:\\AIProject\\flowers\\"
output_path = "D:\\AIProject\\flower-dataset\\train"

# get the class label limit
class_limit = 17

# take all the images from the dataset
image_paths = glob.glob(input_path + "\\*.jpg")

# variables to keep track
label = 0
i = 0
j = 80

# define your classes here
class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
			   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
			   "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
			   "windflower", "pansy"]

# change the current working directory
os.chdir(output_path)

# loop over the class labels
for x in range(1, class_limit+1):
	# create a folder for that class
	os.system("mkdir " + class_names[label])
	# get the current path
	cur_path = output_path + "\\" + class_names[label] + "\\"
	# loop over the images in the dataset
	for image_path in image_paths[i:j]:
		original_path = image_path
		image_path = image_path.split("\\")
		image_path = image_path[len(image_path)-1]
		os.system("copy " + original_path + " " + cur_path + image_path)
	i += 80
	j += 80
	label += 1

# print end time
print ("[INFO] program ended on - " + str(datetime.datetime.now))