import os, numpy, random, cPickle, gzip, sys
from PIL import Image
from utils import Timer

directory = ''
pklname = ''
timer = Timer('Time to read image: ')

def file_names(directory):
	for index, subdirectory in enumerate(sorted(os.listdir(directory))):
		files = sorted(os.listdir(os.path.join(directory,subdirectory)))
		for file in files:
			file = os.path.join(directory,subdirectory,file)
			print file
			yield index, file

def images():
	for index, file in file_names(directory):
		img = Image.open(open(file))
		img = numpy.asarray(img, dtype='float64') / 256.0
		img = img.transpose(2, 0, 1).reshape(3 * 28 * 28)
		yield img, index

foods = ["apple","banana","biryani","bread","burger","cereal",
	"chiken","dosa","fries","idli","lemonrice","mango","milk",
	"omelette","orange","papaya","pineapple","pizza","pulao",
	"puri","rice","roti","samosa","water",]

def images_list():
	img_list = []
	for x, y in images():
		img_list.append((x,y))
	return numpy.asarray(img_list)

def split_train_valid_test(img_list):
	list_ = range(len(img_list))
	len_ = len(img_list)
	random.shuffle(list_)
	img_list = img_list[list_]

	split = int(0.8 * len(img_list))
	train_x, train_y  = img_list[:split,0], img_list[:split,1]
	test_x, test_y = img_list[split:,0], img_list[split:,1]
	valid_x, valid_y = img_list[len_-split:,0], img_list[len_-split:,1]

	print "train",train_x, train_y
	print 'shape', train_x.shape, train_y.shape
	print "test",test_x, test_y
	print 'shape', test_x.shape, test_y.shape
	print "valid", valid_x,valid_y
	print 'shape', valid_x.shape,valid_y.shape

	datasets = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

	print "saving file. please wait..."
	with gzip.open(pklname, 'w') as f:
		f.write(cPickle.dumps(datasets))

	print "Done..."

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print """usage: directory can be "dataset" or "dataset_mini"
		$ python create_pkl_gz_dataset.py --directory dataset_mini
		        --OR--
		$ python create_pkl_gz_dataset.py --directory dataset

		"""
		exit()

	directory = sys.argv[2]

	if (directory == 'dataset_mini'):
		pklname = 'food_mini.pkl.gz'
	elif(directory == "dataset"):
		pklname = 'food.pkl.gz'
	else:
		print "invalid dataset directory"
		exit()

	timer.start()
	split_train_valid_test(images_list())
	timer.stop()
	print timer.message()
