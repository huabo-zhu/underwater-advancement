function img = load_image(num)
% num is the num of image with 

prefix = 'images/';
suffix = '.jpg';
path = [prefix,num,suffix]
img = imread(path);