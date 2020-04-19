% If you use this code, please cite this paper: Chongyi Li, Jichang Guo,Runmin Cong, Yanwei Pang, Bo Wang,
%“Underwater image enhancement by dehazing with minimum information loss and histogram distribution prior”,
%IEEE Transactions on Image Processing, 25(12), pp. 5664-5677 (2016). 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dehazed: underwater image dehazing results
% dehazed_exposure: dehazed results processed by adaptive exposure
% HE_prior: final result by minimum information loss and histrogram distribution prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
% images are returned with absolute path
% add your path
addpath('D:\论文\水下图像\程序\TIP2016-code李崇义\Dependencies');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [fn,pn,fi]=uigetfile('*.bmp;*.jpg;*.png','pick up one picture');
% str0='D:\img\OutputImages\';
% save_path=[str0,fn];

imgPath = 'D:\img\InputImages\'; % 图像库路径
imgDir = dir([imgPath '*.jpg']); % 遍历所有jpg格式文件
tic
for i = 1:length(imgDir) % 遍历结构体就可以一一处理图片了
    s = strsplit(imgDir(i).name,'.');
    name = [char(s(1)),'_li.jpg'];
    img_input = imread([imgPath imgDir(i).name]); %读取每张图片
    fprintf(imgDir(i).name);
    fprintf('\n');
    [dehazed, dehazed_exposure, HE_prior]=master(img_input);
    %figure,imshow([img_input,dehazed,dehazed_exposure,HE_prior]);
    %figure,imshow([img_input,HE_prior]);
    imwrite(HE_prior, fullfile('D:\img\OutputImages\', name));
end
% tic
% img_input=imread([pn fn]);
% [dehazed, dehazed_exposure, HE_prior]=master(img_input);
% %figure,imshow([img_input,dehazed,dehazed_exposure,HE_prior]);
% %figure,imshow([img_input,HE_prior]);
% imwrite(HE_prior,save_path);
mytimer1=toc;
disp(mytimer1)