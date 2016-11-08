
%%% This is the testing demo for gray image (Gaussian) denoising.
%%% Training data: 400 images of size 180X180

run('D:\Davey\matconvnet-1.0-beta23\matlab\vl_setupnn.m') ;
% clear; clc;
addpath('utilities');
%folderTest  = 'Train400';


showResult  = 1;
useGPU      = 1;
pauseTime   = 1;

JPEG_Quality = 50;   
net1.layers = {};
net2.layers = {};
%load ComCNN
load(fullfile('model','ComCNN_QF=50.mat'));
net1.layers = net.layers(1:end-1);
%load RecCNN
load(fullfile('model','RecCNN_QF=50.mat'));
net2.layers = net.layers(1:end-1);

%%% move to gpu
if useGPU
    net1 = vl_simplenn_move(net1, 'gpu') ;
    net2 = vl_simplenn_move(net2, 'gpu') ;
end

%read image
% label = imread('butterfly.bmp'); 
label = imread('Lena.png');

if size(label,3)>1
    label = rgb2gray(label);
end

label = im2single(label);

[hei,wid] = size(label);
if useGPU
    label = gpuArray(label);
end

% tic
res = vl_simplenn(net1,label,[],[],'conserveMemory',true,'mode','test');
Low_Resolution = res(end).x;
% toc

if useGPU
    Low_Resolution = gather(Low_Resolution);
end

imwrite(im2uint8(Low_Resolution),'compressed_image.jpg','jpg','Quality',JPEG_Quality);%Compression

im_input = im2single(imread('compressed_image.jpg'));
input = imresize(im_input,[hei,wid],'bicubic');

%%% convert to GPU
if useGPU
    input = gpuArray(input);
end
% tic
res    = vl_simplenn(net2,input,[],[],'conserveMemory',true,'mode','test');
output = input - res(end).x;
% toc
[PSNRWithNet, SSIMWithNet] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);

if useGPU
    output = gather(output);
    input  = gather(input);
end
imwrite(im2uint8(Low_Resolution),'output.jpg','jpg')
figure, imshow(label); title(sprintf('Raw-Input'));

figure, imshow(output); title(sprintf('After CNN Network, PSNR: %.2f dB,SSIM: %.4f', PSNRWithNet, SSIMWithNet));

if useGPU
    label = gather(label);
end
JPEG_Quality1= 10;
imwrite(label,'JPEG-Directly.jpg','jpg','Quality',JPEG_Quality1);%
im_direct = im2single(imread('JPEG-Directly.jpg'));
[PSNR_direct, SSIM_direct] = Cal_PSNRSSIM(im2uint8(label),im2uint8(im_direct),0,0);
figure, imshow(im_direct); title(sprintf('JPEG-Directly, PSNR: %.2f dB,SSIM: %.4f', PSNR_direct, SSIM_direct));



