clc
clear all
close all

learn_rate = 0.07;

vid = VideoReader('mouse_mosse.mp4');
myvid = VideoWriter('mouse_tracked');
myvid_frame = VideoWriter('frame');
myvid_filter = VideoWriter('filter');

current_image = readFrame(vid);
grey = rgb2gray(current_image);
%%
ROW1 = 115;
ROW2 = 260;
COL1 = 265;
COL2 = 410;
SIZE = ROW2 - ROW1;
MID_ROW = floor((ROW2 - ROW1)/ 2);
MID_COL = floor((COL2 - COL1)/ 2);

filter = grey(ROW1:ROW2, COL1:COL2);
filter = imadjust(filter);
gaussian = fspecial('gaussian', max(size(filter)), 20);

grey_fft = fft2(filter);
filter_fft = fft2(filter);
gaussian_fft = fft2(gaussian);

A = gaussian_fft .* conj(filter_fft);
B = filter_fft .* conj(filter_fft);

trained_filter_fft = A ./ B;
trained_filter = ifftshift(ifft2(conj(trained_filter_fft)));

test_check_fft = grey_fft .* trained_filter_fft;
test_check = ifft2(test_check_fft);

[max_num, max_idx]=max(test_check(:));
[Y,X]=ind2sub(size(test_check),max_idx);

%box plots in col then rows (for some reason)
box = insertShape(current_image,'rectangle',[COL1 + Y - max(size(gaussian))/2, ROW1 + X - max(size(gaussian))/2,...
    max(size(gaussian)), max(size(gaussian))],'LineWidth',5);

norm_train = abs(trained_filter);
norm_train = (norm_train - min(norm_train(:)))*(1/(max(norm_train(:))-min(norm_train(:))));

open(myvid);
open(myvid_frame);
open(myvid_filter);

writeVideo(myvid, box);
writeVideo(myvid_frame, filter);
writeVideo(myvid_filter, double(norm_train));

%%
while hasFrame(vid)
    current_image = readFrame(vid);
    grey = rgb2gray(current_image);
    
    ROW1 = X - MID_ROW + ROW1 - 1;
    ROW2 = ROW1 + SIZE;
    COL1 = Y - MID_COL + COL1 - 1;
    COL2 = COL1 + SIZE;
    
    if COL1 < 1 || COL2 > size(current_image, 2)
        break
    end
    
    if ROW1 < 1 || ROW2 > size(current_image, 1)
        break
    end
    
    filter = grey(ROW1:ROW2, COL1:COL2);
    filter = imadjust(filter);
    
    grey_fft = fft2(filter);
    filter_fft = fft2(filter);
    gaussian_fft = fft2(gaussian);
    
    A = learn_rate*gaussian_fft .* conj(filter_fft) + (1 - learn_rate) * A;
    B = learn_rate*filter_fft .* conj(filter_fft) + (1 - learn_rate) * B;

    trained_filter_fft = A ./ B;
    trained_filter = ifftshift(ifft2(conj(trained_filter_fft)));
    
    test_check_fft = grey_fft .* trained_filter_fft;
    test_check = ifft2(test_check_fft);

    [max_num, max_idx]=max(abs(test_check(:)));
    [X,Y]=ind2sub(size(test_check),max_idx);

    if X < 2 || Y < 2
        break
    end
    

    box = insertShape(current_image,'rectangle',[COL1 + Y - max(size(gaussian))/2, ROW1 + X - max(size(gaussian))/2,...
    max(size(gaussian)), max(size(gaussian))],'LineWidth',5);

    norm_train = abs(trained_filter);
    norm_train = (norm_train - min(norm_train(:)))*(1/(max(norm_train(:))-min(norm_train(:))));

    writeVideo(myvid, box); 
    writeVideo(myvid_frame, filter);
    writeVideo(myvid_filter, double(norm_train));
end

close(myvid);
close(myvid_frame);
close(myvid_filter);