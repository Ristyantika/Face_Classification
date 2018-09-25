clc;clear;close all;
 
image_folder = 'Dataset Wajah';
filenames = dir(fullfile(image_folder, '*.jpg'));
total_object = numel(filenames);

rata = {1,2,3,4,5,6,7,8,'class'};
for n = 1:total_object
    name = filenames(n).name;
    for j=1:length(name)
        if name(j) == '.'
            break;
        end
    end
    class = name(1:j-1);
    %features(n+1,5) = cellstr(class);

    full_name= fullfile(image_folder, name);
    
    img = imread(full_name);
    level = rgb2gray(img);
    citraFace = histeq(level);
    citra = imbinarize(citraFace);
    
    %imshow(citrairis);
    offset_GLCM = [0 1; -1 1; -1 0; -1 -1];
    offset = [1*offset_GLCM ; 2*offset_GLCM; 3*offset_GLCM];
    [Grauwertmatrix, S] = graycomatrix(citra,'NumLevels', 2, 'GrayLimits', [], 'Offset',offset);
    GrauwertStats = graycoprops(Grauwertmatrix);
    rata(n+1,:) = {GrauwertStats.Contrast(1),GrauwertStats.Correlation(1),GrauwertStats.Energy(1),GrauwertStats.Homogeneity(1),GrauwertStats.Contrast(2),GrauwertStats.Correlation(2),GrauwertStats.Energy(2),GrauwertStats.Homogeneity(2),class};
end
imshow(citra);
filename = 'glcm_libari.csv';
cell2csv(filename,rata);
save glcm_librari.mat;