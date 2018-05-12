fid = fopen('./testset/test.txt');

tline = fgetl(fid);
while ischar(tline)
    im = imread(strcat('./test/SegmentationClass/', tline, '.png'));
    save(strcat('./test/SegmentationClass_1D_255/', tline, '.mat'), 'im');
    [m, n] = size(im);
    for i = 1:m
        for j=1:n
            if im(i,j) == 255
                im(i,j) = 0;
            end
        end
    end
    save(strcat('./test/SegmentationClass_1D/', tline, '.mat'), 'im');
    tline = fgetl(fid);
end