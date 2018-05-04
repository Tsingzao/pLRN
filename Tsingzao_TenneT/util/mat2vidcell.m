function Vid = mat2vidcell(D,height,width,ImgFormat)

[M N F] = size(D);
if strcmp(ImgFormat,'gray')
    Vid = cell(N,1);
    for i = 1:N
        Vid{i,1} = reshape(D(:,i,:),height,width,F);
    end
end