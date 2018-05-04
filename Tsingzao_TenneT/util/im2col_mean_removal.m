function im = im2col_mean_removal(varargin)
% 

InImg = varargin{1};
patchsize12 = varargin{2}; 

im = cell(1,1);
iim = im2colstep(InImg(:,:,:),patchsize12);
im = bsxfun(@minus, iim, mean(iim)); 
    
    