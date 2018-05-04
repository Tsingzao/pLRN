function f = TentNet_FeatExt(InVid, V, TentNet)
% =======INPUT=============
% InVid      Input videos (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)  
% TentNet    TentNet parameters (struct)
%           .NumStages      
%               the number of stages in TentNet; e.g., 2  
%           .PatchSize
%               the patch size (filter size) for square patches; e.g., [5 3]
%               means patch size equalt to 5 and 3 in the first stage and second stage, respectively 
%           .NumFilters
%               the number of filters in each stage; e.g., [16 8] means 16 and
%               8 filters in the first stage and second stage, respectively
%           .HistBlockSize 
%               the size of each block for local histogram; e.g., [10 10]
%           .BlkOverLapRatio 
%               overlapped block region ratio; e.g., 0 means no overlapped 
%               between blocks, and 0.3 means 30% of blocksize is overlapped 
%           .Pyramid
%               spatial pyramid matching; e.g., [1 2 4], and [] if no Pyramid
%               is applied
% =======OUTPUT============
% f         TentNet features (each column corresponds to feature of each video)
% BlkIdx    index of local block from which the histogram is compuated
% =========================

NumVid = size(InVid,1);

OutVid = InVid;
VidIdx = (1:NumVid)';
clear InVid;
for stage = 1:TentNet.NumStages
    [OutVid VidIdx] = TentNet_output(OutVid, VidIdx, ...
        TentNet.PatchSize(stage), TentNet.NumFilters(stage), V{stage});
end

OutVid = TentNet_temporal_pooling(OutVid);
[f BlkIdx] = HashingHist(TentNet,VidIdx,OutVid);
%f = TentNet_AveragePooling(OutVid);