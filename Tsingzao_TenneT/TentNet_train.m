function [f V BlkIdx] = TentNet_train(InVid,TentNet,IdtExt)
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
% IdtExt    a number in {0,1}; 1 do feature extraction, and 0 otherwise  
% =======OUTPUT============
% f         TentNet features (each column corresponds to feature of each video)
% V         learned TentNet filter banks (cell)
% BlkIdx    index of local block from which the histogram is compuated
% =========================
NumVid = size(InVid,1);

V = cell(TentNet.NumStages, 1);
OutVid = InVid;
VidIdx = (1:NumVid)';
clear InVid

for stage = 1:TentNet.NumStages
	display(['Computing TentNet filter bank and its outputs at stage ' num2str(stage) '...'])
   
	V{stage} = TentNet_FilterBank(OutVid, TentNet.PatchSize(stage), TentNet.NumFilters(stage)); % compute TentNet filter banks

    if stage ~= TentNet.NumStages % compute the TentNet outputs only when it is NOT the last stage
        [OutVid VidIdx] = TentNet_output(OutVid, VidIdx, ...
            TentNet.PatchSize(stage), TentNet.NumFilters(stage), V{stage});  
    end
end

if IdtExt == 1
    f = cell(NumVid, 1);
    for idx = 1 : NumVid
        if 0 == mod(idx,10)
            display(['Extracting TentNet feasture of the ' num2str(idx) 'th training sample...']); 
        end
        OutVidIndex = VidIdx==idx;
        [OutVid_i VidIdx_i] = TentNet_output(OutVid(OutVidIndex), ones(sum(OutVidIndex),1), ...
            TentNet.PatchSize(end), TentNet.NumFilters(end), V{end});  
        
        OutVid_i = TentNet_temporal_pooling(OutVid_i);
        [f{idx} BlkIdx] = HashingHist(TentNet,VidIdx_i,OutVid_i); % compute the feature of image "idx"
        
%        f{idx} = TentNet_AveragePooling(OutVid_i);
        
        OutVid(OutVidIndex) = cell(sum(OutVidIndex),1);
        
    end
    f = sparse([f{:}]);
else
    f=[];
    BlkIdx=[];
end

end
