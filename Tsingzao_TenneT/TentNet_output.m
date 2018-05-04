function [OutVid OutVidIdx] = TentNet_output(InVid, InVidIdx, PatchSize, NumFilters, V)
% Computing TentNet filter outputs
% ======== INPUT ============
% InVid         Input videos (cell structure); each cell can be either a matrix (Gray) or a 3D tensor (RGB)   
% InVidIdx      Video index for InVid (column vector)
% PatchSize     Patch size (or filter size); the patch is set to be sqaure
% NumFilters    Number of filters at the stage right before the output layer 
% V             TentNet filter banks (cell structure); V{i} for filter bank in the ith stage  
% ======== OUTPUT ===========
% OutVid           filter output (cell structure)
% OutVidIdx        Video index for OutVid (column vector)
% ===========================

% [VidZ ~]  = size(InVid);
% mag    = (PatchSize-1)/2;
% OutVid = cell(NumFilters*VidZ,1);
% cnt = 0;
% for i = 1 : VidZ
%     im = [];
%     for k = 1 : VidF
%         [VidX, VidY, NumChls] = size(InVid{i,k});
%         vid = zeros(VidX+PatchSize-1,VidY+PatchSize-1,NumChls);
%         vid((mag+1):end-mag,(mag+1):end-mag,:) = InVid{i,k};
%         im = [im; im2col_mean_removal(vid,[PatchSize PatchSize])]; % collect all the patches of the ith image in a matrix, and perform patch mean removal
%     end
%     for j = 1 : NumFilters
%         cnt = cnt + 1;
%         temp = reshape(V(:,j,:),PatchSize^2,VidF);
%         OutVid{cnt} = reshape(temp(:)'*im, VidX, VidY);
%     end
%     InVid{i} = [];
% end
% OutVidIdx = kron(InVidIdx, ones(NumFilters,1));

VidZ = length(InVid);
mag = (PatchSize-1)/2;
OutVid = cell(NumFilters*VidZ,1); 
cnt = 0;
for i = 1:VidZ
    [VidX, VidY, VidF] = size(InVid{i});
    vid = zeros(VidX+PatchSize-1,VidY+PatchSize-1, VidF+PatchSize-1);
    vid((mag+1):end-mag,(mag+1):end-mag,(mag+1):end-mag) = InVid{i};     
    im = im2col_mean_removal(vid,[PatchSize PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    for j = 1:NumFilters
        cnt = cnt + 1;
        OutVid{cnt} = reshape(V(:,j)'*im,VidX, VidY, VidF);  % convolution output
    end
    InVid{i} = [];
end
OutVidIdx = kron(InVidIdx,ones(NumFilters,1)); 