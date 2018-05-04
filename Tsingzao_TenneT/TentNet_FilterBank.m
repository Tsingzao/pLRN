function Vo = TentNet_FilterBank(InVid, PatchSize, NumFilters)
% =======INPUT=============
% InVid            Input videos (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of TentNet filters in the bank.
% =======OUTPUT============
% Vo                TentNet filter banks, arranged in column-by-column manner
% =========================

[VidZ ~] = size(InVid);
NumRSamples = min(VidZ,100000);
RandIdx = randperm(VidZ);
RandIdx = RandIdx(1:NumRSamples);

%% Learning TentNet filters (V)
NumChls = 1;
Rx = zeros(NumChls*PatchSize^3,NumChls*PatchSize^3);

for i = RandIdx %1:ImgZ
    im = im2col_mean_removal(InVid{i},[PatchSize PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    Rx = Rx + im*im'; % sum of all the input images' covariance matrix
end
Rx = Rx/(NumRSamples*size(im,2));
[E D] = eig(Rx);
[~, ind] = sort(diag(D),'descend');
Vo = E(:,ind(1:NumFilters));  % principal eigenvectors 
