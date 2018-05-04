% ==== TentNet Demo =======
% Tingzhao Yu [tingzhao.yu@nlpr.ia.ac.cn]
% Please email me if you find bugs, or have suggestions or questions!
% ========================
clear all;
clc;
addpath('./tSVD','./Liblinear','./util');

VidSize   = 16;
VidLength = 16;
VidFormat = 'gray';
VidName   = 'Penn';

%% Load UCF Demo Data
str_res  = num2str(VidSize);
data     = permute(double(h5read(['/data/LowResolution/' VidName '_Train_' str_res '.h5'],'/data')),[5,4,3,2,1]);
feature  = permute(reshape(data, length(data), VidSize*VidSize, VidLength), [2,1,3]);
label    = double(h5read(['/data/LowResolution/' VidName '_Train_' str_res '.h5'],'/label'));

TrnData  = feature;
TrnLabel = label;

data     = permute(double(h5read(['/data/LowResolution/' VidName '_Test_' str_res '.h5'],'/data')),[5,4,3,2,1]);
feature  = permute(reshape(data, length(data), VidSize*VidSize, VidLength), [2,1,3]);
label    = double(h5read(['/data/LowResolution/' VidName '_Test_' str_res '.h5'],'/label'));

TestData = feature;
TestLabel= label;
clear feature label str_res data
% ==========================
nTestImg = length(TestLabel);

%% TentNet parameters
% Can be determined based on the validation set
TentNet.NumStages       = 2;
TentNet.PatchSize       = [3 3];
TentNet.NumFilters      = [10 10];
TentNet.HistBlockSize   = [5 5];
TentNet.BlkOverLapRatio = 0.5;
TentNet.Pyramid         = [];

fprintf('\n ========= TentNet Parameters ========== \n')
TentNet

%% TentNet Train

fprintf('\n ======== TentNet Training ============ \n')
TrnData_VidCell = mat2vidcell(TrnData,VidSize,VidSize,VidFormat); % convert columns in TrnData to cells 
clear TrnData
tic;
[featureTrain V] = TentNet_train(TrnData_VidCell, TentNet, 1);
TentNet_TrnTime = toc;
clear TrnData_VidCell;

fprintf('\n ====== Parameter Selection ======== \n')
tic;
best_C = parameter_selection(TrnLabel,featureTrain');
%best_C = 4;
ParameterSelection_Time = toc;

fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabel, featureTrain', ['-s 1 -c ',num2str(best_C),' -q']);
LinearSVM_TrnTime = toc;
clear ftrain;

%% TentNet Test

TestData_VidCell = mat2vidcell(TestData,VidSize,VidSize,VidFormat);

fprintf('\n ======= TentNet Testing ==========\n');
nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

predict_label = zeros(nTestImg,1);
tic;
for idx = 1:nTestImg
%     ftest = featureTrain(:,idx);
    ftest = TentNet_FeaExt(TestData_VidCell(idx,:),V,TentNet); % extract a test feature using trained TentNet model 
 
    [xLabel_est, accuracy, decision_values] = predict(TestLabel(idx),...
        sparse(ftest'), models, '-q'); % label predictoin by libsvm
   
    if xLabel_est == TestLabel(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 0==mod(idx,10); 
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end     
    
    predict_label(idx) = xLabel_est;
    
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 

%% Display Resluts
fprintf('\n ===== Results of TentNet, followed by a linear SVM classifier =====');
fprintf('\n     TentNet training time: %.2f secs.', TentNet_TrnTime);
fprintf('\n     Parameter Selection time: %.2f secs.', ParameterSelection_Time);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing Accuracy: %.2f%%', 100*Accuracy);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);
