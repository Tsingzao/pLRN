function best_C = parameter_selection(TrainLabel,TrainVec)

%grid of parameters
folds = 5; 
C = -5:1:10; 
%# grid search, and cross-validation 
cv_acc = zeros(length(C),1); 
for i=1:length(C)   
    fprintf('\n ====== %d/%d ======== \n',[i length(C)])
    cv_acc(i) = train(TrainLabel,TrainVec, sprintf('-c %f -v %d -q', 2^C(i), folds));
end
%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc); 
best_C = 2^C(idx);

%%
% %grid of parameters
% folds = 5; 
% [C,gamma] = meshgrid(-5:2:15, -15:2:3); 
% %# grid search, and cross-validation 
% cv_acc = zeros(numel(C),1); 
% d= 2;
% for i=1:numel(C)   
%     cv_acc(i) = train(TrainLabel,TrainVec, ...          
%         sprintf('-c %f -g %f -v %d -t %d', 2^C(i), 2^gamma(i), folds,d));
% end
% %# pair (C,gamma) with best accuracy
% [~,idx] = max(cv_acc); 
% %# contour plot of paramter selection 
% contour(C, gamma, reshape(cv_acc,size(C))), colorbar
% hold on;
% text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...  
%     'HorizontalAlign','left', 'VerticalAlign','top') 
% hold off 
% xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy') 
% %# now you can train you model using best_C and best_gamma
% best_C = 2^C(idx); best_gamma = 2^gamma(idx);
