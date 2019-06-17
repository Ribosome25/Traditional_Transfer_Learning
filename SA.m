clear;

BM_id =10;
BioMarkerID = [];
BaselineError = [];
SAError = [];

for BM_id = 1:16

load(strcat('./data/stdX/MB_',num2str(BM_id),'.mat'));
% load(strcat('./data/rawX/Data_BM_',num2str(BM_id),'.mat'));
fprintf("==============\nBM_%d\n",BM_id);
Xs = zscore(X2);
Xt = zscore(X1);
Ys = Y2;
Yt = Y1;
%%
subspace_dim_d = 80;
[Xss,~,~] = pca(Xs);
[Xtt,~,~] = pca(Xt);
PCs = Xss(:,1:subspace_dim_d);
PCt = Xtt(:,1:subspace_dim_d);

Target_Aligned_Source_Data = Xs*(PCs * PCs'*PCt);
Target_Projected_Data = Xt*PCt;

newS = Target_Aligned_Source_Data;
newT = Target_Projected_Data;

%% Baseline
rng(10);
Mdl_bsl = TreeBagger(200,zscore(Xs),Ys,'Method','regression');

Y_bsl = predict(Mdl_bsl,zscore(Xt));
bsl_err = NRMSE(Y_bsl,Yt); 
fprintf("Baseline error = %f \n",bsl_err);

% %% PCA Baseline
% 
% Xs = Xs*PCs;
% Xt = Xt*PCt;
% rng(0);
% Mdl_bsl_pca = TreeBagger(200,zscore(Xs),Ys,'Method','regression');
% 
% Y_bsl_pca = predict(Mdl_bsl_pca,zscore(Xt));
% bsl_err_pca = NRMSE(Y_bsl_pca,Yt); 
% fprintf("Baseline error using PCA = %f \n",bsl_err_pca);
%%
rng(10); % For reproducibility
Mdl_tca = TreeBagger(200,newS,Ys,'Method','regression');
Y_tca = predict(Mdl_tca,newT);
tca_err = NRMSE(Y_tca,Yt);
fprintf(" SA error = %f\n",tca_err);
%% Save
BioMarkerID(end+1) = BM_id;
BaselineError(end+1) = bsl_err;
SAError(end+1) = tca_err;

end
%% Print
BioMarkerID(end+1) = 999;
BaselineError(end+1) = mean(BaselineError);
SAError(end+1) = mean(SAError);

BioMarkerID = BioMarkerID';
BaselineError = BaselineError';
SAError = SAError';
T = table(BioMarkerID,BaselineError,SAError)
%%
function err = NRMSE(Y_Predict,Y_Target)
    Y_Bar = mean(Y_Target);
    Nom = sum((Y_Predict - Y_Target).^2);
    Denom = sum((Y_Bar - Y_Target).^2);
    err = sqrt(Nom/Denom);
end