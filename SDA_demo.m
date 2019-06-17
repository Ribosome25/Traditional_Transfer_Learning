clear;
Xs = mvnrnd([1 3],[1 0; 0 1],600);
Xt = mvnrnd([4 1],[10 -5; -5 5],400);
Xs = zscore(Xs);
Xt = zscore(Xt);
% %%
% BM_id = 1;
% 
% load(strcat('./data/stdX/MB_',num2str(BM_id),'.mat'));
% % load(strcat('./data/rawX/Data_BM_',num2str(BM_id),'.mat'));
% fprintf("==============\nBM_%d\n",BM_id);
% Xs = zscore(X2);
% Xt = zscore(X1);
% Ys = Y2;
% Yt = Y1;
%%
subspace_dim_d = 2;
[Xss,~,Ess] = pca(Xs);
[Xtt,~,Ett] = pca(Xt); % the E is Principal component variances; already sorted
PCs = Xss(:,1:subspace_dim_d);
PCt = Xtt(:,1:subspace_dim_d);
PEs = (Ess(1:subspace_dim_d));
PEt = (Ett(1:subspace_dim_d));

Ms = PCs * (PCs'*PCt) * diag((PEs.^-0.5 .* PEt.^0.5)) ;
Mt = PCt;
%
newS = Xs * Ms;
newT = Xt * Mt;

%%
std(newS)
std(newT)
std(Xs*PCs)
std(Xt*PCt)
%%
close all
figure
hold on
scatter(Xs(:,1),Xs(:,2),'r.')
scatter(Xt(:,1),Xt(:,2),'b.')
% plot([0,Ms(1)],[0,Ms(2)],'r','LineWidth',3)
% plot([0,Mt(1)],[0,Mt(2)],'b','LineWidth',3)
title ('Standardizd data')
% hold on
% scatter(newS(:,1),newS(:,2),'r.')
% scatter(newT(:,1),newT(:,2),'b.')
% 
[f1,xi1] = ksdensity(newS(:,1));
[f2,xi2] = ksdensity(newT(:,1));
figure;
plot(xi1,f1,xi2,f2)
title("Distribution after matching to 1d")