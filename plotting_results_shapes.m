% This script generates some of the plots of the paper by Mamalakis et al. 2022

% citation: 
% Mamalakis, A., E.A. Barnes, I. Ebert-Uphoff (2022) “Investigating the fidelity of explainable 
% artificial intelligence methods for application of convolutional neural networks in geoscience,” 
% arXiv preprint https://arxiv.org/abs/2202.03407. 


% Editor: Dr Antonios Mamalakis (amamalak@colostate.edu)


clear 
clc
close all

%% LOAD DATA/RESULTS

% deep taylor
ncid0= netcdf.open('DT.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'DT');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
DT = d0;
DT=permute(DT,[2,1,3]);
clear d0 ncid0 varid0

% LRP a1b0
ncid0= netcdf.open('LRP.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRP');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPab = d0;
LRPab=permute(LRPab,[2,1,3]);
clear d0 ncid0 varid0

% LRPz
ncid0= netcdf.open('LRPz.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRPz');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPz = d0;
LRPz=permute(LRPz,[2,1,3]);
clear d0 ncid0 varid0

% LRP composite
ncid0= netcdf.open('LRPseqA.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRPseqA');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPseqA = d0;
LRPseqA=permute(LRPseqA,[2,1,3]);
clear d0 ncid0 varid0

% LRP composite flat
ncid0= netcdf.open('LRPseqAflat.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRPseqAflat');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPseqAflat = d0;
LRPseqAflat=permute(LRPseqAflat,[2,1,3]);
clear d0 ncid0 varid0

% input*gradient
ncid0= netcdf.open('ItG.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'ItG');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
ItG = d0;
ItG=permute(ItG,[2,1,3]);
clear d0 ncid0 varid0

% integrated gradients
ncid0= netcdf.open('intG.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'intG');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
intG = d0;
intG=permute(intG,[2,1,3]);
clear d0 ncid0 varid0

% gradient
ncid0= netcdf.open('Grad.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'Grad');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
Grad = d0;
Grad=permute(Grad,[2,1,3]);
clear d0 ncid0 varid0

% smoooth gradient
ncid0= netcdf.open('SmooG.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'SmooG');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
SmooG = d0;
SmooG=permute(SmooG,[2,1,3]);
clear d0 ncid0 varid0

% PatternNet
ncid0= netcdf.open('PN.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'PN');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
PN = d0;
PN=permute(PN,[2,1,3]);
clear d0 ncid0 varid0

% PatternAttribution
ncid0= netcdf.open('PA.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'PA');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
PA = d0;
PA=permute(PA,[2,1,3]);
clear d0 ncid0 varid0

% predictions of the CNN
ncid0= netcdf.open('predictions.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'y_hat_NN');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
y_NN = d0;
clear d0 ncid0 varid0

% synthetic dataset
load('synth_data_shapes.mat')
[LON,LAT]=meshgrid(lon,lat);

%% XAI results

close all 
% pick a sample of the testing data
t= 345 % figure 4 in the paper
t= 3567 % figure 3 in the paper

% print y(t) and network prediction
y(450000+t)>0
y_NN(t)

Gtemp=Cnt(:,:,450000+t); 
Xtemp=X(:,:,450000+t); 

figure() % input
image(Xtemp,'CDataMapping','scaled')
colorbar 
caxis([-1 1])
colormap(bluewhitered)
title(['Input'])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp

figure() % ground truth of attribution
image(Gtemp,'CDataMapping','scaled')
colorbar 
caxis([-1 1])
colormap(bluewhitered)
title(['Ground Truth'])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp

%% Edge detection (for reviewer; not for paper)

BW1 = edge(Xtemp,'sobel');
BW2 = edge(Xtemp,'canny');
tiledlayout(1,2)

nexttile
imshow(BW1)
title('Sobel Filter')

nexttile
imshow(BW2)
title('Canny Filter')

%% plotting XAI results

%gradient
temp=Grad(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete'); % correlation
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete'); % rank correlation
title(['NN: Gradient;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%smooth gradient
temp=SmooG(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Smooth Gradients;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%patternnet
temp=PN(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Pattern Net;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%pattern attribution
temp=PA(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Pattern Attribution;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%LRPa1b0
temp=LRPab(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRP_a_1_b_0;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%LRPz
temp=LRPz(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRP_z;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%LRPcomposite
temp=LRPseqA(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRPseqA;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%LRPcompflat
temp=LRPseqAflat(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRPseqAflat;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%input*grad
temp=ItG(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Input*Grad;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%integrated Gradients
temp=intG(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: integrated Gradients;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%DeepTaylor
temp=DT(:,:,t);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Deep Taylor;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%% Tracking performance across many samples using correlation

close all

y_real=[(y(450001:end)'>0)]'*1;

j=0;
for i=1:100 %length(y_NN)
    
    if y_real(i)==y_NN(i)
        
        j=j+1;
    
        temp=reshape(Cnt(:,:,450000+i),[],1);
        

        id(j)=i;

        c_ItG(j)=corr(temp,reshape(ItG(:,:,i),[],1),'rows','complete');
        c_intG(j)=corr(temp,reshape(intG(:,:,i),[],1),'rows','complete');
        c_Grad(j)=corr(temp,reshape(Grad(:,:,i),[],1),'rows','complete');
        c_SmooG(j)=corr(temp,reshape(SmooG(:,:,i),[],1),'rows','complete');

        c_LRPab(j)=corr(temp,reshape(LRPab(:,:,i),[],1),'rows','complete');
        c_LRPz(j)=corr(temp,reshape(LRPz(:,:,i),[],1),'rows','complete');
        c_LRPseqA(j)=corr(temp,reshape(LRPseqA(:,:,i),[],1),'rows','complete');
        c_LRPseqAflat(j)=corr(temp,reshape(LRPseqAflat(:,:,i),[],1),'rows','complete');
        c_PN(j)=corr(temp,reshape(PN(:,:,i),[],1),'rows','complete');
        c_PA(j)=corr(temp,reshape(PA(:,:,i),[],1),'rows','complete');
        
        clear temp 
    
    end
end


%% plotting correlation histograms


figure()
histogram(c_ItG,'BinWidth', 0.025);
hold on
histogram(c_intG,'BinWidth', 0.025);
hold on
%histogram(c_Grad,'BinWidth', 0.025);
%hold on
%histogram(c_SmooG,'BinWidth', 0.025);
%hold on
%histogram(c_PN,'BinWidth', 0.025);
%hold on
histogram(c_PA,'BinWidth', 0.025);

figure()
histogram(c_LRPab,'BinWidth', 0.025);
hold on
histogram(c_LRPz,'BinWidth', 0.025);
hold on
histogram(c_LRPseqA,'BinWidth', 0.025);
hold on
histogram(c_LRPseqAflat,'BinWidth', 0.025);

%%

clear 
clc
close all

%% Load XAI results that explain the non-predicted class for the 345th testing sample

ncid0= netcdf.open('DT0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'DT0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
DT = d0;
DT=permute(DT,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('LRP0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRP0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPab = d0;
LRPab=permute(LRPab,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('LRPz0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRPz0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPz = d0;
LRPz=permute(LRPz,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('LRPseqA0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRPseqA0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPseqA = d0;
LRPseqA=permute(LRPseqA,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('LRPseqAflat0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'LRPseqAflat0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
LRPseqAflat = d0;
LRPseqAflat=permute(LRPseqAflat,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('ItG0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'ItG0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
ItG = d0;
ItG=permute(ItG,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('intG0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'intG0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
intG = d0;
intG=permute(intG,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('Grad0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'Grad0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
Grad = d0;
Grad=permute(Grad,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('SmooG0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'SmooG0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
SmooG = d0;
SmooG=permute(SmooG,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('PN0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'PN0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
PN = d0;
PN=permute(PN,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('PA0.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'PA0');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
PA = d0;
PA=permute(PA,[2,1,3]);
clear d0 ncid0 varid0

ncid0= netcdf.open('predictions.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'y_hat_NN');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
y_NN = d0;
clear d0 ncid0 varid0

load('synth_data_shapes.mat')
[LON,LAT]=meshgrid(lon,lat);

%%
close all 

t=345 %figure 5 in the paper
y(450000+t)>0
y_NN(t)

Gtemp=Cnt(:,:,450000+t); 
Xtemp=X(:,:,450000+t); 

figure() %input
image(Xtemp,'CDataMapping','scaled')
colorbar 
caxis([-1 1])
colormap(bluewhitered)
title(['Input'])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp

figure() %ground truth
image(-Gtemp,'CDataMapping','scaled')
colorbar 
caxis([-1 1])
colormap(bluewhitered)
title(['Ground Truth'])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp


%%

temp=PN(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Pattern Net;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=PA(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Pattern Attribution;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=LRPab(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRP_a_1_b_0;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=LRPz(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRP_z;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=LRPseqA(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRPseqA;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=LRPseqAflat(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: LRPseqAflat;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=ItG(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Input*Grad;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=intG(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: integrated Gradients;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=Grad(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Gradient;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=SmooG(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Smooth Gradients;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

temp=DT(:,:);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: Deep Taylor;  r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%% SHAP RESULTS FOR SPECIFIC SAMPLES

clc
clear
close all

ncid0= netcdf.open('SHAP.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'SHAP');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
SHAP = d0;
SHAP=permute(SHAP,[2,1,3,4]);
clear d0 ncid0 varid0

% synthetic dataset
load('synth_data_shapes.mat')
[LON,LAT]=meshgrid(lon,lat);

Gtemp=Cnt(:,:,450000+345); 
temp=SHAP(:,:,1,1);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: DEEP SHAP; r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

Gtemp=Cnt(:,:,450000+3567); 
temp=SHAP(:,:,2,2);
figure()
image(temp,'CDataMapping','scaled')
colorbar 
caxis([-max(max(abs(temp))) max(max(abs(temp)))])
colormap(bluewhitered)
r1=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'rows','complete');
r2=corr(reshape(temp,[],1),reshape(Gtemp,[],1),'Type','Spearman','rows','complete');
title(['NN: DEEP SHAP; r_t_r_u_t_h=',num2str(r1),';  rho_t_r_u_t_h=',num2str(r2)])
set(gca,'Yticklabel',[]) 
set(gca,'Xticklabel',[]) %to just get rid of the numbers but leave the ticks.
clear temp r1 r2

%% SHAP long results (over the first 100 testing samples); this is extra analysis

clc
clear
close all

ncid0= netcdf.open('SHAPlong.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'SHAP');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
SHAP = d0;
SHAP=permute(SHAP,[2,1,3,4]);
clear d0 ncid0 varid0

% predictions of the CNN
ncid0= netcdf.open('predictions.nc','NC_NOWRITE');
varid0= netcdf.inqVarID(ncid0,'y_hat_NN');
d0=netcdf.getVar(ncid0,varid0,'double');
netcdf.close(ncid0);
y_NN = d0;
clear d0 ncid0 varid0

% synthetic dataset
load('synth_data_shapes.mat')

y_real=[(y(450001:end)'>0)]'*1;

j=0;
for i=1:100 %length(y_NN)
    
    if y_real(i)==y_NN(i)
        
        j=j+1;
    
        temp=reshape(Cnt(:,:,450000+i),[],1);
        
        id(j)=i;

        c_shap(j)=corr(temp,reshape(SHAP(:,:,y_real(i)+1,i),[],1),'rows','complete');
        
        
        clear temp 
    
    end
end

figure()
histogram(c_shap, 'BinWidth', 0.025);
