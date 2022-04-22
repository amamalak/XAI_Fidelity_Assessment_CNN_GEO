
% This script generates a synthetic dataset for benchmarking XAI methods
% as in Mamalakis et al. 2022
% NOTE: if this script is run it will produce a different benchmark dataset than
% the one in the paper (due a different seed in the random generator).
% The dataset consists of images of square frames and circular frames of random number and size
% (within some limits) and of random positioning in the image.
% The difference of the area of all square frames minus all circular frames
% for each image is also calculated and saved.
 
% citation: 
% Mamalakis, A., E.A. Barnes, I. Ebert-Uphoff (2022) “Investigating the fidelity of explainable 
% artificial intelligence methods for application of convolutional neural networks in geoscience,” 
% arXiv preprint https://arxiv.org/abs/2202.03407. 

% Editor: Dr Antonios Mamalakis (amamalak@colostate.edu)

clear 
clc
close all

%% GENERATING THE DATASET

% SET PARAMETERS (USER-DEFINED)

L1=65; % size of the image on the vertical (number of pixels)
L2=65; % size of the image on the horizontal
[xx, yy] = meshgrid(1:L2,1:L1); % x and y coordinates of each pixel


N=500000; % number of samples (i.e., number of images)

nn=2;% max number of squares (or circles) per picture
lmax=22; lmin=12; % max/min side length per square


f = waitbar(0, 'Starting'); % starting the generation process


A=zeros(L1,L2,N); % creating N blank images
i=1;
while i<=N
    
    index=2;
    
    n(i,:)=randsample([0:nn],2); %randomly picking the number of squares and circles 
        
    while index>=2
        
        clear l_sq l_cir O_sq1 O_sq2 O_cir1 O_cir2 
        
        l_sq=randsample([lmin:lmax],n(i,1),true)'; % randomly picking the side length of the square
        l_cir=randsample([lmin:lmax]./sqrt(pi),n(i,2),true)'; % randomly picking the radius of the circle
        
        % randomly positioning the squares and the circles
        O_sq1=randsample([lmax/2+2:L1-lmax/2-1],length(l_sq))'; % position of the centre of the square (veritical coordinate)
        O_sq2=randsample([lmax/2+2:L2-lmax/2-1],length(l_sq))'; % position of the centre of the square (horizontal coordinate)
        
        O_cir1=randsample([lmax/sqrt(pi):L1-lmax/sqrt(pi)],length(l_cir))'; % position of the centre of the circle (vertical coordinate)
        O_cir2=randsample([lmax/sqrt(pi):L2-lmax/sqrt(pi)],length(l_cir))'; % position of the centre of the circle (horizontal coordinate)

        % adding the squares and the circles in a blank image
        A_pseudo=zeros(L1,L2);

        for j=1:length(l_sq) % adding all squares
            A_pseudo(O_sq1(j)-l_sq(j)/2-1:O_sq1(j)+l_sq(j)/2,O_sq2(j)-l_sq(j)/2-1:O_sq2(j)+l_sq(j)/2)=A_pseudo(O_sq1(j)-l_sq(j)/2-1:O_sq1(j)+l_sq(j)/2,O_sq2(j)-l_sq(j)/2-1:O_sq2(j)+l_sq(j)/2)+1;
        end
        for j=1:length(l_cir) % adding all cirlces
            A_pseudo(((xx-O_cir2(j)).^2+(yy-O_cir1(j)).^2)<=(l_cir(j)+1)^2)=A_pseudo(((xx-O_cir2(j)).^2+(yy-O_cir1(j)).^2)<=(l_cir(j)+1)^2)+1; 
        end

        index=max(max(A_pseudo)); % if index is equal or higher than 2 then there is object overlap. The loop is repeated.
    
    end % if index is less than 2 then we continue below.
    
    
    % Now that we are sure there is no overlap, we add the objects again to
    % a blank image but now: square frames are 1 and circular frames are -1.
   
    A_pseudo=zeros(L1,L2);
    for j=1:length(l_cir) % up to the outer side of the ring frame
        A_pseudo(((xx-O_cir2(j)).^2+(yy-O_cir1(j)).^2)<=(l_cir(j))^2)=A_pseudo(((xx-O_cir2(j)).^2+(yy-O_cir1(j)).^2)<=(l_cir(j))^2)-1; 
    end
    for j=1:length(l_cir) % up to the inner side of the ring frame
        rt=randsample([(lmin-5):l_cir(j)*sqrt(pi)-4]./sqrt(pi),1);
        A_pseudo(((xx-O_cir2(j)).^2+(yy-O_cir1(j)).^2)<=rt^2)=A_pseudo(((xx-O_cir2(j)).^2+(yy-O_cir1(j)).^2)<=rt^2)+1; 
        clear rt
    end
    
    A(:,:,i)=A_pseudo;
    clear A_pseudo
    
    for j=1:length(l_sq) % up to the outer side of the square frame
        A(O_sq1(j)-l_sq(j)/2:O_sq1(j)+l_sq(j)/2-1,O_sq2(j)-l_sq(j)/2:O_sq2(j)+l_sq(j)/2-1,i)=A(O_sq1(j)-l_sq(j)/2:O_sq1(j)+l_sq(j)/2-1,O_sq2(j)-l_sq(j)/2:O_sq2(j)+l_sq(j)/2-1,i)+1;
    end
    for j=1:length(l_sq) % up to the inner side of the square frame
        rt=randsample([(lmin-5):l_sq(j)-4],1);
        A(O_sq1(j)-rt/2:O_sq1(j)+rt/2-1,O_sq2(j)-rt/2:O_sq2(j)+rt/2-1,i)=A(O_sq1(j)-rt/2:O_sq1(j)+rt/2-1,O_sq2(j)-rt/2:O_sq2(j)+rt/2-1,i)-1;
        clear rt
    end
    
    
    clear l_sq l_cir O_sq1 O_sq2 O_cir1 O_cir2 
    
    % final check
    % if the area of all circular frames is exactly equal to that of all square frames, then the entire loop is repeated. 
    % otherwise, we move to generate the next image.
    if sum(sum(A(:,:,i)))==0
         i=i; % repeat entire loop
         A(:,:,i)=zeros(L1,L2,1); 
     else
         i=i+1; % move to next image
    end
    
    waitbar(i/N, f, sprintf('Progress: %d %%', floor(i/N*100)));
    pause(0.1);
    
end

close(f)

%% UN-COMMENT IF YOU WANT TO ADD IMAGES WHERE THE SQUARE FRAMES ARE INSIDE THE CIRCULAR ONES OR VICE VERSA

% NN=5; %number of controlled samples
%
% A(:,:,end-NN+1:end)=zeros(L1,L2,NN);
%
% i=size(A,3)-NN+1;
%
% while i<=N
%     
%     n(i,1:2)=ones(1,2);
%         
%     clear l_frame O_sq1 O_sq2 O_cir1 O_cir2 
%         
%     l_frame=randsample([lmin,lmax],2)';
%     l_frame(1)=l_frame(1)/sqrt(pi);% for the circle    
%     
%     O_sq1=L1/2; O_sq2=L2/2;
%     O_cir1=L1/2; O_cir2=L2/2;
% 
%     
%     A_pseudo=zeros(L1,L2);
%     % up to the outer side of the ring frame
%     A_pseudo(((xx-O_cir2).^2+(yy-O_cir1).^2)<=(l_frame(1))^2)=A_pseudo(((xx-O_cir2).^2+(yy-O_cir1).^2)<=(l_frame(1))^2)-1;
%     % up to the inner side of the ring frame
%     rt=l_frame(1)-2;
%     A_pseudo(((xx-O_cir2).^2+(yy-O_cir1).^2)<=rt^2)=A_pseudo(((xx-O_cir2).^2+(yy-O_cir1).^2)<=rt^2)+1; 
%     clear rt
%     A(:,:,i)=A_pseudo;
%     clear A_pseudo
%     
%     % up to the outer side of the square frame
%     A(O_sq1-l_frame(2)/2:O_sq1+l_frame(2)/2-1,O_sq2-l_frame(2)/2:O_sq2+l_frame(2)/2-1,i)=A(O_sq1-l_frame(2)/2:O_sq1+l_frame(2)/2-1,O_sq2-l_frame(2)/2:O_sq2+l_frame(2)/2-1,i)+1;
%     % up to the inner side of the square frame
%     rt=l_frame(2)-2;
%     A(O_sq1-rt/2:O_sq1+rt/2-1,O_sq2-rt/2:O_sq2+rt/2-1,i)=A(O_sq1-rt/2:O_sq1+rt/2-1,O_sq2-rt/2:O_sq2+rt/2-1,i)-1;
%     clear rt
%     
%     clear l_frame O_sq1 O_sq2 O_cir1 O_cir2 
%     
%     if sum(sum(A(:,:,i)))==0
%          i=i;
%          A(:,:,i)=zeros(L1,L2,1);
%      else
%          i=i+1;
%     end
%     
% end
% 
% 
% for i=1:NN
%    figure(i)
%    image(A(:,:,end-NN+i),'CDataMapping','scaled')
%    colorbar 
%    caxis([-1 1])
%    title(['sample #',num2str(size(A,3)-NN+i)])
% end


%% UN-COMMENT IF YOU WANT TO ADD AN IMAGE WHICH IS THE FLIPPED VERSION OF ANOTHER ONE

% id_flip=max(find((n(:,1)==1) .* (n(:,2)==2) == 1))
% A(:,:,id_flip+1)=flip(flip(A(:,:,id_flip),1),2);
% 
% figure()
%    image(A(:,:,id_flip),'CDataMapping','scaled')
%    colorbar 
%    caxis([-1 1])
% figure()
%    image(A(:,:,id_flip+1),'CDataMapping','scaled')
%    colorbar 
%    caxis([-1 1])

%% PLOTTING SOME IMAGES RANDOMLY TO GET A "SENSE" OF THE DATASET
% not used in the paper

plotid=sort(randsample([1:N],10)');

for i=1:length(plotid)
   figure(i)
   image(A(:,:,plotid(i)),'CDataMapping','scaled')
   colorbar 
   caxis([-1 1])
   title(['sample #',num2str(plotid(i))])
end


%% MORE PLOTS ANS STATS

X=abs(A); % the synthetic input into the network
y=squeeze(sum(sum(A,1),2)); % the synthetic output to be learned by the network

Cnt=A.*repmat(permute((y>0)*2-1,[3 2 1]),L1,L2,1);
lat=[1:L1]'; % pseudo latitude
lon=[1:L2]'; % pseudo longitude

% printing some stats
sum(y>0)
sum(y==0)
sum(y<0)

figure() % not used in the paper
autocorr(y)

figure() % not used in the paper
hist(n(:,1)) % histogram of number of squares per image

figure() % not used in the paper
hist(n(:,2)) % histogram of number of squares per image

% printing the probability of having different combinations of numbers of squares/circles 
sum((n(:,1)==0) .* (n(:,2)==0))/N % probability of zero squares and zero circles (blank image)
sum((n(:,1)==0) .* (n(:,2)==1))/N % probability of zero squares and 1 circle in the image
sum((n(:,1)==0) .* (n(:,2)==2))/N % probability of zero squares and 2 circles in the image
sum((n(:,1)==1) .* (n(:,2)==0))/N
sum((n(:,1)==1) .* (n(:,2)==1))/N
sum((n(:,1)==1) .* (n(:,2)==2))/N
sum((n(:,1)==2) .* (n(:,2)==0))/N
sum((n(:,1)==2) .* (n(:,2)==1))/N
sum((n(:,1)==2) .* (n(:,2)==2))/N

%% SAVING DATASET

save('synth_data_shapes.mat','lat','lon','y','X','Cnt','id_flip','-v7.3')



