
% disp('Reading input distance matrix')
 % xx2=load('Data/ToyDataSetR5r.csv');
%   xx2=ZeroFiveNineFeature';
%xx2=load('Data/EllipsesDataPoints2.csv');
load('D:\SCI Paper4\Experiments\Datasets\WaterQuality\TestSet\Test12Attr.mat');
testClean=testSet;
noiseIDs=find(testSet(:,end)<6);
testClean(noiseIDs,:)=[];

load('D:\SCI Paper4\Experiments\Datasets\WaterQuality\TrainSet\Train12Attr.mat');

AttrNum=12;
xx2=TrainSet(:,1:AttrNum);%%%%% used for MNIST data
% 
% [ND,Dim]=size(xx2);


% for DimVar=2:Dim
%     StdAttr=std(xx2(:,DimVar));
%     MeanAttr=mean(xx2(:,DimVar));
%     Normalized=(xx2(:,DimVar)- MeanAttr)./StdAttr;
% %        MinAttr=min(ALLShape(:,DimVar));
% %       MaxAttr=max(ALLShape(:,DimVar));
% %     Normalized=(ALLShape(:,DimVar)- MinAttr)./(MaxAttr-MinAttr);
%    xx2(:,DimVar)= Normalized; 
%  end
%  
tic;
%percent=2; %%%0.2-2 is good for curve dataset
% percent=6; %%%6 is good for 7 clustered toydataset
percent=5; %%%8 is good for ZeroFiveNine USPS

%percent=8;
 alpha=0.5:0.001:0.502;


xx1=xx2;   
 %xx1=xx2(:,2:end);%%15:consider the dimension of population of each center
 [rowN,colN]=size(xx1);
 ND=rowN;
 

 xx1T=xx1'; 
X2 = sum(xx1T.^2,1);
distSQ = repmat(X2,rowN,1)+repmat(X2',1,rowN)-2*xx1T'*xx1T;
dist=sqrt(distSQ);
maxVal=max(max(dist));
for i=1:ND
    dist(i,i)=maxVal;
end
%% 确定 dc, xuji:percent的值越大，越容易将数据点划分为halo

      
fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);

minVal=min(min(dist));
 Interval=(maxVal-minVal);
 dc=minVal+0.01*Interval*percent;
 
for i=1:ND
    dist(i,i)=0;
end


%dc=max(xx(:,3));
%% 计算局部密度 rho (利用 Gaussian 核)

fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);

%% 将每个数据点的 rho 值初始化为零

% ND=800;
for i=1:ND
  rho(i)=0.;
end

% Gaussian kernel
for i=1:ND-1
  for j=i+1:ND
     rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
     rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
  end
end



%% 先求矩阵列最大值，再求最大值，最后得到所有距离值中的最大值
maxd=max(max(dist)); 

%% 将 rho 按降序排列，ordrho 保持序（徐计：保存原始数据点编号）
[rho_sorted,ordrho]=sort(rho,'descend');


%% 处理 rho 值最大的数据点
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;

%% 生成 delta 和 nneigh 数组
for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1
     if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj); 
        %% 记录 rho 值更大的数据点中与 ordrho(ii) 距离最近的点的编号 ordrho(jj)
     end
   end
end

%% 生成 rho 值最大数据点的 delta 值
delta(ordrho(1))=max(delta(:));

%% normalize delta  %% Why Normalize Delta 2016-6-15
% deltaNew=(delta-min(delta))/max(delta);
% delta=deltaNew;

%%[delta_sorted,orddelta]=sort(delta,'descend');
%%%%%If there two points with the same greatest rho
%%%%%Very rare situation
[delta_sorted,orddelta]=sort(delta);
if nneigh(orddelta(ND-1))==0
   temp=orddelta(ND-1);
   orddelta(ND-1)=orddelta(ND);
   orddelta(ND)=temp;
end

gamma=rho.*delta;
[gamma_sorted, ordGamma]=sort(gamma);

%%%==============================================================%%%%%%%%%
%%%%%%%%computing the objective function value of Optimal Granulation%%%%%


SumDist=sum(dist(:));



colors={'red','green','blue','cyan', 'black','magenta'};

figure(1)
subplot(1,5,1);
box on;
xlabel ('Number of granuels','FontSize',12.0); hold on;
ylabel ('Objective function value','FontSize',12.0); hold on;

 
[SortRhoAscend,ordrhoAscend]=sort(rho);
PossibleNgMax=100;
objFunVal=zeros(PossibleNgMax,1);
OptiNg=zeros(length(alpha),1);
OptFunVals=zeros(length(alpha),1);

  for IterCnt=1:length(alpha)
     fprintf('Now process alpha No. %d\n', IterCnt);
     AllGrCost =zeros(PossibleNgMax,1); 
      GrCost=zeros(PossibleNgMax,1);
      
   [remainPointSet,GrPointCollections ] = FindGranulePoints( nneigh,ordrhoAscend,ordGamma(end-PossibleNgMax+1:end) );%%% +1!!!!
       
for i=PossibleNgMax:-1:1 %%%%%Possible Number of granules
   if i<PossibleNgMax %%%%executed after the first round
    DegradedCenterID= ordGamma (end-i); 
    %[remainPointSet,GrPointCollections ] = IncreFindGranulePoints( nneigh,remainPointSet,GrPointCollections,DegradedCenterID );
   [GrPointCollections ] = IncreFindGranulePointsR2( nneigh,GrPointCollections,DegradedCenterID );
   
   end 
  % ObjElement=zeros(i,1);
   if i==PossibleNgMax %%%% for the finest grain, compute all
   for j=1:i %%%% the jth  granule
        currentGrSet=GrPointCollections{j};
        GrLen= length(currentGrSet);
%         sumPointsInGr=sum(xx1(currentGrSet,:));
%         VirtualCenter=sumPointsInGr./GrLen;%%%%center DistCost No good
       
       for k=2:GrLen %%%%the center is always the first points,elimilate it from the delta summation!!!
      %     GrCost(i-1)=GrCost(i-1)+norm(xx1(currentGrSet(k),:)-VirtualCenter(1,:));
         PoiInGrId= currentGrSet(k);
         GrCost(i)=GrCost(i)+delta(PoiInGrId);
       end
   end
   else
       GrCost(i)=  GrCost(i+1)+delta(DegradedCenterID);
   end 
   %objFunVal(i)=log(i)*alpha(IterCnt)+(1-alpha(IterCnt))*(sum(GrCost(i))); %%%Log for Curves Dataset 4-25
   objFunVal(i)=(i*0.1)*alpha(IterCnt)+(1-alpha(IterCnt))*GrCost(i); %%%sqrt 4-8
  
 end
  plot((1:PossibleNgMax),objFunVal,'Color',colors{IterCnt},'linewidth',1);  hold on;
 
    [minval,MinID]=min(objFunVal);
   OptiNg(IterCnt)=MinID;
  OptFunVals(IterCnt)=minval;
  
 end
toc;
 %hleg=legend('\alpha=0.3','\alpha=0.4','\alpha=0.5','\alpha=0.6');
 hleg=legend('\alpha=0.4','\alpha=0.5','\alpha=0.6');
[SortRhoAscend,ordrhoAscend]=sort(rho);
 %symbolsStrs=[ '*b'; 'dc';'+r';'og';'>k';'.b';'+b'];
%NgDisplay=min(7,OptiNg);
for alphaCnt=1:length(alpha)
   NgDisplay=OptiNg(alphaCnt);
 [remainPointSet,GrPointCollections ] = FindGranulePoints( nneigh,ordrhoAscend,ordGamma(end-NgDisplay+1:end) );
 %visualization of the granualtiong result

  % if alphaCnt==2
       % subplot(1,2,2);
      subplot(1,5,alphaCnt+1);
%      xlabel ('X','FontSize',12.0);hold on;
      xlabel(sprintf( 'alpha = %3.1f, Ng=%d',alpha(alphaCnt),NgDisplay),'FontSize',12.0); hold on;
%      for i=1:NgDisplay
% 
%         box on;
%     
%         plot(xx1(GrPointCollections{i},1),xx1(GrPointCollections{i},2),'MarkerFaceColor',[10+i*5,250-i*5,120]/255,'MarkerSize',5) ;
%        hold on;
%      %%%%%plot the center
%       plot(xx1(ordGamma(end-NgDisplay+1:end),1),xx1(ordGamma(end-NgDisplay+1:end),2),'ks','MarkerSize',6,'MarkerFaceColor','y') ;
%       hold on; 
%      end
  % end
end

Centers=ordGamma(end-OptiNg+1:end);



%%%%%%%%%%%%%Label Inference%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
% testSet=testSet(1:1000,:);
testClean=testClean(1:1000,:);
TestSize=size(testClean,1);
predictLabel=zeros(TestSize,1);
for i=1:TestSize
    predictLabel(i)=IncreLabelInferOLF(testClean(i,1:AttrNum),ND,rho,delta,nneigh,TrainSet,dc);
    if (mod(i,200)==0)
        fprintf('test instance No. %d\n',i);
    end
end
toc;
 myTotoalSqError=sum((predictLabel-testClean(:,end)).^2);
 
 
%  plot(predictLabel(1:100),'sr-.','linewidth',1.0);hold on;
%  plot(testSet(1:100,end),'^b-','linewidth',1.0);
  plot(predictLabel,'r-.','linewidth',1.0);hold on;
 plot(testClean(:,end),'b-','linewidth',1.0);
%   hleg=legend('Predicted Vaule','Real Value');
%      