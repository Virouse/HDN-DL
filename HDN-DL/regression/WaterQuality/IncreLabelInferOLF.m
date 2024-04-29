function labelNew=IncreLabelInferOLF(newX,ND,rho,delta,nneigh,xx2,dc)
%newX=testSet(2,1:6);
%%%%% Super Bug is xx2(:,end ) is NOT the training Label!!!!!
distNewV=zeros(ND,1);
for i=1:ND
   thisV= xx2(i,1:end-1)-newX(1,:);
    distNewV(i)=norm(thisV);
    if  distNewV(i)==0 %%% query the exact sample labeled!
        labelNew=xx2(i,end);
        return;
    end
end

%%%%%update Rho
rhoCopy=rho;
RhoNew=0;
for i=1:ND
      distij=distNewV(i);
       rhoGain=exp(-(distij/dc)*(distij/dc));
      RhoNew=RhoNew+rhoGain;
        rhoCopy(i)=rhoCopy(i)+rhoGain;
end
deltaNew=max(distNewV);
NNeighNew=-1;

deltaCopy=delta;%%% isolate the effect of any one new datum
neighCopy=nneigh;
%%%%update Nneigh and Delta, incorporate the new datum into the OLF
%%%%% (a) find Nneigh(ND+1) and delta(ND+1)
for i=1:ND
   distij=distNewV(i);
   if rhoCopy(i)>=RhoNew && distij<deltaNew %%%%update nneigh and delta for newX
       NNeighNew=i;
       deltaNew=distij;
   end
   if  rhoCopy(i)<RhoNew && deltaCopy(i)>distij %%%%deltaCopy(i)<distij is wrong! update nneigh and delta for others
       neighCopy(i)=ND+1;
       deltaCopy(i)=distij;
   end
end
childrenINds=find (neighCopy==ND+1);

%%%%% Label inference with weighted summation
if ~isempty(childrenINds)
   WeightSum= sum(exp(-distNewV(childrenINds)));
 %WeightSum= sum(distNewV(childrenINds).^(-1));
  WeightedLabelSum=sum((exp(-distNewV(childrenINds))).*xx2(childrenINds,end));
  % WeightedLabelSum=sum((distNewV(childrenINds).^(-1)).*xx2(childrenINds,end));
   
 if WeightSum==0
      labelNew=sum(xx2(childrenINds,end))/length(childrenINds);
  else
   labelNew=WeightedLabelSum/WeightSum;
  end
else
   labelNew=xx2(NNeighNew,end);
%       brothersInds=find(neighCopy==NNeighNew);
%       if isempty(brothersInds)
%         labelNew=xx2(NNeighNew,end); %%%seems too simple to be accurate!
%       else
%   
%   ParentLabel=xx2(NNeighNew,end);
%   
%   if distNewV(NNeighNew)==0
%     labelNew=ParentLabel;
%     return;
%   end
%   
%  %  NewXWeight=exp(-distNewV(NNeighNew));
%      NewXWeight=1/distNewV(NNeighNew);
%    brotherWeightSum=sum(delta(brothersInds).^(-1))+NewXWeight;
%    weighV=delta(brothersInds).^(-1);
%    LabelV=xx2(brothersInds,end);
%     BroWeightedLabelSum=sum(weighV.*LabelV');
%    labelNew=(ParentLabel*brotherWeightSum-BroWeightedLabelSum)/NewXWeight; 
%       end
end
end

