ND=size(KaixianCountyPH,1);
AttrLen=5;
InstanceNum=ND-(AttrLen+1); %%%%AttrLen Attributes plus one label value.
Instances=zeros(InstanceNum,AttrLen+1);
for i=1:InstanceNum
   Instances(i,:)= KaixianCountyPH(i:i+AttrLen,2);
end
 p=randperm(InstanceNum);
 TrainSet=Instances(p(1:10000),:);
 testSet=Instances(p(10001:end),:);