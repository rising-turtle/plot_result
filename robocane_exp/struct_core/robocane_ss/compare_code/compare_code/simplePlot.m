% ddd1=importdata("/Users/jin/Downloads/ipad/02_08/ARposes.txt");
ddd1=importdata("/Volumes/BlackSSD/11_21_2020/mall/2020-11-22T12-44-26/ARposes.txt");
% ddd1=importdata("/Volumes/BlackSSD/ss2/comp/div/faceNearLong_est.csv");
% ddd1=importdata("/Volumes/BlackSSD/ss2/comp/div/faceFarshort_est.csv");
% ddd1=importdata("/Volumes/BlackSSD/ss2/comp/div/faceFarLong_est.csv");

% ddd1 = ddd1(9009:15828,:);
ddd1(:,1) = ddd1(:,1) - ddd1(1,1);
% figure,plot3(ddd1(:,2),ddd1(:,3),ddd1(:,4),'b','LineWidth',2),axis equal,grid minor

figure,patch(ddd1(:,2),ddd1(:,3),ddd1(:,4),ddd1(:,1),'edgecolor','flat','facecolor','none','LineWidth',3)
grid minor
view(3);colorbar
axis equal
set(gcf,'color','w');


xlabel("X (m)",'FontSize',20,'FontWeight','bold');
ylabel("Y (m)",'FontSize',20,'FontWeight','bold');
zlabel("Z (m)",'FontSize',20,'FontWeight','bold');

view(0,0)

%%

ddd1= importdata("/Users/jin/Q_Mac/work/temp/gt/corridor4_gt.csv");
figure,

subplot(3,1,1)
plot(ddd1(:,1),ddd1(:,2),'g'),grid minor
% hold on,plot(ddd1(:,1),ddd1(:,2),'go'),grid minor

subplot(3,1,2)
plot(ddd1(:,1),ddd1(:,3),'g'),grid minor

subplot(3,1,3)
plot(ddd1(:,1),ddd1(:,4),'g'),grid minor


%%
% clc
A = cursor_info1(4).Position;
B = cursor_info1(3).Position;
C = cursor_info1(2).Position;
D = cursor_info1(1).Position;
% E = cursor_info9(5).Position;
norm(A-B)
norm(B-C)
norm(C-D)
norm(D-A)
% norm(D-E)
% norm(A-E)
% norm(D-A)