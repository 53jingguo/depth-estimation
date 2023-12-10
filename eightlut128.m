close all;clear;clc
S='bs_1.jpg';% read any one image
f=imread(S); % S is a string (image's name)1024*512
f=imresize(f,[128,256]);   % if the origional image is too large, you can resize the image. ****Maybe you don't need it****
[h,w,td]=size(f);
rImg=zeros(h,w,td);
pai=3.14159;
kk=1;

yita = 360/256*(pi/180)*2;
left = [0 -sin(yita) cos(yita)]';
right = [0 sin(yita) cos(yita)]';
up = [-sin(yita) 0 cos(yita)]';
down = [sin(yita) 0 cos(yita)]';
left_up = [-sin(yita)*sin(pi/4) -sin(yita)*cos(pi/4) cos(yita)]';
left_down = [sin(yita)*sin(pi/4) -sin(yita)*cos(pi/4) cos(yita)]';
right_up = [-sin(yita)*cos(pi/4) sin(yita)*sin(pi/4) cos(yita)]';
right_down = [sin(yita)*cos(pi/4) sin(yita)*sin(pi/4) cos(yita)]';
% left_left_up = [-sin(yita)*cos(pi/8) -sin(yita)*sin(pi/8) cos(yita)]';
% left_up_up = [-sin(yita)*sin(pi/8) -sin(yita)*cos(pi/8) cos(yita)]';
% left_left_down = [sin(yita)*sin(pi/8) -sin(yita)*cos(pi/8) cos(yita)]';
% left_down_down = [sin(yita)*cos(pi/8) -sin(yita)*sin(pi/8) cos(yita)]';
% right_right_up = [-sin(yita)*sin(pi/8) sin(yita)*cos(pi/8) cos(yita)]';
% right_up_up = [-sin(yita)*cos(pi/8) sin(yita)*sin(pi/8) cos(yita)]';
% right_right_down = [sin(yita)*sin(pi/8) sin(yita)*cos(pi/8) cos(yita)]';
% right_down_down = [sin(yita)*cos(pi/8) sin(yita)*sin(pi/8) cos(yita)]';

LUT=zeros(h*w,2);
kk=1;
for i=1:h
    for j=1:w
        theta=(i-1)*180/(h-1)*(pai/180);%（0,pai）
        if theta==0
            theta = theta+1e-4;
        end
        if theta>3.1415
            theta = theta-1e-4;
        end
        fai=(j-1)*360/(w-1)*(pai/180);%（0,2*pai）
        if fai==0
            fai = fai+1e-4;
        end
        if fai>6.283
            fai = fai-1e-6;
        end
        X=sin(theta)*cos(fai);
        Y=sin(theta)*sin(fai);
        Z=cos(theta);
        if fai<pi
          roll = -theta;
          yaw = fai-(pi/2);
          pitch=0*(pi/180);
        end  
        if fai>pi
          roll = theta;
          yaw = (fai-pi)-(pi/2);
          pitch=0*(pi/180);
        end
        if fai==pi
           roll = 0;
           yaw = 0;
           pitch=-theta;
        end
        if fai==0
           roll = 0;
           yaw = 0;
           pitch=theta;
        end
        Rx=[1 0 0;0 cos(roll) -sin(roll);0 sin(roll) cos(roll)];
        Ry=[cos(pitch) 0 sin(pitch);0 1 0;-sin(pitch) 0 cos(pitch)];
        Rz=[cos(yaw) -sin(yaw) 0;sin(yaw) cos(yaw) 0;0 0 1];
        R=Rz*Ry*Rx;
        %R=Rx*Ry*Rz;%以前顺序乘反了
        Cor_new=R*[0 0 1]';
        new_X=Cor_new(1);
        new_Y=Cor_new(2);
        new_Z=Cor_new(3);

        theta_new = acos(new_Z);
        if new_X==0
            new_X = new_X+1e-15;
        end
        fai = atan2(new_Y,new_X);
        fai2 = fai;
        if fai2>0
            fai2 = 0;
        end
        if fai2<0
            fai2 = 2*pi;
        end
        fai_new = fai2+fai;
        x_new=fai_new*(w-1)/360*180./pi;

%         if x_new<=0
%             x_new=0;
%         end
        y_new=theta_new*(h-1)/180*180./pi;
%         if y_new<=0
%             y_new=0;
%         end
        %rImg(i,j,:)=f(y_new,x_new,:);
        
%         if j==1
%             x_new=-1;
%         else
        x_new = (x_new - 127.5)/127.5;
        y_new = (y_new - 63.5)/63.5;
        LUT(kk,:)=[x_new,y_new];%已经完成了归一化，并且x和y的坐标没错乱，不需要在使用的时候交叉赋值了
        kk=kk+1;
% % % %         if new_Z>0
% % % %             theta_new=atan(sqrt(new_X.^2+new_Y.^2)/new_Z);
% % % %         else
% % % %             theta_new=pi-atan(-sqrt(new_X.^2+new_Y.^2)/new_Z);
% % % %         end
% % % %         if new_X>0&&new_Y>0
% % % %             fai_new=atan(new_Y/new_X);
% % % %         end
% % % %         if new_X<0&&new_Y>0
% % % %             fai_new=pi+atan(new_Y/new_X);
% % % %         end
% % % %         if new_X<0&&new_Y<0
% % % %             fai_new=pi+atan(new_Y/new_X);
% % % %         end
% % % %         if new_X>0&&new_Y<0
% % % %             fai_new=2*pi-atan(-new_Y/new_X);
% % % %         end
% % % %         x_new=floor(fai_new*w/360*180./pi);
% % % %         if x_new<=0
% % % %             x_new=1;
% % % %         end
% % % %         y_new=floor(theta_new*h/180*180./pi);
% % % %         if y_new<=0
% % % %             y_new=1;
% % % %         end
% % % %         rImg(i,j,:)=f(y_new,x_new,:);
% % % %         x_new = (x_new - 256.0)/256.0;
% % % %         y_new = (y_new - 128.0)/128.0;
% % % %         LUT(kk,:)=[x_new,y_new];%已经完成了归一化，并且x和y的坐标没错乱，不需要在使用的时候交叉赋值了
% % % %         kk=kk+1;
    end
end
% rImg=uint8(rImg);
% subplot(211),imshow(f)
% subplot(212),imshow(rImg)
% imwrite(rImg,'P70Y90rimg.bmp')
str=strcat('G:\liujingguo\LUT\2LUT128\LUT_mid.mat');
save(str,'LUT')
kk=kk+1;