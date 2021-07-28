clc;
clear all;
close all;

%% Capture Single Image using Basler CMOS Camera
% imaqreset;               %% Reset Camera & Its Settings

% imaqtool               %% Open Image Acquisition Toolbox
vid = videoinput('gentl', 1, 'RGB8');
src = getselectedsource(vid);

vid.FramesPerTrigger = 1;                       %% Set Frames per trigger
src.AcquisitionFrameRateEnable = 'True';        %% Acquisition Frame Rate Enable- True/False
src.BalanceRatioSelector = 'Red';               %% Balance Ratio Selector- Red/Blue/Green
src.BalanceRatio = 1;                       %% Balance Ratio varrying from- 0 to 15.98
src.BalanceRatioSelector = 'Green';               %% Balance Ratio Selector- Red/Blue/Green
src.BalanceRatio = 0;                       %% Balance Ratio varrying from- 0 to 15.98
src.BalanceRatioSelector = 'Blue';               %% Balance Ratio Selector- Red/Blue/Green
src.BalanceRatio = 0;                       %% Balance Ratio varrying from- 0 to 15.98
src.ExposureAuto = 'Off';                       %% ExposureAuto- Off /Once /Continous
src.ExposureTime =200;                          %% Set Exposure time- 59 to 50000
src.LightSourcePreset = 'Off';        %% Light Source Preset- Daylight5000K /Off /Tungsten2800K /Daylight6500K

src.TriggerActivation = 'RisingEdge';           %% Trigger Activation- RisingEdge /FallingEdge
src.TriggerDelay = 6.8;                           %%Trigger Delay
src.TriggerMode = 'On';                         %% Set Trigger mode- On/Off
src.TriggerSelector = 'FrameStart';             %% Trigger Selector- FrameStart /FrameBurstStart
src.TriggerSource = 'Line3';                    %% Trigger Source- Line1 /Software /Line3 /Line4 /SoftwareSignal1 /SoftwareSignal2 /SoftwareSignal3
triggerconfig(vid, 'hardware', 'DeviceSpecific', 'DeviceSpecific');
src.AcquisitionStatusSelector = 'FrameTriggerWait';
% src.EnableAcquisitionFrameRate
src.AcquisitionFrameRate = 200;                 %% Acquisition Frame Rate from 1 to 222

n = 1280  ;  %%dimesions of screen or SLM
m = 1024   ; %%dimesions of screen or SLM

pixels = 32;
nn = n/pixels;
mm = m/pixels;
% preview(vid);                                   %% Start Video Preview 
valueset = [0];
desiredsize = [nn,mm,1];
popp = valueset(randi(numel(valueset), desiredsize));

figure('Position',[1920+1920+1 57 1280 1024],'MenuBar','none','ToolBar','none', 'resize', 'off');
set(gca,'InnerPosition',[0 0 1 1], 'Visible', 'off');
II = image(popp(:,:,1));
axis off
colormap gray
pause(0.25);
start(vid);
% 
imwrite(getdata(vid), 'C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\initial.tiff');               
Z_initial =imread('C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\initial.tiff'); 

% Z_initial = getdata(vid);

initial = insertShape(Z_initial,'circle',[640+100 512-100 15],'LineWidth',2);
% imshow(RGB1)
% figure, imagesc(CapturedImage,'CDataMapping','scaled');


x_rate = 0.5;
N = 8;
%lamb = 1250;
gen = 10;
fprintf('i1')
disp(target_intensity(Z_initial,n,m))
valueset = [0,255];
desiredsize = [nn,mm,N];
pop = valueset(randi(numel(valueset), desiredsize));

max_fitness = [target_intensity(Z_initial,n,m)];
steps = [0];
tic
for s = 1:gen
    disp(s);
    steps = [steps s];
    fitness =[];
    si = size(pop);
%     pop1 = pop(:,:,1);
    for i = 1:si(3) 
        %% SLM Initialize and display
        figure('Position',[1920+1920+1 57 1280 1024],'MenuBar','none','ToolBar','none', 'resize', 'off');
        set(gca,'InnerPosition',[0 0 1 1], 'Visible', 'off');
        II = image(pop(:,:,i),'CDataMapping','scaled');
        axis off
        colormap gray
        pause(0.5);
% % % % assert(0)
        st = ['p',int2str(i)];
        path = ['C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\',st,'.tiff'];


        start(vid);
        imwrite(getdata(vid), path); 
        close;   %%for closing opened figures
        
        Z =imread(path);
%         Z = getdata(vid);
        intensity = target_intensity(Z,n,m);
        fitness = [fitness intensity];
%         hold off;
    end
    max_fitness = [max_fitness max(fitness)];
    if max(fitness) < max_fitness(length(max_fitness))
       pop(:,:,1) = pop1; 
    end    
    [C,IX] = sort(fitness,'descend');
    pop = pop(:,:,IX);
    pop = pop(:,:,1:floor(x_rate*N));
    st1 = ['p',int2str(IX(1))];
    cd Generation_Images;
    movefile([st1,'.tiff'],'Max_fitness');
    cd ..;
    
    
    %Mating
    if s ~= gen
        crossover = floor(mm/2);
        e = crossover;
        for j = 1:2:N
            [off1,off2]= mate(pop(:,:,j),pop(:,:,j+1),e,mm);
            pop = cat(3,pop,off1);
            pop = cat(3,pop,off2);
        
        end
        if max(fitness)<=max_fitness(length(max_fitness))
            mu = 0.07;
        else
            mu = 0.001;
        end
        mutations = floor(mu*N*nn*mm);
        mc = rand_int(mm,mutations);
        mr = rand_int(nn,mutations);
        mh = rand_int(N,mutations);
        for k = 1:length(mr)
            if pop(mr(k),mc(k),mh(k)) == 0
                pop(mr(k),mc(k),mh(k)) = 255;
            else
                pop(mr(k),mc(k),mh(k)) = 0;
            end    
       end    
        
            
    end
end

toc
st2 = ['p',int2str(IX(1))];
path2 = ['C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\Max_fitness\',st2,'.tiff'];

%path2 = ['C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\p4.tiff'];
Z = imread(path2);
fprintf('i2')
disp(target_intensity(Z,n,m))
%figure, imshow(Z);
RGB = insertShape(Z,'circle',[640+100 512-100 15],'LineWidth',2);
imwrite(RGB, 'C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\final.tiff');
fig = plot(steps,max_fitness);
saveas(fig,'C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\plot.png');

  
% FinalImage = imread('C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\final.tiff');
% figure(1); A = imagesc(128*FinalImage);
% hold; 
% figure(2); B = imagesc(128*RGB1);

[X2,map1]=imread('C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\final.tiff');
[X1,map2]=imread('C:\Users\Medicalphyscs\Desktop\GA code\Generation_Images\initial.tiff');
close;
figure(1); image(initial); 
figure(2); image(X2);
figure(3); plot(steps,max_fitness);
% subplot(1,2,1), imshow(X1,map1)
% subplot(1,2,2), imshow(X2,map2)

