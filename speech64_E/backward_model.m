% Basic example demonstrating how to use the LM package to fit a linear
% backward model reconstructing the stimulus based on EEG data.
%
allSID = {'YH00','YH01','YH02','YH03','YH04','YH06','YH07','YH08','YH09','YH10','YH11','YH14','YH15','YH16','YH17','YH18','YH19','YH20'};

parts = 1:4;
condition = 'clean';
Fs = 100;

nChan = 64;
procEEG = 'BP-1-12-INTP-AVR';

typeEnv = 'rectified';
procEnv = 'LP-12';


% Time region in which to derive the decoder. Time lag is understood as lag
% of predictor (here EEG) with respect to predicted data (here stimulus).
% Hence, here positive time lags correspond to the causal part of the
% decoder (response after stimulus).
minLagT = -100e-3;
maxLagT = 500e-3;

% estimate performance (CC & MSE) on windows of this duration
% (negative = use all available data)
tWinPerf = [5,10,25,-1]; % in seconds

% stimOpt and EEGopt are arbitray multi-dimensional variables (e.g.
% matrices, cell, structures) containing the required information to load
% stimulus and EEG data respectively. Each element of stimOpt and EEGopt
% will be passed to user-defined loading function (see below) that will
% load the data accordingly. Hence arbitray parameters can be passed to the
% loading functions.
%
% In this example, we will simply store the path of the files to load in
% cell arrays.
nParts = numel(parts);
nSub = numel(allSID);
stimOpt = cell(nParts,1);
% EEGopt have to be of size: [size(stimOpt),nSub], i.e. each stimulus file
% corresponds to 1 EEG recording per subject.
EEGopt = cell(nParts,nSub);

% load channel location
chanLocs = LM_loadChanLocs();
% define a channel order to be used for all data
chanOrder = {chanLocs(:).labels};

for iPart = 1:nParts
    envFileName = sprintf('env_Fs-%i-%s-%s_%s_%i.mat',Fs,procEnv,typeEnv,condition,iPart);
    stimOpt{iPart} = fullfile(pwd(),'data','envelopes',envFileName);
    
    for iSub = 1:nSub
        EEGFolder = fullfile(pwd(),'data','EEG',allSID{iSub});
        EEGFileName = sprintf('%s-Fs-%i-%s_%s_%i.set',procEEG,Fs,allSID{iSub},condition,iPart);
        
        EEGopt{iPart,iSub} = {EEGFolder,EEGFileName,chanOrder};
    end
end

% options passed to the call to get the appropriate matrices to fit the
% linear model
opt = struct();
opt.nStimPerFile = 1;
% These are loading function taking one element of stimOpt and EEGopt
% respectively as input, and loading stimulus / EEG data.
opt.getStimulus = @LM_loadFeature_speech64_E;
% This function should return as 1st output a [nPnts x nChan] data matrix,
% and as 2nd outut a vector of indices (size nStimPerFile x 1) indicating
% where each stimulus begins in the data. These indices should be sorted in
% the same order as the stimuli returned by opt.getStimulus.
opt.getResponse = @LM_loadEEG_speech64_E;

% nb of features describing each stimulus
opt.nFeatures = 1;
% nb of channels in the EEG data
opt.nChan = nChan;

% converting lags for time to indices
opt.minLag = floor(minLagT * Fs);
opt.maxLag = ceil(maxLagT * Fs);

opt.sumSub = false;
opt.sumStim = false; % does not matter here
opt.sumFrom = 0; % return cross matrices for each part 

% false: the predictor data (here EEG data) will be zeros padded at its
% edges. true: no padding.
opt.unpad.do = false;
% removing means = fitting models without offsets
opt.removeMean = true;

% convert to samples
opt.nPntsPerf = ceil(tWinPerf*Fs)+1;


nLags = opt.maxLag - opt.minLag + 1;

% options to fit the model
trainOpt = struct();
trainOpt.method.name = 'ridge-eig-XtX'; % use ridge regression
% regularisation coefficients for which we'll fit the model
trainOpt.method.lambda = 10.^(-6:0.5:6);
trainOpt.method.normaliseLambda = true;
trainOpt.accumulate = true; % the input is XtX & Xty, and not X & y

nLambda = numel(trainOpt.method.lambda);
nPerfSize = numel(tWinPerf);

% We will fit subject specific models using a leave-one-part-out
% cross-validation procedure. For each subject, a model will be fitted over
% all the data bar one part. The excluded part subject will be used as
% testing data.

% Testing value sets for each window duration in tWinPerf, each data part
% and each subject (number of values in each set will depend on the length
% of each data part, and duration of each testing window).
CC = cell(nPerfSize,nParts,nSub);
MSE = cell(nPerfSize,nParts,nSub);

coeffs = nan(nLags,nChan,nLambda,nSub);

for iSub = 1:nSub

    % forming XtX and Xty for all data parts for iSub
    [XtX,Xty] = LM_crossMatrices(stimOpt,EEGopt(:,iSub),opt,'backward');
    
    % leave-one-part-out cross-validation for iSub
    for iTestPart = 1:nParts
        
        % all parts, except iTestpart
        iTrainParts = [1:(iTestPart-1),(iTestPart+1):nParts];
        
        % model fitted using only training parts for iSub
        XtX_train = sum(XtX(:,:,iTrainParts),3);
        Xty_train = sum(Xty(:,:,iTrainParts),3);
        
        model_train = LM_fitLinearModel(XtX_train,Xty_train,trainOpt);
        model_train = model_train.coeffs;
        
        % testing on the remaining part
        stim_test = stimOpt(iTestPart);
        EEG_test = EEGopt(iTestPart,iSub);
        
        [ CC(:,iTestPart,iSub),...
            MSE(:,iTestPart,iSub)] = LM_testModel(model_train,stim_test,EEG_test,opt,'backward');
    end
    
    % finally train & store decoder on all data
    XtX = sum(XtX,3);
    Xty = sum(Xty,3);
    model = LM_fitLinearModel(XtX,Xty,trainOpt);
    coeffs(:,:,:,iSub) = reshape(model.coeffs,[nLags,nChan,nLambda]);
    
end

% Return a time vector associated with the coefficients, and make sure the
% causal part of the coefficients are at positive latencies.
[tms,coeffs] = LM_getTime(opt,Fs,'backward',coeffs,1);
tms = 1e3 * tms; % in milliseconds

%%
% looking at the data using 10s slices
dur0 = 10;
iDur0 = find(tWinPerf == dur0,1);

% pooling all the testing results for windows of duration dur0 
CC0 = vertcat(CC{iDur0,:});
nWin = size(CC0,1) / nSub;
CC0 = reshape(CC0,[nWin,nSub,nLambda]);
mCC = squeeze(mean(CC0,1))';
sCC = squeeze(std(CC0,[],1))';

% regularisation curve for each subject
figure;
ax = axes();
plot(trainOpt.method.lambda ,mCC);
ax.XAxis.Scale = 'log';
ax.XAxis.Label.String = '\lambda_n';
ax.YAxis.Label.String = 'Correlation coefficient';
ax.Title.String = 'Regularisation curve for each subject';

% regularisation coefficient
lambda0 = 10^-0.5;
[~,iLambda0] = min(abs(trainOpt.method.lambda-lambda0));

% CC for each subject @ lambda0, sorted
[mCC_,iSort] = sort(mCC(iLambda0,:),'ascend');
sCC_ = sCC(iLambda0,iSort);

% corresponding decoder, averaged across subjects
meanDecoder = mean(coeffs(:,:,iLambda0,:),3);

figure;
ax = axes();
errorbar(1:nSub,mCC_,sCC_,'ko');
ax.XAxis.Label.String = 'Subject #';
ax.YAxis.Label.String = 'Correlation coefficient';
ax.Title.String = sprintf('Sorted correlation coefficients for \\lambda_n = 10^{%.1f}',log10(lambda0));
ax.XAxis.Limits = [0,nSub+1];

% Plotting decoder weights at a given time
t0 = 130; % ms 
[~,it0] = min(abs(tms-t0));

figure;
topoplot(meanBestDecoder(it0,:),chanLocs);
title(sprintf('Decoder topography at t = %i ms',t0));
%
% The decoder obtained for each subject at e.g. the best CC + 1 std could
% then be used on some held out data.
%
%