function chanLocs = LM_loadChanLocs(nbChan)

if nargin < 1
    nbChan = 64;
end

% assuming this is run from LMpackage_examples/someExample
chanLocs = load(fullfile('..',sprintf('chanLocs-%i.mat',nbChan)));
chanLocs = chanLocs.chanLocs;

end