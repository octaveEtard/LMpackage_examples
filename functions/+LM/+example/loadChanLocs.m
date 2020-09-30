function chanLocs = loadChanLocs(nbChan)
%
% LM.example.loadChanLocs
% Part of the Linear Model (LM) package.
% Author: Octave Etard
%
if nargin < 1
    nbChan = 64;
end

% assuming this is run from LMpackage_examples/someExample
chanLocs = load(fullfile('..',sprintf('chanLocs-%i.mat',nbChan)));
chanLocs = chanLocs.chanLocs;

end