function feature = loadFeature(c)
%
% LM.example.speech64E.loadFeature
% Part of the Linear Model (LM) package.
% Author: Octave Etard
%
% Input c is a cell containing the path towards one feature file
% describing the stimulus
%
% This function simply loads the file, and returns the feature.
%
d = load(c{1});
feature = d.attended;

end
%
%