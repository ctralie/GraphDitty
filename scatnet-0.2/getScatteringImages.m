% Do the scattering transform on a bunch of images that are the same size

path_to_scatnet = fileparts(mfilename('fullpath'));

addpath(fullfile(path_to_scatnet, 'classification'));
addpath(fullfile(path_to_scatnet, 'convolution'));
addpath(fullfile(path_to_scatnet, 'core'));
addpath(genpath(fullfile(path_to_scatnet, 'demo')));
addpath(fullfile(path_to_scatnet, 'display'));
addpath(fullfile(path_to_scatnet, 'filters'));
addpath(fullfile(path_to_scatnet, 'filters/selesnick'));
addpath(fullfile(path_to_scatnet, 'utils'));
addpath(genpath(fullfile(path_to_scatnet, 'papers')));
addpath(fullfile(path_to_scatnet, 'scatutils'));
addpath(genpath(fullfile(path_to_scatnet, 'unittest')));
addpath(fullfile(path_to_scatnet, 'utils'));
addpath(fullfile(path_to_scatnet, 'reconstruction'));

clear path_to_scatnet;
load('x.mat');

[Wop,filters] = wavelet_factory_2d(size(x(:, :, 1)));
NImages = size(x, 3);
res = cell(1, NImages);
for ii = 1:NImages
    ii
    tic;
    [S,U] = scat(x(:, :, ii),Wop);
    res{ii} = image_scat(S, renorm, 0);
    toc;
end
save('res.mat', 'res');
