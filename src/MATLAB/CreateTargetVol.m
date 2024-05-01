function volMask = CreateTargetVol(vol, location)
%
% volMask = CreateTargetVol(vol, location)
%
% DESCRIPTION: Create a new target space that has a sphere
%              at the aorta bifurcation
%
% INPUT:
%   vol - Original 3d image
%   location - xyz location of the aorta bifurcation
% 
% OUTPUT:
%   volMask - Binary mask representing the sphere around the aorta bifurcation
%
% Toan Ly
% 2/15/2024

% Create a blank space
volMask = zeros(size(vol));

[x, y, z] = meshgrid(1:size(vol, 1), 1:size(vol, 2), 1:size(vol, 3));


distances = sqrt((x - location(1)).^2 + (y - location(2)).^2 + (z - location(3)).^2);

sigma = 7;

% Create Gaussian
gaussian = exp(-0.5 * distances.^2 / sigma^2);
gaussian(gaussian < 0.1) = 0;

volMask = gaussian;

% Create a sphere
% volMask = (x-location(1)).^2 + (y-location(2)).^2 + (z-location(3)).^2 < radius^2;

end