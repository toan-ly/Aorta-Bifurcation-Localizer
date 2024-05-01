function patches = ExtractPatches(im, patchSz, overlap, isLabel)
%
% patches = ExtractPatches(im, patchSz, overlap, isLabel) 
%
% DESCRIPTION: Extract equal-size 3d or 2d patches from a 3d image
%              with overlap
% INPUT:
%   im: Input 3d image
%   patchSz - Patch size [patchSizeX, patchSizeY, patchSizeZ]
%   overlap - Overlap size [overlapX, overlapY, overlapZ]
%   isLabel - boolean indicating whether patches are labels or not
%
% OUTPUT:
%   patches - Cell array containing the extracted patches
%
% Author: Toan Ly
% Date: 2/14/2024

step = patchSz - overlap;

% Calculate the number of patches along each dimension
numPatchesX = ceil((size(im, 1) - patchSz(1)) / step(1)) + 1;
numPatchesY = ceil((size(im, 2) - patchSz(2)) / step(2)) + 1;
numPatchesZ = ceil((size(im, 3) - patchSz(3)) / step(3)) + 1;

% Initialize cell array to store patches
patches = cell(numPatchesX, numPatchesY, numPatchesZ);
% Extract patches
for x = 1:numPatchesX
    for y = 1:numPatchesY
        for z = 1:numPatchesZ
            startX = (x - 1) * step(1) + 1;
            startY = (y - 1) * step(2) + 1;
            startZ = (z - 1) * step(3) + 1;
            
            endX = min(startX + patchSz(1) - 1, size(im, 1));
            endY = min(startY + patchSz(2) - 1, size(im, 2));
            endZ = min(startZ + patchSz(3) - 1, size(im, 3));
            
            % Adjust the start and end indices to ensure patches are of equal size
            startX = endX - patchSz(1) + 1;
            startY = endY - patchSz(2) + 1;
            startZ = endZ - patchSz(3) + 1;
            
            if isLabel
                % Extract the middle slice for labels
                midSlice = floor(patchSz(3) / 2) + startZ;
                patches{x, y, z} = im(startX:endX, startY:endY, midSlice);
                continue
            end
            
            % Extract patch
            patches{x, y, z} = im(startX:endX, startY:endY, startZ:endZ); 
        end
    end
end

end