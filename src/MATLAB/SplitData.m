function SplitData()
%
% SplitData()
%
% DESCRIPTION: Combine datasets from aorta bifurcation and sacrum mini project 1,
%              extract patches and labels, randomly split into 3 new datasets with
%              400 training, 50 test, and 50 validation, and store into new folder
%
% Author: Toan Ly
% Date: 2/14/2024
% Riverain Tech

% Read the Excel file
data = readtable('Project\AortaBifurcationProject\DataGroup.xlsx');

% Define paths to original and new folders
originalPath = 'MiniProjects\AortaBifurcationAndSacrum\Data';
newPath = 'C:\Toan_Project\Data';

targetSz = [128, 128, 5];
overlap = [64, 64, 2];

% Get labels table
tbl = CombineLabelTables(originalPath);

% Loop through excel file
for i = 1:height(data)
    dataset = data.DataSet(i);
    filename = data.Name(i);
    group = data.Group(i);
    
    % Save patches to destination folder
    filePath = fullfile(originalPath, dataset, filename);
    patchPath = fullfile(newPath, group); % output file
    
    load(char(filePath)); % load vol
    patches = ExtractPatches(vol, targetSz, overlap, false);
    
    
    % Save labels to destination folder
    locationIdx = find(strcmp(tbl.dataset, dataset) & strcmp(tbl.pat, filename));
    xyzLocation = tbl.xyzAortaBifur(locationIdx, :);
    volMask = CreateTargetVol(vol, xyzLocation);
    labels = ExtractPatches(volMask, targetSz, overlap, true);
    
    filename = char(filename);
    filename = filename(1:end-4);
    for x = 1:size(patches, 1)
        for y = 1:size(patches, 2)
            for z = 1:size(patches, 3)
                imStack = patches{x, y, z};
                imLabel = single(labels{x, y, z});
                
                % Check if label patch has positive values
                if any(imLabel(:) > 0)
                    labelFilename = sprintf('%s_%s_%d_%d_%d_positive.mat', char(dataset), filename, x, y, z);
                else
                    labelFilename = sprintf('%s_%s_%d_%d_%d.mat', char(dataset), filename, x, y, z);
                end
                save(char(fullfile(patchPath, labelFilename)), 'imStack', 'imLabel', '-v7');   
            end
        end
    end
end

disp('Done Splitting Data');

end


