function GenerateWarpedVol()
%
% GenerateWarpedVol()
%
% DESCRIPTION:
%
% Author: Toan Ly
% Date: 4/16/2024

% Read the Excel file
data = readtable('Project\AortaBifurcationProject\DataGroup.xlsx');

% Define paths to original and new folders
originalPath = 'MiniProjects\AortaBifurcationAndSacrum\Data';
newPath = 'DataWarped';
if ~exist(newPath, 'dir')
    mkdir(newPath);
end

targetSz = [160, 160, 160];
sigma = [4, 7, 10, 10];
warpMag = [4, 7, 8, 9];

% Get labels table
tbl = CombineLabelTables(originalPath);

progress_bar = waitbar(0, 'Processing Data...', 'Name', 'Generating Warped Volumes');
% Loop through excel file
for i = 1:height(data)
    tic
    dataset = data.DataSet(i);
    filename = data.Name(i);
    group = data.Group(i);
    
    % Save patches to destination folder
    filePath = fullfile(originalPath, dataset, filename);
    patchPath = fullfile(newPath, group); % output file
    
    if strcmp(group, 'Train')
        load(char(filePath)); % load vol
        coronalVol = permute(vol, [3, 2, 1]); % Change to coronal view
        
        % Save labels to destination folder
        locationIdx = find(strcmp(tbl.dataset, dataset) & strcmp(tbl.pat, filename));
        xyzLocation = tbl.xyzAortaBifur(locationIdx, :);
        xyzLocation([2 3]) = xyzLocation([3 2]);
        x = xyzLocation(1);
        y = xyzLocation(2);
        z = xyzLocation(3);
        
        progress_percentage = i / height(data);
        waitbar(progress_percentage, progress_bar, sprintf('Processing Data... %d/%d',i, height(data)));
        for n = 1:4
            [warpedVol, df] = RandWarp3d(coronalVol, sigma(n), warpMag(n), 'featherDist', warpMag(n));
            warpedVol = int16(warpedVol);
            
            dist = (df.Xp - x).^2 + (df.Yp - y).^2 + (df.Zp - z).^2;
            [minDist, idx] = min(dist(:));
            [x, y, z] = ind2sub(size(coronalVol), idx);
            xyzLocation = [x, y, z];
            
            [~, newFilename, ext] = fileparts(char(filename));
            newFilename = sprintf('%s_%s_warped_%d%s', char(dataset), newFilename, n, ext);
            
            save(char(fullfile(newPath, group, newFilename)), 'warpedVol', 'xyzLocation', '-v7');
        end
    toc
    end
end
close(progress_bar);

disp('Done Splitting Data');

end


