function table = CombineLabelTables(originalPath)
%
% table = CombineLabelTables(originalPath)
%
% DESCRIPTION: Combine table containing the location of aorta bifurcation
%              from datasets into a new table
%
% INPUT:
%   originalPath - path of the tables
%
% OUTPUT:
%   table - Combined table containing the location information from all datasets
%
% Author: Toan Ly
% Date: 2/15/2024

table = [];
folders = {'Tr', 'Ts', 'Va'};

for idx = 1:length(folders)
    load(char(fullfile(originalPath, folders(idx), 'tblPoints.mat')));
    table = [table; tbl];
end

disp('Done merging xyz location tables!');

end
