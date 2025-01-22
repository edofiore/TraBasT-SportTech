% File name
name = 'file_name'; % e.g.: 'Nick_1'

% Load .mat file
data = load([name '.mat']);

% Access 'Skeletons' data
Skeletons = data.(name).Skeletons;

% Extract PositionData (3x24x12000)
positions = Skeletons.PositionData;

% Add segment labels (SegmentLabels is a 1x24 cell)
labels = Skeletons.SegmentLabels;

% Number of frames
% num_frames = size(positions, 3);

% Manual selection
% Frame range to export
start_frame = 2200; % First frame
end_frame = 2300; % Last frame
num_frames_to_export = end_frame - start_frame; % Total number of frames

% Prepare column header: x, y, z for each segment
label_names = string(labels); % Convert labels to strings
headers = strings(1, size(labels, 2) * 3);

for i = 1:length(label_names)
headers(3 * (i - 1) + 1:3 * i) = [label_names(i) + "_x", label_names(i) + "_y", label_names(i) + "_z"];
end

% Prepare data matrix (each frame is a row, with x, y, z for each segment)
data_matrix = zeros(num_frames_to_export, length(labels) * 3);

% Populate data matrix
% for frame = 1:num_frames
for frame_idx = 1:num_frames_to_export
frame = start_frame + frame_idx; % Calculate actual frame
for segment = 1:length(labels)
% Extract x, y, z coordinates for current frame and segment
x = positions(1, segment, frame);
y = positions(2, segment, frame);
z = positions(3, segment, frame);

% Add values ​​to matrix
data_matrix(frame_idx, 3 * (segment - 1) + 1) = x;
data_matrix(frame_idx, 3 * (segment - 1) + 2) = y;
data_matrix(frame_idx, 3 * (segment - 1) + 3) = z;
end
end

% Create table with data and headers
data_table = array2table(data_matrix, 'VariableNames', headers);

% Write CSV file
output_file = [name '-cut_good.csv'];
writetable(data_table, output_file);

disp(['Saved CSV file: ', output_file]);