for k = 1:10
    bag = rosbag(strcat(num2str(k),'.bag'));
    bSel = select(bag,'Topic','/sirab/azure/body_tracking_data');
    msgStructs = readMessages(bSel,'DataFormat','struct');
%    nofjoints = 32;
    
    length = size(msgStructs,1);
    
    A = zeros(32,7);
    mkdir(num2str(k))
    for i = 1:length
    a = size(msgStructs{i,1}.Markers);
    nofjoints = a(2);
        for j = 1:nofjoints
            if ~isempty(msgStructs{i, 1}.Markers)
                A(j,:) = [msgStructs{i, 1}.Markers(j).Pose.Position.X... 
                    msgStructs{i, 1}.Markers(j).Pose.Position.Y... 
                    msgStructs{i, 1}.Markers(j).Pose.Position.Z...
                    msgStructs{i, 1}.Markers(j).Pose.Orientation.X...
                    msgStructs{i, 1}.Markers(j).Pose.Orientation.Y...
                    msgStructs{i, 1}.Markers(j).Pose.Orientation.Z...
                    msgStructs{i, 1}.Markers(j).Pose.Orientation.W];
            else
                continue;
            end
        end
        writematrix(A,strcat(num2str(k),"/",num2str(i),'.csv'))
    end
end