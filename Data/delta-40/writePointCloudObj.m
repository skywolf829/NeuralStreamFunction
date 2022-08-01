function writePointCloudObj(vdata,filename)
fileID = fopen(filename,'w');

%vertices
formatSpec = 'v %6.8f %6.8f %6.8f\n';
fprintf(fileID,formatSpec,...
    vdata');
fprintf(fileID,'\n');

fclose(fileID);
end

