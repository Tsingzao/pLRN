function TPVid = temporal_pooling(Vid)

    num = length(Vid);
    TPVid = cell(num,1);
    for i=1:num
        TPVid{i}=mean(Vid{i},3);
    end

end
