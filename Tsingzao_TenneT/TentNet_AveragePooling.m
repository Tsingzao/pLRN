function feature = TentNet_AveragePooling(OutVid_i)

feature = zeros(2048,1);
for p = 1:64
    feat = zeros(4,4,2);
    temp = OutVid_i{p};
    for i = 1 : 8 : 32
        for j = 1 : 8 : 32
            for k = 1 : 8 : 16
                feat(mod(i,8)+1, mod(j,8)+1, mod(k,8)+1)=max(max(max(temp(i:i+7,j:j+7,k:k+7))));
            end
        end
    end
    feature((p-1)*32+1:p*32)=feat(:);
end

end

