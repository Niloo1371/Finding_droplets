sigma = 5;
cutoff = ceil(3*sigma);
h = fspecial('gaussian',2*cutoff+1,sigma);

h_1d = fspecial('gaussian',[1,2*cutoff+1],sigma); % 1D filter

dh = h .* (-cutoff:cutoff) / (-sigma^2);

h

h_1d

dh


