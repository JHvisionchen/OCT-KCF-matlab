function kf = linear_correlation(xf, yf)
%LINEAR_CORRELATION
	
	%cross-correlation term in Fourier domain
	kf = sum(xf .* conj(yf), 3) / numel(xf);

end

