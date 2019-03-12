function kf = polynomial_correlation(xf, yf, a, b)
%POLYNOMIAL_CORRELATION
	
	%cross-correlation term in Fourier domain
	xyf = xf .* conj(yf);
	xy = sum(real(ifft2(xyf)), 3);  %to spatial domain
	
	%calculate polynomial response for all positions, then go back to the
	%Fourier domain
	kf = fft2((xy / numel(xf) + a) .^ b);

end

