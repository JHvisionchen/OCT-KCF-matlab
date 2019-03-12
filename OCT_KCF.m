function [positions, time] = OCT_KCF(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, s, num_init, threshold_g, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)

	%if the target is large, lower the resolution, we don't need that much detail
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end

	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    
    for frame = 1:numel(img_files),
		%load image
		img = imread([video_path img_files{frame}]);
        im=img;
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
		end

		tic()

		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			patch = get_subwindow(im, pos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			
			%calculate response of the classifier at all shifts
            switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                case 'polynomial',
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                case 'linear',
                    kzf = linear_correlation(zf, model_xf);
            end
            
			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            max_res(frame) = max(response(:)); %peak value of the responses on each frame
            
            if frame >= num_init
                temp_init = mean(max_res(3:num_init));
                miu_t = mean(max_res(2:frame-1)); %mean value of the gaussian model (denoted "miu^t" in the paper)
                sigma_t = sqrt(var(max_res(2:frame-1)));  %standard deviation value of the gaussian model (denoted "sigma^t" in the paper)
                hat_y_t = max_res(frame); %maximal response of current frame (denoted "hat_y^t" in the paper)

                if (max_res(frame) < 0.25 && (temp_init > 0.35))||...
                        (((hat_y_t - miu_t) / sigma_t) < -threshold_g)  %Eq.26 in the paper
                    
                    % coarse and fine process to precisely localize the target for sample selection in a local region
                    coarse_regions = coarse_sampler(pos, 0.8 * (sqrt(0.025 / max_res(frame) * target_sz(1)^2 + 0.25 * target_sz(2)^2)), 5, 16);
                    responses_coarse = cell(length(coarse_regions),1);
                    max_response_coarse = zeros(1,length(coarse_regions));
                    
                    %calculate response of all coarse regions
                    for index_coarse = 1:length(coarse_regions)     
                        pos = coarse_regions(index_coarse,:);
                        patch = get_subwindow(im, pos, window_sz);
                        zf = fft2(get_features(patch, features, cell_size, cos_window));
                        %calculate response of the classifier at all shifts
                        switch kernel.type
                            case 'gaussian',
                                kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                            case 'polynomial',
                                kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                            case 'linear',
                                kzf = linear_correlation(zf, model_xf);
                        end  
                        responses_coarse{index_coarse} = real(ifft2(model_alphaf .* kzf));
                        max_response_coarse(index_coarse)=max(max(responses_coarse{index_coarse}));
                    end
                    
                    %calculate the patch in which the target appears with maximum probability
                    index_patch = find(max_response_coarse == max(max_response_coarse), 1);
                    response = responses_coarse{index_patch};
                    pos = coarse_regions(index_patch,:);
                end
            end
            
            %execute the fine-tuning process
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
            
            if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
            end
            
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
		end

		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
		xf = fft2(get_features(patch, features, cell_size, cos_window));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian',
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training

		if frame == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
        else
            %subsequent frames, interpolate model
            eta = (kf + lambda) ./ (kf + lambda + 4 * lambda * s);  %denoted "eta" in the paper. See Eq.(24)
            distance = norm(eta - interp_factor,2);
            if distance > 3.1||distance < 2.3
                eta = interp_factor;
            end
            model_alphaf = (1 - eta) .* model_alphaf + eta .* alphaf;  %Eq. (25)
            model_xf = (1 - interp_factor) .* model_xf + interp_factor .* xf;
		end

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();
		%visualization
        if show_visualization,
            box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            if frame == 1,  %first frame, create GUI
                figure('Number','off', 'Name',['Tracker - ' video_path]);
                im_handle = imshow(uint8(img), 'Border','tight', 'InitialMag', 100 + 100 * (length(img) < 500));
                rect_handle = rectangle('Position',box, 'EdgeColor','g');
                text_handle = text(10, 10, int2str(frame));
                set(text_handle, 'color', [0 1 1]);
            else
                try  %subsequent frames, update GUI
                    set(im_handle, 'CData', img)
                    set(rect_handle, 'Position', box)
                    set(text_handle, 'string', int2str(frame));
                catch
                    return
                end
            end
            drawnow
        end
    end
    
    if resize_image,
        positions = positions * 2;
    end
end

