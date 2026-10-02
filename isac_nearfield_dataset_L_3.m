%% XL-MIMO Near-Field ISAC Channel Generation
% Generate near-field communication channels and sensing matrices for
% XL-MIMO integrated sensing and communication (ISAC) systems under
% different numbers of shared communication-sensing scatterers.
%
% Number of shared scatterers: 1, 2, 3

% Each ISAC overlap case is saved separately.

clear;
clc;

%% ======================== ISAC System Parameters =======================

N = 256;                       % Number of BS antennas
lambda = 0.01;                 % Carrier wavelength (m)
antenna_spacing = lambda / 2;  % Inter-element antenna spacing

Ln = 3;                        % Number of near-field communication paths

 %num_sta = 5000;               % Number of spatial channel realizations for training
num_sta = 200;                  % Number of spatial channel realizations for testing
num_ffading = 10;              % Small-scale fading realizations per geometry
num_train = num_sta * num_ffading;

% Number of shared scatterers between communication and sensing
num_overlap01 = 1;
num_overlap02 = 2;
num_overlap03 = 3;

% Gaussian sensing-parameter estimation errors
aoa_error_std = 1e-4;          % AoA estimation error standard deviation (rad)
range_error_std = 1e-2;        % Range estimation error standard deviation (m)

% Average communication path power
Cp_near = 1;

%% ======================== Data Initialization ==========================

% Near-field communication channel
Channel_mat = complex(zeros(num_train, N));

% Ideal communication-side near-field steering matrix
Channel_near_Radar_mat = complex(zeros(num_train, N, Ln));

% Sensing-side steering matrices with 1, 2, and 3 shared scatterers
Channel_near_Radar_overlap1_mat = complex(zeros(num_train, N, Ln));
Channel_near_Radar_overlap2_mat = complex(zeros(num_train, N, Ln));
Channel_near_Radar_overlap3_mat = complex(zeros(num_train, N, Ln));

%% ======================== ISAC Channel Generation ======================

for i = 1:num_sta

    %% ---------------- Communication-Side Parameters --------------------

    % Near-field communication AoA
    theta_near = pi * (rand(1, Ln) - 0.5);

    % Near-field communication distance, uniformly distributed in [10, 80] m
    r_near = 70 * rand(1, Ln) + 10;

    %% ------------- Sensing Parameters: 1 Shared Scatterer --------------

    theta_radar01 = zeros(1, Ln);
    r_radar01 = zeros(1, Ln);

    % The first sensing scatterer is shared with the communication channel
    theta_radar01(1:num_overlap01) = theta_near(1:num_overlap01);
    r_radar01(1:num_overlap01) = r_near(1:num_overlap01);

    % The remaining sensing scatterers are independently generated
    theta_radar01(num_overlap01+1:end) = ...
        pi * (rand(1, Ln - num_overlap01) - 0.5);

    r_radar01(num_overlap01+1:end) = ...
        70 * rand(1, Ln - num_overlap01) + 10;

    % Add sensing parameter estimation errors
    theta_radar01 = theta_radar01 + ...
        aoa_error_std * randn(1, Ln);

    r_radar01 = r_radar01 + ...
        range_error_std * randn(1, Ln);

    %% ------------- Sensing Parameters: 2 Shared Scatterers -------------

    theta_radar02 = zeros(1, Ln);
    r_radar02 = zeros(1, Ln);

    % The first two sensing scatterers are shared with the communication channel
    theta_radar02(1:num_overlap02) = theta_near(1:num_overlap02);
    r_radar02(1:num_overlap02) = r_near(1:num_overlap02);

    % The remaining sensing scatterers are independently generated
    theta_radar02(num_overlap02+1:end) = ...
        pi * (rand(1, Ln - num_overlap02) - 0.5);

    r_radar02(num_overlap02+1:end) = ...
        70 * rand(1, Ln - num_overlap02) + 10;

    % Add sensing parameter estimation errors
    theta_radar02 = theta_radar02 + ...
        aoa_error_std * randn(1, Ln);

    r_radar02 = r_radar02 + ...
        range_error_std * randn(1, Ln);

    %% ------------- Sensing Parameters: 3 Shared Scatterers -------------

    theta_radar03 = zeros(1, Ln);
    r_radar03 = zeros(1, Ln);

    % All three sensing scatterers are shared with the communication channel
    theta_radar03(1:num_overlap03) = theta_near(1:num_overlap03);
    r_radar03(1:num_overlap03) = r_near(1:num_overlap03);

    % When num_overlap03 = Ln, the following index ranges are empty
    theta_radar03(num_overlap03+1:end) = ...
        pi * (rand(1, Ln - num_overlap03) - 0.5);

    r_radar03(num_overlap03+1:end) = ...
        70 * rand(1, Ln - num_overlap03) + 10;

    % Add sensing parameter estimation errors
    theta_radar03 = theta_radar03 + ...
        aoa_error_std * randn(1, Ln);

    r_radar03 = r_radar03 + ...
        range_error_std * randn(1, Ln);

    %% ---------------- Small-Scale Fading Realizations ------------------

    for n = 1:num_ffading

        sample_idx = (i - 1) * num_ffading + n;

        %% Near-Field ISAC Steering Matrices

        delta = (2 * (1:N)' - N - 1) / 2;

        % Communication-side near-field steering matrix
        B_near = complex(zeros(N, Ln));

        % Sensing-side near-field steering matrices
        B_near_overlap1 = complex(zeros(N, Ln));
        B_near_overlap2 = complex(zeros(N, Ln));
        B_near_overlap3 = complex(zeros(N, Ln));

        for l = 1:Ln

            % Communication path distance from each antenna element
            r_dis = sqrt( ...
                r_near(l)^2 ...
                + delta.^2 * antenna_spacing^2 ...
                - 2 * r_near(l) * delta * antenna_spacing ...
                * sin(theta_near(l)));

            % Sensing path distance: 1 shared scatterer
            r_dis_overlap1 = sqrt( ...
                r_radar01(l)^2 ...
                + delta.^2 * antenna_spacing^2 ...
                - 2 * r_radar01(l) * delta * antenna_spacing ...
                * sin(theta_radar01(l)));

            % Sensing path distance: 2 shared scatterers
            r_dis_overlap2 = sqrt( ...
                r_radar02(l)^2 ...
                + delta.^2 * antenna_spacing^2 ...
                - 2 * r_radar02(l) * delta * antenna_spacing ...
                * sin(theta_radar02(l)));

            % Sensing path distance: 3 shared scatterers
            r_dis_overlap3 = sqrt( ...
                r_radar03(l)^2 ...
                + delta.^2 * antenna_spacing^2 ...
                - 2 * r_radar03(l) * delta * antenna_spacing ...
                * sin(theta_radar03(l)));

            % Communication-side steering vector
            B_near(:, l) = 1 / sqrt(N) * exp( ...
                -1j * 2 * pi / lambda * ...
                (r_dis - r_near(l)));

            % Sensing-side steering vector: 1 shared scatterer
            B_near_overlap1(:, l) = 1 / sqrt(N) * exp( ...
                -1j * 2 * pi / lambda * ...
                (r_dis_overlap1 - r_radar01(l)));

            % Sensing-side steering vector: 2 shared scatterers
            B_near_overlap2(:, l) = 1 / sqrt(N) * exp( ...
                -1j * 2 * pi / lambda * ...
                (r_dis_overlap2 - r_radar02(l)));

            % Sensing-side steering vector: 3 shared scatterers
            B_near_overlap3(:, l) = 1 / sqrt(N) * exp( ...
                -1j * 2 * pi / lambda * ...
                (r_dis_overlap3 - r_radar03(l)));
        end

        %% Near-Field Communication Channel Gain

        f_near = ...
            Cp_near / sqrt(2) * randn(Ln, 1) + ...
            1j * Cp_near / sqrt(2) * randn(Ln, 1);

        h_near = B_near * f_near;

        %% Save Generated ISAC Samples

        % Ideal communication-side steering matrix
        Channel_near_Radar_mat(sample_idx, :, :) = B_near;

        % Sensing-side steering matrices under different overlap settings
        Channel_near_Radar_overlap1_mat(sample_idx, :, :) = ...
            B_near_overlap1;

        Channel_near_Radar_overlap2_mat(sample_idx, :, :) = ...
            B_near_overlap2;

        Channel_near_Radar_overlap3_mat(sample_idx, :, :) = ...
            B_near_overlap3;

        % Normalized near-field communication channel
        Channel_mat(sample_idx, :) = ...
            (sqrt(N / Ln) * h_near).';
    end
end

%% ======================== Save ISAC Datasets Separately ================

output_dir = 'data';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% One shared communication-sensing scatterer
save( ...
    fullfile(output_dir, 'Pnear_overlap1_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_2000.mat'), ...
    'Channel_mat', ...
    'Channel_near_Radar_overlap1_mat');
% 
% Two shared communication-sensing scatterers
save( ...
    fullfile(output_dir, 'Pnear_overlap2_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_2000.mat'), ...
    'Channel_mat', ...
    'Channel_near_Radar_overlap2_mat');

%Three shared communication-sensing scatterers
save( ...
    fullfile(output_dir, 'Pnear_overlap3_AOAerr_1e-4_RangeErr_1e-2_Ln3_N256_2000.mat'), ...
    'Channel_mat', ...
    'Channel_near_Radar_overlap3_mat');

disp('ISAC dataset generation completed.');
disp('Three datasets were saved for 1, 2, and 3 shared scatterers.');
