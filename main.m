%%       Please Note: This is the raw unformatted copy. Thank you.       %%
%#########################################################################%
%Please include the reference below in your publication, if you find any  %
%part of this work useful                                                 %
%                            Reference:                                   %
%O.G. Olaleye, M.A. Iqbal, A. Aly, D. Perkins, M. Bayoumi, "An Energy-    %
%Detection-based Cooperative Spectrum Sensing Scheme for Minimizing the   % 
%Effects of NPEE and RSPF," MSWiM ’16, November 13-17, 2016, Malta, Malta %
%								AND										  %
%O.G. Olaleye, A. Aly, A. Ali, M.A. Iqbal, D. Perkins, M. Bayoumi,        %
%"Towards Performance and Scalability Benchmarking of Spectrum Awareness  %
%in Heterogeneous Cognitive Radio Networks," IEEE Transactions on         %
%Cognitive Communications and Networking, vol. 1, no. 1, pp. 1-14, 2017.  %
%																		  %
%#########################################################################%
%#########################################################################%
% Author:  Oladiran G. Olaleye                                            %
% Project: Towards Detection and Elimination of Uncertainties in          %
%          Big Spectrum Data Synthesis                                    %
% School:  The Center for Advanced Computer Studies                       %
%          University of Louisiana at Lafayette                           %
% Email:   ogo8842@louisiana.edu                                          %
% Website: http://web.cacs.louisiana.edu/labs/wisper/index.html           %
%                                                                         %
%   						   02/17/2017                                 %
%						      Copyright(c)								  %
%#########################################################################%
%% This code plots:
% (a) ...
% (b) ...
% (c) ...
%
%% List of Assumptions: 
% 1) Additive White Guassian Noise (AWGN);
% 2) 2D grid (x-y coord);
% 3) Network users' antennas are omini-directional
% 4) RSS prediction accuracy = 80-100%
% 5) Noise power estimation accuracy  = 80-100%
% 6) if (noise est << (rss mea - rss pred)), r_noise = 0 [[[[?????]]]]
% 7) Independent multipath and shadowing fading
% 8) P_N  = zero-mean Guassian random variable with variance sigma_i
% 9) No antennna gain and no RF loss 
% 10) Assume Pn_SU_avg is constant for all SUs and for the entire set of experiments with varying P_FA
% 11) (???) = [For the simulation, if (Pn_estimated < Pn_meas), we assume Pn_act...]
% 12) 
% 13) 
% 14) 
%%
%
%----------------------------- BEGIN CODE --------------------------------%
%%                   Simulation SetUp and Parameters:                    %%

clear all; close all; clc
set(0,'defaultAxesFontName', 'Times New Roman');

% Network Topology
Netw_Area = 1e6; % in m^2 (1000x1000 square meters)
Netw_width = 1000; % (meters)
Left_boundary = -round(Netw_width/2);
Right_boundary = round(Netw_width/2);

% Primary User
num_PrimaryUsers = 1;                          % Number of primary users
%R_PUcoverage = ;                              % Radius of each PU coverage, in meter     
r_KeepOffZone = 50;                            % Radius of the keep off zone, in meter
Pt_PU = 10.*ones(num_PrimaryUsers,1);          % Transmit Power in dBm (=10mW)
P_TXAntennaGain = 0.0; % in (db)
height_antenna = 500;                          % in meters

% Secondary User
num_SecondaryUsers = 50;                       % Number of secondary users
P_RXAntennaGain = 0.0 .* ones(num_SecondaryUsers,1); % in (db)

% Signal
S_signal = 3e8;                                % Speed of light (meters/sec)
frequency = 615*10^6;                          % Frequency of signal (Hz); [DTV operating in the UHF band]
Lambda = S_signal / frequency;                 % Signal wavelength
n_PLE = 2.3;                                   % Pathloss ExponentPathloss Exponent = {1.8... 5.0} [Seidel and Rappaport]
sensing_time = 0.2;                            % milliSeconds (unit) 
%numSamples = 2 * (sensing_time/10^3) * (frequency);   % Number of samples at each SU
numSamples = 10; % Number of samples at each SU
Pf = repmat(0.01:0.01:1.0,num_SecondaryUsers,1); % Fixed Probability of false alarm

% Noise
%Pn_SU_actual = repmat(Pn_SU_LowerLimit + (Pn_SU_UpperLimit - Pn_SU_LowerLimit).*rand(num_SecondaryUsers,1),1,length(Pf(1,:))); % Actual noise power (dBm)
%Pn_SU_actual = repmat(95.*randn(num_SecondaryUsers,1),1,length(Pf(1,:))); % Actual noise power (dBm)
Pn_SU_actual_vector = -68.0;
Pn_SU_actual = Pn_SU_actual_vector.*ones(num_SecondaryUsers,length(Pf(1,:))); % Actual noise power (dBm)

% Plots
graph_color = ('brgkgrbkgrbgrbkbrgcmybrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymcbrgkymc'); % Plot colors
graph_ID = ('sovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*xsovd+h^p*x'); % Plot identifiers
graphMarker_size = [4 6 7 8 9 10]; % Plot sizes
choosePlot = 50;

% Actual PU Location Grid (Network Topology)
% PU_interval = floor(Netw_width/(round(sqrt(num_PrimaryUsers))-1));
% length_boundary_PU = length(Left_boundary:PU_interval:Right_boundary);
% CoordXY_PU_actual = zeros(num_PrimaryUsers,2);  % PU-centric model
% for xx = Left_boundary:PU_interval:Right_boundary
%     for yy = Left_boundary:PU_interval:Right_boundary
%         Next_Coord = round(  ((yy/PU_interval)+(1-Left_boundary/PU_interval))  +  (((xx/PU_interval)+(1-Left_boundary/PU_interval))-1)*length_boundary_PU  );
%         CoordXY_PU_actual(Next_Coord,1)=yy;
%         CoordXY_PU_actual(Next_Coord,2)=xx;
%     end  
% end
CoordXY_PU_actual(1,:) = [0 0]; % Single PU network

% Actual SU Location radial (Network Topology)
SU_interval = (2*pi:pi/8:(65/8)*pi)';
CoordXY_SU_actual = zeros(num_SecondaryUsers,2);  % SU-grid model
CoordXY_SU_actual(:,1) = 20.*SU_interval.*cos(SU_interval);
CoordXY_SU_actual(:,2) = 20.*SU_interval.*sin(SU_interval);
% % Actual SU Location radial (Network Topology)
% SU_interval = floor(Netw_width/(sqrt(num_SecondaryUsers)-1));
% num_SU_perRing = ceil((num_SecondaryUsers)/(Right_boundary/SU_interval));
% CoordXY_SU_actual = zeros(num_SecondaryUsers,2);  % SU-grid model
% for xx = 1:1:ceil(Right_boundary/SU_interval)    % SU_interval:SU_interval:Right_boundary
%     for yy = 1:1:num_SU_perRing
%         Next_Coord = (xx-1)*num_SU_perRing + yy;
%         CoordXY_SU_actual(Next_Coord,:) = GetCoord2D([0 0],xx*SU_interval,yy*360/num_SU_perRing);
%     end
% end
% % Actual SU Location Grid (Network Topology)
% SU_interval = floor(Netw_width/(round(sqrt(num_SecondaryUsers))-1));
% length_boundary_SU = length(Left_boundary:SU_interval:Right_boundary);
% CoordXY_SU_actual = zeros(num_SecondaryUsers,2); % SU-grid model
% for xx = Left_boundary:SU_interval:Right_boundary
%     for yy = Left_boundary:SU_interval:Right_boundary
%         Next_Coord = round(  ((yy/SU_interval)+(1-Left_boundary/SU_interval))  +  (((xx/SU_interval)+(1-Left_boundary/SU_interval))-1)*length_boundary_SU  );        
%         CoordXY_SU_actual(Next_Coord,1)=yy;
%         CoordXY_SU_actual(Next_Coord,2)=xx;
%     end  
% end

% Calculating the actual SU-PU Distance (L_SU4rmPU_actual)
L_SU4rmPU_actual = zeros(num_SecondaryUsers,num_PrimaryUsers);  % SU-PU distance
for nn = 1:num_SecondaryUsers
    for mm = 1:num_PrimaryUsers
        L_SU4rmPU_actual(nn,mm) = GetDistance2D(CoordXY_SU_actual(nn,:), CoordXY_PU_actual(mm,:));
    end
end

% Calculating the actual received signal strength at the SUs (RSS_actual)
RSS_actual = zeros(num_SecondaryUsers, length(Pf(1,:)));   % Actual received signal strength at each SU (dBm)
for nn = 1:num_SecondaryUsers
    RSS_actual(nn,1) = Pt_PU' + P_TXAntennaGain + P_RXAntennaGain(nn) - (10*n_PLE).*(log10( (4*pi/Lambda).*L_SU4rmPU_actual(nn,:) )); 
end
RSS_actual = repmat(RSS_actual(:,1),1,length(Pf(1,:)));

% Calculating the predicted received signal strength at the SUs
RSS_pred = pow2db(db2pow(RSS_actual) .* (0.8 + (1.0-0.8).*rand(num_SecondaryUsers, length(Pf(1,:)))));

% Measured RSS
RSS_meas = pow2db(db2pow(RSS_actual) + db2pow(Pn_SU_actual));

% Network Topology Plot
Fig1 = figure(1);
%set(Fig1, 'Position', [0 0 0 0])
plot3(CoordXY_PU_actual(:,1),CoordXY_PU_actual(:,2),0.*CoordXY_PU_actual(:,2),[graph_color(2), graph_ID(7)],'MarkerSize',graphMarker_size(5),'MarkerFaceColor','r'), axis equal
hold on
for nn = 1:num_SecondaryUsers
    if (CoordXY_SU_actual(nn,1) == 0 && CoordXY_SU_actual(nn,2) == 0)
    elseif(1)
        plot3(CoordXY_SU_actual(nn,1),CoordXY_SU_actual(nn,2),0.*CoordXY_SU_actual(nn,2),[graph_color(1), graph_ID(1)],'MarkerSize',graphMarker_size(1),'MarkerFaceColor','b'), axis equal
    end
    hold on
end
%title('Network Topology'); 
box on
grid on
xlabel('x');
ylabel('y');
legend('Primary User (PU)','Secondary Users (SU)');
axis([-(Netw_width/2+300),Netw_width/2+300,-(Netw_width/2+300),Netw_width/2+300]);

%%    CASE A: Noise Power = Actual for all SUs; RSS = Actual for all SUs.    %%

% Noise power Estimates
Pn_SU_caseA = Pn_SU_actual; % in (dBm)

% Calculating the received signal strength at the SUs
RSS_caseA = RSS_actual;

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseA = db2pow(RSS_caseA - Pn_SU_caseA);

% Energy Threshold 
%threshold_Energy_caseA = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseA = (qfuncinv(Pf)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseA = db2pow(Pn_SU_caseA) .* ((qfuncinv(Pf)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseA = marcumq(sqrt(2.*snr_at_SU_caseA), sqrt(threshold_Energy_caseA), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseA = qfunc(((threshold_Energy_caseA - (snr_at_SU_caseA + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseA) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseA = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseA) + db2pow(Pn_SU_caseA) )) .* ( db2pow(RSS_caseA) + db2pow(Pn_SU_caseA) -threshold_Energy_caseA)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseA = zeros(num_SecondaryUsers,length(0.01:0.01:1.0)); % Initializing OR-fused false alarm probability
Qd_caseA = zeros(num_SecondaryUsers,length(0.01:0.01:1.0)); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseA(kk,:) = Qf_caseA(kk,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf(nn,:)).^nn .* (1-Pf(nn,:)).^(kk-nn);
        Qd_caseA(kk,:) = Qd_caseA(kk,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseA(nn,:)).^nn .* (1-Pd_caseA(nn,:)).^(kk-nn);
    end
end

% Fused Probability of missed detection
Qm_caseA = 1 - Qd_caseA;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseA = Qf_caseA + Qm_caseA;

% Plot: Number of SU's versus Qm
Fig2 = figure(2);

Fig3 = figure(3);

%%    CASE Bb: Noise Power = Estimated (-74:2:-62 dBm) for all SUs; RSS = Actual for all SUs.    %%

% Noise power Estimates in (dBm)
Pn_SU_caseBb_vector = -6+Pn_SU_actual_vector:2:6+Pn_SU_actual_vector; % in (dBm)
length_caseBb = length(Pn_SU_caseBb_vector);
Pn_SU_caseBb = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb); % in (dBm)
for nnmm = 1:1:length_caseBb
    Pn_SU_caseBb(:,:,nnmm) = Pn_SU_caseBb_vector(nnmm).*ones(num_SecondaryUsers,length(Pf(1,:))); % in (dBm)
end

% Calculating the received signal strength at the SUs
RSS_caseBb = repmat(RSS_actual,1,1,length_caseBb);

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseBb = db2pow(RSS_caseBb - Pn_SU_caseBb);

% Energy Threshold
Pf_caseBb = repmat(Pf,1,1,length_caseBb);
%threshold_Energy_caseBb = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseBb = (qfuncinv(Pf_caseBb)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseBb = db2pow(Pn_SU_caseBb) .* ((qfuncinv(Pf_caseBb)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseBb(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseBb), sqrt(threshold_Energy_caseBb), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseBb = qfunc(((threshold_Energy_caseBb - (snr_at_SU_caseBb + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseBb) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseBb = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseBb) + db2pow(Pn_SU_caseBb) )) .* ( db2pow(RSS_caseBb) + db2pow(Pn_SU_caseBb) -threshold_Energy_caseBb)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseBb = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb); % Initializing OR-fused false alarm probability
Qd_caseBb = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseBb(kk,:,:) = Qf_caseBb(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseBb(nn,:,:)).^nn .* (1-Pf_caseBb(nn,:,:)).^(kk-nn);
        Qd_caseBb(kk,:,:) = Qd_caseBb(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseBb(nn,:,:)).^nn .* (1-Pd_caseBb(nn,:,:)).^(kk-nn);
    end 
end

% Probability of missed detection
Qm_caseBb = 1 - Qd_caseBb;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseBb = Qf_caseBb + Qm_caseBb;

% Root Mean Square Error (RMSE) for Pd_caseBb
RMSE_Pd_caseBb = zeros(length_caseBb,length(Pf(1,:)));    
for nnmm = 1:1:length_caseBb
    RMSE_Pd_caseBb(nnmm,:) = sqrt( sum( (Pd_caseBb(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end  

% Plot: Number of SU's versus Qm
Fig7 = figure(7);

Fig8 = figure(8);

% Plot
Fig9 = figure(9);
   
%%                  Case Bb2: Applying BSDS to case Bb                   %%

% Synthetic noise power
length_caseBb2 = length_caseBb;
Pn_SU_caseBb2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb2); % in (dBm)
r_est_caseBb2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb2); % Reliability of estimated noise power
r_meas_caseBb2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb2); % Reliability of measured noise power    
r_pred_caseBb2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb2); % Reliability of predicted noise power
for nnmm = 1:1:length_caseBb    
    for nn = 1:num_SecondaryUsers
        for mm = 1:length(Pf(1,:))
            r_est_caseBb2(nn,mm,nnmm) = 0.5 - 0.5.*abs(db2pow(Pn_SU_caseBb(nn,mm,nnmm)) - (db2pow(RSS_meas(nn,mm))-db2pow(RSS_pred(nn,mm)))) ./ (db2pow(RSS_meas(nn,mm))-db2pow(RSS_pred(nn,mm)));
            if (r_est_caseBb2(nn,mm,nnmm) > 0.5)
                Pn_SU_caseBb2(nn,mm,nnmm) = Pn_SU_caseBb(nn,mm,nnmm);
            elseif(1)
                if (r_est_caseBb2(nn,mm,nnmm) < 0.0), r_est_caseBb2(nn,mm,nnmm) = 0.0; end
                r_meas_caseBb2(nn,mm,nnmm) = 1 - r_est_caseBb2(nn,mm,nnmm);
                r_pred_caseBb2(nn,mm,nnmm) = r_meas_caseBb2(nn,mm,nnmm);        
                Pn_SU_caseBb2(nn,mm,nnmm) = pow2db( r_est_caseBb2(nn,mm,nnmm).*db2pow(Pn_SU_caseBb(nn,mm,nnmm)) + r_meas_caseBb2(nn,mm,nnmm).*(db2pow(RSS_meas(nn,mm))-db2pow(RSS_pred(nn,mm))) );
            end
        end 
    end  
end

% Calculating the received signal strength at the SUs
RSS_caseBb2 = RSS_caseBb;

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseBb2 = db2pow(RSS_caseBb2 - Pn_SU_caseBb2);

% Energy Threshold 
Pf_caseBb2 = Pf_caseBb;
%threshold_Energy_caseBb2 = gammaincinv(Pf_caseBb2,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseBb2 = (qfuncinv(Pf_caseBb2)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseBb2 = db2pow(Pn_SU_caseBb2) .* ((qfuncinv(Pf_caseBb2)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseBb2 = marcumq(sqrt(2.*snr_at_SU_caseBb2), sqrt(threshold_Energy_caseBb2), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseBb2 = qfunc(((threshold_Energy_caseBb2 - (snr_at_SU_caseBb2 + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseBb2) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseBb2 = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseBb2) + db2pow(Pn_SU_caseBb2) )) .* ( db2pow(RSS_caseBb2) + db2pow(Pn_SU_caseBb2) -threshold_Energy_caseBb2)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseBb2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb2); % Initializing OR-fused false alarm probability
Qd_caseBb2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseBb2); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseBb2(kk,:,:) = Qf_caseBb2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseBb2(nn,:,:)).^nn .* (1-Pf_caseBb2(nn,:,:)).^(kk-nn);
        Qd_caseBb2(kk,:,:) = Qd_caseBb2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseBb2(nn,:,:)).^nn .* (1-Pd_caseBb2(nn,:,:)).^(kk-nn);
    end 
end

% Fused Probability of missed detection
Qm_caseBb2 = 1 - Qd_caseBb2;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseBb2 = Qf_caseBb2 + Qm_caseBb2;

% Root Mean Square Error (RMSE) for Pd_caseBb2
RMSE_Pd_caseBb2 = zeros(length_caseBb2,length(Pf(1,:)));    
for nnmm = 1:1:length_caseBb2
    RMSE_Pd_caseBb2(nnmm,:) = sqrt( sum( (Pd_caseBb2(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

   % Plot: Number of SU's versus Qm
Fig10 = figure(10);

Fig11 = figure(11);

% Plot
Fig12 = figure(12);

%%    CASE C: Noise Power = Estimated (-74 dBm) for some SUs; RSS = Actual for all SUs.    %%

% Noise power Estimates in (dBm)
caseC = [100 50 25 10 5 2 1];
length_caseC = length(caseC);
Pn_SU_caseC = repmat(Pn_SU_actual,1,1,length_caseC); % in (dBm)
for nn = 2:1:length_caseC
    for mm = caseC(nn):caseC(nn):choosePlot
        if (caseC(nn) == 1)
            Pn_SU_caseC(mm,:,nn) = Pn_SU_caseBb_vector(1).*ones(1,length(Pf(1,:))); % in (dBm)
        else
            Pn_SU_caseC(mm-1,:,nn) = Pn_SU_caseBb_vector(1).*ones(1,length(Pf(1,:))); % in (dBm)
        end
    end
end

% Calculating the received signal strength at the SUs
RSS_caseC = repmat(RSS_actual,1,1,length_caseC);

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseC = db2pow(RSS_caseC - Pn_SU_caseC);

% Energy Threshold 
Pf_caseC = repmat(Pf,1,1,length_caseC);
%threshold_Energy_caseC = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseC = (qfuncinv(Pf)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseC = db2pow(Pn_SU_caseC) .* ((qfuncinv(Pf_caseC)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseC(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseC), sqrt(threshold_Energy_caseC), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseC = qfunc(((threshold_Energy_caseC - (snr_at_SU_caseC + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseC) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseC = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseC) + db2pow(Pn_SU_caseC) )) .* ( db2pow(RSS_caseC) + db2pow(Pn_SU_caseC) -threshold_Energy_caseC)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseC = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC); % Initializing OR-fused false alarm probability
Qd_caseC = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseC(kk,:,:) = Qf_caseC(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseC(nn,:,:)).^nn .* (1-Pf_caseC(nn,:,:)).^(kk-nn);
        Qd_caseC(kk,:,:) = Qd_caseC(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseC(nn,:,:)).^nn .* (1-Pd_caseC(nn,:,:)).^(kk-nn);
    end 
end

% Probability of missed detection
Qm_caseC = 1 - Qd_caseC;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseC = Qf_caseC + Qm_caseC;

% Root Mean Square Error (RMSE) for Pd_caseC
RMSE_Pd_caseC = zeros(length_caseC,length(Pf(1,:)));
for nnmm = 1:1:length_caseC
    RMSE_Pd_caseC(nnmm,:) = sqrt( sum( (Pd_caseC(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

% Plot: Number of SU's versus Qm
Fig13 = figure(13);   

Fig14 = figure(14);

% Plot
Fig15 = figure(15);
   
%%                  CASE C2: Applying BSDS to case C                     %%

% Synthetic noise power
length_caseC2 = length_caseC;
Pn_SU_caseC2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC2); % in (dBm)
r_est_caseC2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC2); % Reliability of estimated noise power
r_meas_caseC2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC2); % Reliability of measured noise power    
r_pred_caseC2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC2); % Reliability of predicted noise power
for nnmm = 1:1:length_caseC    
    for nn = 1:num_SecondaryUsers
        for mm = 1:length(Pf(1,:))
            r_est_caseC2(nn,mm,nnmm) = 0.5 - 0.5.*abs(db2pow(Pn_SU_caseC(nn,mm,nnmm)) - (db2pow(RSS_meas(nn,mm))-db2pow(RSS_pred(nn,mm)))) ./ (db2pow(RSS_meas(nn,mm))-db2pow(RSS_pred(nn,mm)));
            if (r_est_caseC2(nn,mm,nnmm) > 0.5)
                Pn_SU_caseC2(nn,mm,nnmm) = Pn_SU_caseC(nn,mm,nnmm);
            elseif(1)
                if (r_est_caseC2(nn,mm,nnmm) < 0.0), r_est_caseC2(nn,mm,nnmm) = 0.0; end
                r_meas_caseC2(nn,mm,nnmm) = 1 - r_est_caseC2(nn,mm,nnmm);
                r_pred_caseC2(nn,mm,nnmm) = r_meas_caseC2(nn,mm,nnmm);        
                Pn_SU_caseC2(nn,mm,nnmm) = pow2db( r_est_caseC2(nn,mm,nnmm).*db2pow(Pn_SU_caseC(nn,mm,nnmm)) + r_meas_caseC2(nn,mm,nnmm).*(db2pow(RSS_meas(nn,mm))-db2pow(RSS_pred(nn,mm))) );
            end
        end 
    end  
end

% Calculating the received signal strength at the SUs
RSS_caseC2 = repmat(RSS_actual,1,1,length_caseC2);

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseC2 = db2pow(RSS_caseC2 - Pn_SU_caseC2);

% Energy Threshold 
Pf_caseC2 = repmat(Pf,1,1,length_caseC2);
%threshold_Energy_caseC2 = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseC2 = (qfuncinv(Pf)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseC2 = db2pow(Pn_SU_caseC2) .* ((qfuncinv(Pf_caseC2)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseC2(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseC2), sqrt(threshold_Energy_caseC2), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseC2 = qfunc(((threshold_Energy_caseC2 - (snr_at_SU_caseC2 + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseC2) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseC2 = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseC2) + db2pow(Pn_SU_caseC2) )) .* ( db2pow(RSS_caseC2) + db2pow(Pn_SU_caseC2) -threshold_Energy_caseC2)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseC2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC2); % Initializing OR-fused false alarm probability
Qd_caseC2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseC2); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseC2(kk,:,:) = Qf_caseC2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseC2(nn,:,:)).^nn .* (1-Pf_caseC2(nn,:,:)).^(kk-nn);
        Qd_caseC2(kk,:,:) = Qd_caseC2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseC2(nn,:,:)).^nn .* (1-Pd_caseC2(nn,:,:)).^(kk-nn);
    end
end

% Probability of missed detection
Qm_caseC2 = 1 - Qd_caseC2;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseC2 = Qf_caseC2 + Qm_caseC2;

% Root Mean Square Error (RMSE) for Pd_caseC2
RMSE_Pd_caseC2 = zeros(length_caseC2,length(Pf(1,:)));
for nnmm = 1:1:length_caseC2
    RMSE_Pd_caseC2(nnmm,:) = sqrt( sum( (Pd_caseC2(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

% Plot: Number of SU's versus Qm
Fig16 = figure(16);    

Fig17 = figure(17);

% Plot
Fig18 = figure(18); 

%%    CASE D: Noise Power = Actual for all SUs; RSS = Falsified (+/-2 to +/-6dBm) for all SUs; Main Variable = RSS deviation    %%
length_caseD = 7;

% Noise power Estimates in (dBm)
Pn_SU_caseD = repmat(Pn_SU_actual,1,1,length_caseD); % in (dBm)

% Calculating the received signal strength at the SUs
RSS_caseD = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD);
for nnmm = -2*floor(length_caseD/2):2:2*floor(length_caseD/2)
    RSS_caseD(:,:,(nnmm/2)+ceil(length_caseD/2)) = (RSS_actual+nnmm).*ones(num_SecondaryUsers,length(Pf(1,:))); % in (dBm)
end

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseD = db2pow(RSS_caseD - Pn_SU_caseD);

% Energy Threshold
Pf_caseD = repmat(Pf,1,1,length_caseD);
%threshold_Energy_caseD = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseD = (qfuncinv(Pf_caseD)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseD = db2pow(Pn_SU_caseD) .* ((qfuncinv(Pf_caseD)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseD(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseD), sqrt(threshold_Energy_caseD), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseD = qfunc(((threshold_Energy_caseD - (snr_at_SU_caseD + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseD) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseD = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseD) + db2pow(Pn_SU_caseD) )) .* ( db2pow(RSS_caseD) + db2pow(Pn_SU_caseD) -threshold_Energy_caseD)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseD = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD); % Initializing OR-fused false alarm probability
Qd_caseD = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseD(kk,:,:) = Qf_caseD(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseD(nn,:,:)).^nn .* (1-Pf_caseD(nn,:,:)).^(kk-nn);
        Qd_caseD(kk,:,:) = Qd_caseD(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseD(nn,:,:)).^nn .* (1-Pd_caseD(nn,:,:)).^(kk-nn);
    end
end

% Probability of missed detection
Qm_caseD = 1 - Qd_caseD;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseD = Qf_caseD + Qm_caseD;

% Root Mean Square Error (RMSE) for Pd_caseD
RMSE_Pd_caseD = zeros(length_caseD,length(Pf(1,:)));    
for nnmm = 1:1:length_caseD
    RMSE_Pd_caseD(nnmm,:) = sqrt( sum( (Pd_caseD(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

% Plot: Number of SU's versus Qm
Fig19 = figure(19);

Fig20 = figure(20);

% Plot
Fig21 = figure(21);

%%                  CASE D2: Applying BSDS to Case D                     %%
length_caseD2 = length_caseD;

% Noise power Estimates in (dBm)
Pn_SU_caseD2 = repmat(Pn_SU_actual,1,1,length_caseD2); % in (dBm)

% Synthetic signal power
RSS_caseD2 = RSS_caseD;
r_est_caseD2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD2); % Reliability of estimated noise power
r_meas_caseD2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD2); % Reliability of measured noise power    
r_pred_caseD2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD2); % Reliability of predicted noise power
for nnmm = 1:1:length_caseD2  
    for nn = 1: num_SecondaryUsers
        for mm = 1:length(Pf(1,:))
            r_meas_caseD2(nn,mm,nnmm) = 0.5 - 0.5.*abs( db2pow(RSS_caseD2(nn,mm,nnmm)) - db2pow(RSS_pred(nn,mm)) ) ./ db2pow(RSS_pred(nn,mm));
            if (r_meas_caseD2(nn,mm,nnmm) < 0.0), r_meas_caseD2(nn,mm,nnmm) = 0.0; end
            r_pred_caseD2(nn,mm,nnmm) = 1 - r_meas_caseD2(nn,mm,nnmm);
            r_est_caseD2(nn,mm,nnmm) = r_meas_caseD2(nn,mm,nnmm);
        end
    end
    RSS_caseD2(:,:,nnmm) = pow2db( r_meas_caseD2(:,:,nnmm).*db2pow(RSS_caseD2(:,:,nnmm)) + r_pred_caseD2(:,:,nnmm).*db2pow(RSS_pred) );
end

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseD2 = db2pow(RSS_caseD2 - Pn_SU_caseD2);

% Energy Threshold
Pf_caseD2 = repmat(Pf,1,1,length_caseD2);
%threshold_Energy_caseD2 = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseD2 = (qfuncinv(Pf_caseD2)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseD2 = db2pow(Pn_SU_caseD2) .* ((qfuncinv(Pf_caseD2)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseD2(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseD2), sqrt(threshold_Energy_caseD2), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseD2 = qfunc(((threshold_Energy_caseD2 - (snr_at_SU_caseD2 + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseD2) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseD2 = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseD2) + db2pow(Pn_SU_caseD2) )) .* ( db2pow(RSS_caseD2) + db2pow(Pn_SU_caseD2) -threshold_Energy_caseD2)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseD2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD2); % Initializing OR-fused false alarm probability
Qd_caseD2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseD2); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseD2(kk,:,:) = Qf_caseD2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseD2(nn,:,:)).^nn .* (1-Pf_caseD2(nn,:,:)).^(kk-nn);
        Qd_caseD2(kk,:,:) = Qd_caseD2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseD2(nn,:,:)).^nn .* (1-Pd_caseD2(nn,:,:)).^(kk-nn);
    end
end

% Probability of missed detection
Qm_caseD2 = 1 - Qd_caseD2;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseD2 = Qf_caseD2 + Qm_caseD2;

% Root Mean Square Error (RMSE) for Pd_caseD2
RMSE_Pd_caseD2 = zeros(length_caseD2,length(Pf(1,:)));    
for nnmm = 1:1:length_caseD2
    RMSE_Pd_caseD2(nnmm,:) = sqrt( sum( (Pd_caseD2(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end  

% Plot: Number of SU's versus Qm
Fig192 = figure(192);

Fig202 = figure(202);

% Plot
Fig212 = figure(212);

%%    CASE E: Noise Power = Actual for all SUs; RSS = Falsified (+6dBm) for some SUs; Main Variable = Number of mal SUs    %%

caseE = [100 50 25 10 5 2 1];
length_caseE = length(caseE);

% Noise power Estimates in (dBm)
Pn_SU_caseE = repmat(Pn_SU_actual,1,1,length_caseE); % in (dBm)

% Calculating the received signal strength at the SUs
RSS_caseE = repmat(RSS_actual,1,1,length_caseE);
for nn = 2:1:length_caseE
    for mm = caseE(nn):caseE(nn):choosePlot
        if (caseE(nn) == 1)
            RSS_caseE(mm,:,nn) = RSS_caseD(mm,:,length_caseD);    %RSS_caseE(mm,:,nn) + 3;
        else
            RSS_caseE(mm-1,:,nn) = RSS_caseD(mm-1,:,length_caseD);    %RSS_caseE(mm-1,:,nn) + 3;
        end
    end
end

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseE = db2pow(RSS_caseE - Pn_SU_caseE);

% Energy Threshold 
Pf_caseE = repmat(Pf,1,1,length_caseE);
%threshold_Energy_caseE = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseE = (qfuncinv(Pf)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseE = db2pow(Pn_SU_caseE) .* ((qfuncinv(Pf_caseE)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseE(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseE), sqrt(threshold_Energy_caseE), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseE = qfunc(((threshold_Energy_caseE - (snr_at_SU_caseE + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseE) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseE = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseE) + db2pow(Pn_SU_caseE) )) .* ( db2pow(RSS_caseE) + db2pow(Pn_SU_caseE) -threshold_Energy_caseE)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseE = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseE); % Initializing OR-fused false alarm probability
Qd_caseE = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseE); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseE(kk,:,:) = Qf_caseE(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseE(nn,:,:)).^nn .* (1-Pf_caseE(nn,:,:)).^(kk-nn);
        Qd_caseE(kk,:,:) = Qd_caseE(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseE(nn,:,:)).^nn .* (1-Pd_caseE(nn,:,:)).^(kk-nn);
    end 
end

% Probability of missed detection
Qm_caseE = 1 - Qd_caseE;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseE = Qf_caseE + Qm_caseE;

% Root Mean Square Error (RMSE) for Pd_caseE
RMSE_Pd_caseE = zeros(length_caseE,length(Pf(1,:)));
for nnmm = 1:1:length_caseE
    RMSE_Pd_caseE(nnmm,:) = sqrt( sum( (Pd_caseE(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

% Plot: Number of SU's versus Qm
Fig22 = figure(22);  

Fig23 = figure(23);

% Plot
Fig24 = figure(24);  
   
%%                  CASE E2: Applying BSDS to case E                     %%

caseE2 = [100 50 25 10 5 2 1];
length_caseE2 = length(caseE2);

% Noise power Estimates in (dBm)
Pn_SU_caseE2 = repmat(Pn_SU_actual,1,1,length_caseE2); % in (dBm)

% Synthetic signal power 
RSS_caseE2 = RSS_caseE;
r_est_caseE2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseE2); % Reliability of estimated noise power
r_meas_caseE2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseE2); % Reliability of measured noise power    
r_pred_caseE2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseE2); % Reliability of predicted noise power
for nnmm = 1:1:length_caseE2  
    for nn = 1: num_SecondaryUsers
        for mm = 1:length(Pf(1,:))
            r_meas_caseE2(nn,mm,nnmm) = 0.5 - 0.5.*abs( db2pow(RSS_caseE2(nn,mm,nnmm)) - db2pow(RSS_pred(nn,mm)) ) ./ db2pow(RSS_pred(nn,mm));
            if (r_meas_caseE2(nn,mm,nnmm) < 0.0), r_meas_caseE2(nn,mm,nnmm) = 0.0; end
            r_pred_caseE2(nn,mm,nnmm) = 1 - r_meas_caseE2(nn,mm,nnmm);
            r_est_caseE2(nn,mm,nnmm) = r_meas_caseE2(nn,mm,nnmm);        
        end 
    end    
    RSS_caseE2(:,:,nnmm) = pow2db( r_meas_caseE2(:,:,nnmm).*db2pow(RSS_caseE2(:,:,nnmm)) + r_pred_caseE2(:,:,nnmm).*db2pow(RSS_pred) );
end

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseE2 = db2pow(RSS_caseE2 - Pn_SU_caseE2);

% Energy Threshold 
Pf_caseE2 = repmat(Pf,1,1,length_caseE2);
%threshold_Energy_caseE2 = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseE2 = (qfuncinv(Pf)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseE2 = db2pow(Pn_SU_caseE2) .* ((qfuncinv(Pf_caseE2)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseE2(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseE2), sqrt(threshold_Energy_caseE2), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseE2 = qfunc(((threshold_Energy_caseE2 - (snr_at_SU_caseE2 + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseE2) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseE2 = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseE2) + db2pow(Pn_SU_caseE2) )) .* ( db2pow(RSS_caseE2) + db2pow(Pn_SU_caseE2) -threshold_Energy_caseE2)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseE2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseE2); % Initializing OR-fused false alarm probability
Qd_caseE2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseE2); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseE2(kk,:,:) = Qf_caseE2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseE2(nn,:,:)).^nn .* (1-Pf_caseE2(nn,:,:)).^(kk-nn);
        Qd_caseE2(kk,:,:) = Qd_caseE2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseE2(nn,:,:)).^nn .* (1-Pd_caseE2(nn,:,:)).^(kk-nn);
    end 
end

% Probability of missed detection
Qm_caseE2 = 1 - Qd_caseE2;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseE2 = Qf_caseE2 + Qm_caseE2;

% Root Mean Square Error (RMSE) for Pd_caseE2
RMSE_Pd_caseE2 = zeros(length_caseE2,length(Pf(1,:)));
for nnmm = 1:1:length_caseE2
    RMSE_Pd_caseE2(nnmm,:) = sqrt( sum( (Pd_caseE2(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

% Plot: Number of SU's versus Qm
Fig222 = figure(222);

Fig232 = figure(232); 

% Plot
Fig242 = figure(242); 

%%    CASE F: Noise Power = Actual for all SUs; RSS = Falsified (+6dBm) for 50% of SUs; Main Variable = Distribution of mal SUs    %%

caseF = [1 2 3 4 5 6 7];
length_caseF = length(caseF);

% Noise power Estimates in (dBm)
Pn_SU_caseF = repmat(Pn_SU_actual,1,1,length_caseF); % in (dBm)

% Calculating the received signal strength at the SUs
RSS_caseF = repmat(RSS_actual,1,1,length_caseF);
for mm = floor(choosePlot/2)+1:1:choosePlot
    RSS_caseF(mm,:,1) = RSS_caseD(mm,:,length_caseD);    %RSS_caseF(mm-1,:,nn) + 3;
end
for mm = [1:1:ceil(choosePlot*(1/8)),ceil(choosePlot*(5/8))+1:1:choosePlot]
    RSS_caseF(mm,:,2) = RSS_caseD(mm,:,length_caseD);    %RSS_caseF(mm-1,:,nn) + 3;
end
for mm = [1:1:floor(choosePlot*(1/4)),ceil(choosePlot*(3/4)):1:choosePlot]
    RSS_caseF(mm,:,3) = RSS_caseD(mm,:,length_caseD);    %RSS_caseF(mm-1,:,nn) + 3;
end
for mm = 2:2:choosePlot
    RSS_caseF(mm-1,:,5) = RSS_caseD(mm-1,:,length_caseD);    %RSS_caseF(mm-1,:,nn) + 3;
end
for mm = [1:1:floor(choosePlot*(3/8)),ceil(choosePlot*(7/8)):1:choosePlot]
    RSS_caseF(mm,:,6) = RSS_caseD(mm,:,length_caseD);    %RSS_caseF(mm-1,:,nn) + 3;
end
for mm = 1:1:ceil(choosePlot/2)
    RSS_caseF(mm,:,7) = RSS_caseD(mm,:,length_caseD);    %RSS_caseF(mm-1,:,nn) + 3;
end        

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseF = db2pow(RSS_caseF - Pn_SU_caseF);

% Energy Threshold 
Pf_caseF = repmat(Pf,1,1,length_caseF);
%threshold_Energy_caseF = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseF = (qfuncinv(Pf)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseF = db2pow(Pn_SU_caseF) .* ((qfuncinv(Pf_caseF)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseF(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseF), sqrt(threshold_Energy_caseF), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseF = qfunc(((threshold_Energy_caseF - (snr_at_SU_caseF + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseF) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseF = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseF) + db2pow(Pn_SU_caseF) )) .* ( db2pow(RSS_caseF) + db2pow(Pn_SU_caseF) -threshold_Energy_caseF)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseF = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseF); % Initializing OR-fused false alarm probability
Qd_caseF = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseF); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseF(kk,:,:) = Qf_caseF(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseF(nn,:,:)).^nn .* (1-Pf_caseF(nn,:,:)).^(kk-nn);
        Qd_caseF(kk,:,:) = Qd_caseF(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseF(nn,:,:)).^nn .* (1-Pd_caseF(nn,:,:)).^(kk-nn);
    end
end

% Probability of missed detection
Qm_caseF = 1 - Qd_caseF;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseF = Qf_caseF + Qm_caseF;

% Root Mean Square Error (RMSE) for Pd_caseF
RMSE_Pd_caseF = zeros(length_caseF,length(Pf(1,:)));
for nnmm = 1:1:length_caseF
    RMSE_Pd_caseF(nnmm,:) = sqrt( sum( (Pd_caseF(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

% Plot: Number of SU's versus Qm
Fig25 = figure(25);    

Fig26 = figure(26);  

% Plot
Fig27 = figure(27); 
   
%%                  CASE F2: Applying BSDS to case F                     %%

caseF2 = [1 2 3 4 5 6 7];
length_caseF2 = length(caseF2);

% Noise power Estimates in (dBm)
Pn_SU_caseF2 = repmat(Pn_SU_actual,1,1,length_caseF2); % in (dBm)

% Synthetic signal power 
RSS_caseF2 = RSS_caseF;
r_est_caseF2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseF2); % Reliability of estimated noise power
r_meas_caseF2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseF2); % Reliability of measured noise power    
r_pred_caseF2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseF2); % Reliability of predicted noise power
for nnmm = 1:1:length_caseF2  
    for nn = 1: num_SecondaryUsers
        for mm = 1:length(Pf(1,:))
            r_meas_caseF2(nn,mm,nnmm) = 0.5 - 0.5.*abs( db2pow(RSS_caseF2(nn,mm,nnmm)) - db2pow(RSS_pred(nn,mm)) ) ./ db2pow(RSS_pred(nn,mm));
            if (r_meas_caseF2(nn,mm,nnmm) < 0.0), r_meas_caseF2(nn,mm,nnmm) = 0.0; end
            r_pred_caseF2(nn,mm,nnmm) = 1 - r_meas_caseF2(nn,mm,nnmm);
            r_est_caseF2(nn,mm,nnmm) = r_meas_caseF2(nn,mm,nnmm);        
        end 
    end    
    RSS_caseF2(:,:,nnmm) = pow2db( r_meas_caseF2(:,:,nnmm).*db2pow(RSS_caseF2(:,:,nnmm)) + r_pred_caseF2(:,:,nnmm).*db2pow(RSS_pred) );
end

% Signal to Noise Ratio, SNR @ Secondary Users (SU)
snr_at_SU_caseF2 = db2pow(RSS_caseF2 - Pn_SU_caseF2);

% Energy Threshold 
Pf_caseF2 = repmat(Pf,1,1,length_caseF2);
%threshold_Energy_caseF2 = gammaincinv(Pf,numSamples/2,'upper'); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%threshold_Energy_caseF2 = (qfuncinv(Pf)./sqrt(numSamples)) + 1; % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
threshold_Energy_caseF2 = db2pow(Pn_SU_caseF2) .* ((qfuncinv(Pf_caseF2)./sqrt(numSamples)) + 1); % IEEE 802.22

% Probability of Detection, Pd at SU's
%Pd_caseF2(:,:,nnmm) = marcumq(sqrt(2.*snr_at_SU_caseF2), sqrt(threshold_Energy_caseF2), numSamples/2); % Ref: W. Zhang, R.K. Mallik, K.B. Letaief "Optimization of Cooperative Spectrum Sensing with Energy Detection in Cognitive Radio Networks"
%Pd_caseF2 = qfunc(((threshold_Energy_caseF2 - (snr_at_SU_caseF2 + 1)).*sqrt(numSamples)) ./ (sqrt(2).*((snr_at_SU_caseF2) + 1))); % (from ACM paper) Ref: Y. C. Liang, "Sensing Throughput Tradeoff in Cognitive Radio"
Pd_caseF2 = 1 - qfunc(((sqrt(numSamples))./( db2pow(RSS_caseF2) + db2pow(Pn_SU_caseF2) )) .* ( db2pow(RSS_caseF2) + db2pow(Pn_SU_caseF2) -threshold_Energy_caseF2)); % IEEE 802.22

% Data fusion by Voting rule with T=0.5
Qf_caseF2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseF2); % Initializing OR-fused false alarm probability
Qd_caseF2 = zeros(num_SecondaryUsers,length(Pf(1,:)),length_caseF2); % Initializing OR-fused detection probability
for kk = 1:num_SecondaryUsers
    for nn = ceil(kk/2):kk
        Qf_caseF2(kk,:,:) = Qf_caseF2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pf_caseF2(nn,:,:)).^nn .* (1-Pf_caseF2(nn,:,:)).^(kk-nn);
        Qd_caseF2(kk,:,:) = Qd_caseF2(kk,:,:) + factorial(kk)/(factorial(nn)*factorial(kk-nn)) .* (Pd_caseF2(nn,:,:)).^nn .* (1-Pd_caseF2(nn,:,:)).^(kk-nn);
    end
end

% Probability of missed detection
Qm_caseF2 = 1 - Qd_caseF2;

% Total error rate [Qfa + (1-Qm)]
Qerr_caseF2 = Qf_caseF2 + Qm_caseF2;

% Root Mean Square Error (RMSE) for Pd_caseF2
RMSE_Pd_caseF2 = zeros(length_caseF2,length(Pf(1,:)));
for nnmm = 1:1:length_caseF2
    RMSE_Pd_caseF2(nnmm,:) = sqrt( sum( (Pd_caseF2(1:1:choosePlot,:,nnmm) - Pd_caseA(1:1:choosePlot,:)).^2 ) ./ choosePlot );
end

% Plot: Number of SU's versus Qm
Fig252 = figure(252);  

Fig262 = figure(262);

% Plot
Fig272 = figure(272);
   
%%                    CASE G: Radio Environment Map                      %%

% Plot
Fig28 = figure(28);
plotxx = CoordXY_SU_actual(:,1);
plotyy = CoordXY_SU_actual(:,2);
plotzz1 = RSS_actual(:,1) + Pn_SU_actual(:,1);
plotzz2 = RSS_caseBb(:,1,length_caseBb) + Pn_SU_caseBb(:,1,length_caseBb);
plotzz3 = RSS_caseBb2(:,1,length_caseBb2) + Pn_SU_caseBb2(:,1,length_caseBb2);
plot3(plotxx,plotyy,plotzz1,[graph_color(1),'-',graph_ID(1)],plotxx,plotyy,plotzz2,[graph_color(2),'-',graph_ID(2)],plotxx,plotyy,plotzz3,[graph_color(3),'-',graph_ID(3)]);
%title('---');
grid on
%axis([0,0,0,0]);
xlabel('2-D x-coordinate');
ylabel('2-D y-coordinate');
zlabel('Measured received signal strength (dBm)');
legend('Actual signal and noise power',...
       'Actual signal and estimated noise power',...
       'Actual signal and synthetic noise power');

%%                   CASE G2: Radio Environment Map                      %%

% Plot
Fig29 = figure(29);
plotxx = CoordXY_SU_actual(35:1:50,1);
plotyy = CoordXY_SU_actual(35:1:50,2);
plotzz1 = RSS_actual(35:1:50,1);
plotzz2 = RSS_caseD(35:1:50,1,1);
plotzz3 = RSS_caseD2(35:1:50,1,1);
plot3(plotxx,plotyy,plotzz1,'k-s',...
      plotxx,plotyy,plotzz2,'r-^',...
      plotxx,plotyy,plotzz3,'b-o');
%title('---');
grid on
%axis([0,0,0,0]);
xlabel('x');
ylabel('y');
zlabel('Measured received signal strength, RSS (dBm)');
legend('Actual RSS',...
       'Falsified RSS, (Actual RSS - 6.0) dBm',...
       'Resulting RSS after applying the proposed method');

%%

%---------------------------- END OF CODE --------------------------------%
