clc
clear all
close all
% Parameters
Nt        = 8;        % # transmit antennas
Nr        = Nt;        % # receive antennas

rng(0);                      % reproducibility
SNRdB_range   = 0:2:20; 
% SNRdB_range = 20 * SNRdB_range
% SNRdB_range = 20 * ones(1,11);
test_one_snr_h = 128;
test_one_snr  = test_one_snr_h*4;               
train_one_snr_h = test_one_snr_h*4; 
train_one_snr = train_one_snr_h*4;
train_size    = train_one_snr * numel(SNRdB_range);
data_size     = train_size + numel(SNRdB_range)*test_one_snr;

Ns = test_one_snr_h * 5; 
rho = 0.98;
H_tv   = (randn(Nr,Nt,Ns+1) + 1j*randn(Nr,Nt,Ns+1))/sqrt(2); 

for k = 2:Ns+1
    W = (randn(Nr,Nt) + 1j*randn(Nr,Nt))/sqrt(2);
    H_tv(:,:, ...
        k) = rho*H_tv(:,:,k-1) + sqrt(1-rho^2)*W; 
end

H_tv = permute(H_tv, [3,1,2]); 
perm = randperm(Ns);  

H_t_1   = zeros(Ns,Nr,Nt) + 1j*zeros(Ns,Nr,Nt);
H_t   = zeros(Ns,Nr,Nt) + 1j*zeros(Ns,Nr,Nt);
for i = 1:Ns
    s = perm(i);
    H_t_1(i,:,:) = H_tv(s,:,:);
    H_t(i,:,:) = H_tv(s+1,:,:);
end

csi_error = (randn(Nr,Nt, test_one_snr_h) + 1j*randn(Nr,Nt, test_one_snr_h))*(0.05 * sqrt(2));
csi_error = permute(csi_error, [3,1,2]); 
H_t_1(train_one_snr_h+1:Ns,:,:) = H_t_1(train_one_snr_h+1:Ns,:,:) + csi_error;

M = 4;  
N = round(Nt*0.75);
  
Y   = complex(zeros(data_size, Nr));
X   = complex(zeros(data_size, N));
V   = complex(zeros(data_size, Nr,Nr));
U   = complex(zeros(data_size, Nr,Nr));
U_P   = complex(zeros(data_size, Nr,Nr));
S   = zeros(data_size, Nr,Nr);
var_w   = zeros(data_size,1);
Bit_Data  = zeros(data_size,N);

snr_   = zeros(data_size,1);
H_id   = zeros(data_size,1);
i = 0;
for snr = SNRdB_range
    idxTrain = (i*train_one_snr + 1) : ((i+1)*train_one_snr);
    idxTest  = train_size + (i*test_one_snr + 1) : train_size + ((i+1)*test_one_snr);
    snr_(idxTrain, :) = snr;
    snr_(idxTest, :) = snr; 
%     H_id(idxTrain,:) = (1:train_one_snr)';

    start = i*train_one_snr;
    H_id(start+1                  :start+train_one_snr_h,:) = (1:train_one_snr_h)';
    H_id(start+1+train_one_snr_h   :start+train_one_snr_h*2,:) = (1:train_one_snr_h)';
    H_id(start+1+train_one_snr_h*2 :start+train_one_snr_h*3,:) = (1:train_one_snr_h)';
    H_id(start+1+train_one_snr_h*3 :start+train_one_snr_h*4,:) = (1:train_one_snr_h)';

    start = train_size + i*test_one_snr;
    H_id(start+1                  :start+test_one_snr_h,:) = (train_one_snr_h+1:Ns)';
    H_id(start+1+test_one_snr_h   :start+test_one_snr_h*2,:) = (train_one_snr_h+1:Ns)';
    H_id(start+1+test_one_snr_h*2 :start+test_one_snr_h*3,:) = (train_one_snr_h+1:Ns)';
    H_id(start+1+test_one_snr_h*3 :start+test_one_snr_h*4,:) = (train_one_snr_h+1:Ns)';
    i = i + 1;
end

H = zeros(data_size,Nr,Nt) + 1j*zeros(data_size,Nr,Nt);
H_p = zeros(data_size,Nr,Nt) + 1j*zeros(data_size,Nr,Nt);
for idx = 1:data_size
    data = randi([0,M-1],N,1);
    xi = qammod(data', M, 'UnitAveragePower',true); 
    h_i = H_id(idx,:);
    actual_H = squeeze(H_t(h_i,:,:));
    precode_H = squeeze(H_t_1(h_i,:,:));
    H(idx,:,:) = actual_H;
    H_p(idx,:,:) = precode_H ;
    X(idx, :) = xi;
    Bit_Data(idx, :) = data;
    
    [U_,S_,V_] = svd(precode_H);  
    V(idx,:,:) = V_; 
    U(idx,:,:) = U_;
    S(idx,:,:) = S_;

    Tx = 0;
    for i = 1:N
       Tx = Tx +  V_(:,i) * xi(i);
    end   
    Rx_p = actual_H*Tx; 
    Rx = awgn(Rx_p,snr_(idx,:),'measured');
    Y(idx,:) = Rx;
   
    var_w(idx) = mean(abs(Rx_p(:)).^2) / (10^(snr_(idx,:)/10)); 


    [U_,S_,V_] = svd(actual_H);  
    U_P(idx,:,:) = U_;
end

filename = sprintf('mimoData_%dX%d_tv_svd_qpsk_%.3f.mat', Nt, Nt,rho);
save(filename,'H','Y','X','U','V','H_p','S', 'Bit_Data', 'var_w', 'U_P');
