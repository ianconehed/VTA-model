%% structure parameters
num_columns = 2;
npp = 100;
ppc = 2*npp;
num_VTA = 100;
N = ppc*num_columns + num_VTA;
pop = num_columns*2;
n = N;
unit_num = 1:2:(2*num_columns);
num_trials = 16;

%% time parameters
dt = 1; %time step
T = 100/dt; %time stimulated
T_R = 150/dt; %time of reward
delta = 600/dt; %time in between stimulation
t_total = 2001; %time of trial
D = 10/dt; %intrinsic delay
tau_w = 40; %estimate time window
tau_dopa = 40;
tau_max = 50;

%% input parameters
eG = .04; %input gain excitatory
iG = .04; %input gain inhibitory
VTA_g = .005;
W_in = input_weights_model(num_columns,npp,N,num_VTA,eG,VTA_g,VTA_g);
W_inin = (iG/eG)*W_in;
W_inin(N-num_VTA:N,:) = 0;
unit_stim = unit_num;
t_stim = [101 801 1501 2201 4001];
t_reward = 1500;
p_r = .03*dt; %poisson rate 40 Hz

%% weight matrix parameters
l5_rec = .00016;%.00012;%.00015;%.645; DO NOT SET TO ZERO OR RECURRENT IDENTITY WILL BE WRONG
l23_rec = .00000;
l5_l23 = .0005; %.0002
l23_l5 = .00001;%.4; %DO NOT SET TO ZERO OR FF IDENTITY WILL BE WRONG
i_l5_rec = .0001;
i_l23_rec = .000;
i_l5_l23 = .02;%.07;%.05
i_l23_l5 = 0;
i_l23_l23 = .5;
i_l5_l5 = .2;
l5_VTA = 0.00001;
p_scale = -.1;
p_l5_rec = .0003;
p_l23_rec = .001;
p_l5_l23 = 0;
p_l23_l5 = 0;
l_23_l_VTA = .00000000000000000000001;
l_VTA_l_VTA = 0*.01;

W_ji = Sparse_L_ij(num_columns,npp,N,num_VTA,l5_rec,l23_rec,l5_l23,l23_l5,0,0,0,0,.000);
M_ki = Sparse_L_ij(num_columns,npp,N,num_VTA,i_l5_rec,i_l23_rec,i_l5_l23,i_l23_l5,i_l23_l23,i_l5_l5,l_23_l_VTA,0,.0000);
P_ik = Sparse_L_ij(num_columns,npp,N,num_VTA,p_l5_rec,p_l23_rec,p_l5_l23,p_l23_l5,0,0,0,0,.0001);
rec_identity = W_ji.*L_ij_no_rand(num_columns,npp,N,num_VTA,1,0,0,0,0,0,0)>0;
fin_identity = W_in.*L_ij_no_rand(num_columns,npp,N,num_VTA,0,0,0,0,0,1,0)>0;
ff_identity = W_ji.*L_ij_no_rand(num_columns,npp,N,num_VTA,0,0,0,1,0,0,0)>0;
hebb_identity = M_ki.*L_ij_no_rand(num_columns,npp,N,num_VTA,0,0,0,0,0,0,1)>0;
total_identity = rec_identity + fin_identity + ff_identity;

%% membrane dynamics parameters
rho = 1/7; %percentage change of synaptic activation with input spikes
tau_se = 80; %time constant for excitatory synaptic activation
tau_si = 20; %time constant for inhibitory synaptic activation
tau_si_VTA = 80;
tau_se_VTA = 20;
norm_noise = 1e5;
norm_noise_VTA = 20;
constant_VTA = .035;
norm_noise_VTA_inh = 2.5;
C_m = .2; %membrane capacitance
g_L = .01; %leak conductance
E_i = -70;
E_l = -60; %leak reversal potential
E_e = -5; %excitatory reversal potential
v_th = -55; %threshold potential
v_th_i = -50; 
v_rest = -60; %resting potential
v_hold = -61; %return potential
t_refractory = 2/dt;

%% Learning parameters
tau_p = 1800; %LTP time constant
tau_d = 800; %LTD time constant
T_max_p = .003; %maximum eligibility trace for LTP
T_max_d = .0032; %maximum eligibility trace for LTD
eta_p = 300; %
eta_d = 130; %
eta_rec_l = .003; %learning rate
eta_fin_l = .001; %learning rate
tau_p1 = 2000;% tau_p1 = 200; %LTP time constant
tau_d1 = 1800;% tau_d1 = 800; %LTD time constant
T_max_p1 = .0015;% T_max_p1 = .003; %maximum eligibility trace for LTP
T_max_d1 = .008;% T_max_d1 = .0034; %maximum eligibility trace for LTD
eta_p1 = 650;% eta_p1 = 25*3000; %
eta_d1 = 50;% eta_d1 = 15*3000; %
trace_refractory = zeros(N,N);

eta_hebb_l = [zeros(1,N-num_VTA) abs(.001*rand(1,num_VTA))];
eta_fin_l = 2*[abs(.435*rand(N,N))];
eta_ff_l = .001; %learning rate
eta_p_rec = 1;
K = 1;
select_rec = 1;

num_thresh_rec = 20;
num_thresh_ff = 0;

lam_decay_1 = 0*.0001;
lam_decay_2 = .2;

%% reward parameters
% delay_time = 25/dt; %added reward delay
% t_reward = t_stim(1:end)+(delay_time);
% rew_vect = zeros(t_total,1);
% del_time = round(delay_time/2);
% for m = t_reward
%     rew_vect(m-del_time:m+del_time) = 1;
%     rew_vect(m+del_time+1) = 2;
% end
