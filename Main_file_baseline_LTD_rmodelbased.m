clear all
close all
%% Parameter initialization
run('spiking_parameters_no_lambda_modelbased.m') %most parameters are stored in this file

ROC_dt = 50;
[all_stim,one_stim,plot_R_it] = deal(zeros(pop,t_total/dt,num_trials+1));
R_it_l = zeros(N,t_total/dt,num_trials);
sc_R_it = zeros(N,t_total/dt,num_trials+1);
[ff_vect, ff_vect1] = deal(zeros(1,num_trials+1));
[rec_vect, rec_vect1] = deal(zeros(1,num_trials+1));
[m_vect, m_vect1] = deal(zeros(1,num_trials+1));
hist_vect = zeros(length(N-num_VTA+1:N),(t_total-1)/ROC_dt,num_trials);
hist_vect1 = zeros(length(N-num_VTA+1:N),(t_total-1)/ROC_dt);
auc_plot = zeros(length(N-num_VTA+ 1:N),(t_total-1)/ROC_dt);
lambda = 0;
temp_t = zeros(num_VTA,t_total/dt);
alba1 = 2;
hepp = .000002;
hepc = .002;


%% main program
for l = 1:num_trials
    %% initialization
    [v_yt,v_it,v_kt] = deal(zeros(N,t_total/dt)+v_rest); %membrane potentials
    [R_yt,R_it,R_kt,sc_R] = deal(zeros(N,t_total/dt)); %rates (sc_R is spikes)
    [s_yt,s_it,s_kt] = deal(zeros(N,t_total/dt)); %activations
    [g_Ey,g_Eyi,g_Ei,g_Ii,g_Ek] = deal(zeros(N,t_total/dt)); %conductances
    [t_ref_y,t_ref_i,t_ref_k] = deal(zeros(N,t_total/dt)); %refractory periods
    [T_ijp,T_ijd,del_W_ji,del_M_ki,del_W_in] = deal(zeros(N,N)); %synapse specific traces, weight update
    [T_pt,T_dt,T_maxie_d,T_maxie_p,dopa,dopa_rec,d_der,dopa_rec2,dopa_ff,temp_tracker] = deal(zeros(1,t_total/dt + dt)); %mean trace for population at time t
    
    all_stim1 = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,unit_stim,t_stim,t_reward,T_R,1); %neuron by neuron stimulation, all pops stimulated
    one_stim1 = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,1,t_stim,t_reward,0,1); %neuron by neuron stimulation, first pop stimulated
    for i = 1:pop
        temp = (i-1)*10;
        all_stim(i,:,l) = mean(all_stim1(temp+1:temp+10,:),1); %population stimulation, all pops stimulated
        one_stim(i,:,l) = mean(one_stim1(temp+1:temp+10,:),1); %population stimulation, first pops stimulated
    end
    
    if l == num_trials/4
        W_in = input_weights_model(num_columns,npp,N,num_VTA,eG,VTA_g,.05*VTA_g);
        W_inin = (iG/eG)*W_in;
        W_inin(N-num_VTA:N,:) = 0;
    end
    if l <= num_trials/4
        t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,unit_stim,t_stim,t_reward,T_R,0*1);
%         t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,unit_stim,t_stim,t_reward,T_R,1);
    elseif l<= num_trials/2
        t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,unit_stim,t_stim,t_reward,T_R,0*1);
%         t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,unit_stim,t_stim,t_reward,T_R,1);
    elseif l <= 3*num_trials/4
        t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,3,t_stim,t_reward,T_R,1);
%     elseif l == 3*num_trials/4
%         t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,1,t_stim,t_reward,T_R,1);
    elseif l < num_trials
        t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,1,t_stim,t_reward,T_R,1);
    elseif l == num_trials
%         t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,unit_stim,t_stim,t_reward,0,1);
        t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,1,t_stim,t_reward,T_R,0);
    end
%     t_miy = t_mik1(n,num_VTA,dt,T,t_total,p_r,npp,1,t_stim,t_reward,0);
    if l<=num_trials/4
        eta_hebb = 0*eta_hebb_l;
        eta_ff = 0*eta_ff_l;
        eta_fin = 0*eta_fin_l;
        eta_rec = eta_rec_l;
        decay_on = 0;
    elseif l<=2*num_trials/4
        eta_ff = eta_ff_l;
        eta_hebb = eta_hebb_l;
        eta_fin = eta_fin_l;
        eta_rec = eta_rec_l;
        decay_on = 1;
    elseif l <= 3*num_trials/4
        eta_ff = 0*eta_ff_l;
        eta_hebb = eta_hebb_l;
        eta_fin = eta_fin_l;
        eta_rec = eta_rec_l;
        decay_on = 0;
    elseif l < num_trials
        eta_ff = 0*eta_ff_l;
        eta_hebb = 0*eta_hebb_l;
        eta_fin = 0*eta_fin_l;
        eta_rec = eta_rec_l;
        decay_on = 1;
    elseif l == num_trials
        eta_ff = 0*eta_ff_l;
        eta_hebb = 0*eta_hebb_l;
        eta_fin = 0*eta_fin_l;
        eta_rec = 0*eta_rec_l;
        decay_on = 0;
    end
%     eta_hebb = eta_hebb_l;
%     eta_fin = eta_ff_l;
%     eta_rec = eta_rec_l;
%     decay_on = 0;
    
    R_it(N-num_VTA+1:N,1:30) = abs((.4 + .5*randn([num_VTA,30]))/num_VTA);
    s_it(N-num_VTA+1:N,1:30) = abs((.3 + .3*randn([num_VTA,30]))/num_VTA);
    R_it = R_it.*(R_it>0);
    T_maxie_d(1) = T_max_d;
    T_maxie_p(1) = T_max_p;
    %% time step loop
    for t = 2:((t_total)/dt) %at each time step
        tempie_e = mean(temp);
        %% pre-synaptic loop
        for i = 1:N %over each pre-synaptic neuron
            %% input layer dynamics
            if t_ref_y(i,t) == 1 %refractory period
                v_yt(i,t) = v_rest; %set voltage of neuron i to ressting potential if in refractory period
                t_miy(i,t) = 0; %no spike for neuron i at this time
            elseif (v_it(i,t-1) < v_th) && t_miy(i,t) == 0 %subthreshold update
                del_v_y = (g_L*(E_l-v_yt(i,t-1)))*(dt/C_m); %change in the membrane potential at each time step
                v_yt(i,t) = v_yt(i,t-1) + del_v_y; %update membrane potential
            elseif (v_yt(i,t-1) >= v_th) || t_miy(i,t) == 1
                v_yt(i,t) = v_hold; %voltage resets, neuron enter refractory phase
                t_miy(i,t) = 1; %spike for neuron i at time k
            end
            if v_yt(i,t) == v_hold %if neuron spikes
                del_R_yt = (1/dt-R_yt(i,t-1))*(dt/tau_w);
                R_yt(i,t) = R_yt(i,t-1) + del_R_yt; %update firing rate
                del_s_y = -(s_yt(i,t-1)*dt/tau_si) + rho*(1-s_yt(i,t-1));
                s_yt(i,t) = s_yt(i,t-1) + del_s_y;
                t_ref_y(i,t:t+t_refractory) = 1;
            else %if neuron does not spike
                del_R_yt = -R_yt(i,t-1)*(dt/(tau_w));
                R_yt(i,t) = R_yt(i,t-1) + del_R_yt;
                del_s_y = -s_yt(i,t-1)*(dt/tau_si);
                s_yt(i,t) = s_yt(i,t-1) + del_s_y;
            end
            
            %% excitatory dynamics
            if t_ref_i(i,t) == 1 %refractory period
                v_it(i,t) = v_rest; %set voltage of neuron i to ressting potential if in refractory period
            elseif (v_it(i,t-1) < v_th) %subthreshold update
                if i> N-num_VTA
                    del_v_i = ((randn/norm_noise_VTA)+ constant_VTA + g_L*(E_l-v_it(i,t-1)) + (g_Ei(i,t-1) + g_Ey(i,t-1))*(E_e - v_it(i,t-1)) + g_Ii(i,t-1)*(E_i - v_kt(i,t-1)))*(dt/C_m);
                else
                    del_v_i = ((randn/norm_noise)+g_L*(E_l-v_it(i,t-1)) + (g_Ei(i,t-1) + g_Ey(i,t-1))*(E_e - v_it(i,t-1)) + g_Ii(i,t-1)*(E_i - v_kt(i,t-1)))*(dt/C_m);
                end
                v_it(i,t) = v_it(i,t-1) + del_v_i; %update membrane potential
            elseif (v_it(i,t-1) >= v_th)
                v_it(i,t) = v_hold; %voltage resets, neuron enter refractory phase
            end
            if v_it(i,t) == v_hold %if neuron spikes
                sc_R(i,t) = 1;
                del_R_it = (1/dt-R_it(i,t-1))*(dt/tau_w);
                R_it(i,t) = R_it(i,t-1) + del_R_it; %update firing rate
                del_s_j = -(s_it(i,t-1)*dt/tau_se) + rho*(1-s_it(i,t-1));
                s_it(i,t) = s_it(i,t-1) + del_s_j;
                t_ref_i(i,t:t+t_refractory) = 1;
            else %if neuron does not spike
                del_R_it = -R_it(i,t-1)*(dt/tau_w);
                R_it(i,t) = R_it(i,t-1) + del_R_it;
                del_s_j = -s_it(i,t-1)*(dt/tau_se);
                s_it(i,t) = s_it(i,t-1) + del_s_j;
            end
            
            %% inhibitory dynamics
            if t_ref_k(i,t) == 1 %refractory period
                v_kt(i,t) = v_rest; %set voltage of neuron i to ressting potential if in refractory period
            elseif (v_kt(i,t-1) < v_th_i) %subthreshold update
                del_v_k = ((randn/norm_noise)+ g_L*(E_l-v_kt(i,t-1)) + (g_Ek(i,t-1) + (iG/eG)*g_Eyi(i,t-1))*(E_e - v_kt(i,t-1)))*(dt/C_m);
                v_kt(i,t) = v_kt(i,t-1) + del_v_k; %update membrane potential
            elseif (v_kt(i,t-1) >= v_th_i)
                v_kt(i,t) = v_hold; %voltage resets, neuron enter refractory phase
            end
            if v_kt(i,t) == v_hold %if neuron spikes
                del_R_kt = (1/dt-R_kt(i,t-1))*(dt/tau_w);
                R_kt(i,t) = R_kt(i,t-1) + del_R_kt; %update firing rate
                del_s_k = -(s_kt(i,t-1)*dt/tau_si) + rho*(1-s_kt(i,t-1));
                s_kt(i,t) = s_kt(i,t-1) + del_s_k;
                t_ref_k(i,t:t+t_refractory) = 1;
            else %if neuron does not spike
                del_R_kt = -R_kt(i,t-1)*(dt/tau_w);
                R_kt(i,t) = R_kt(i,t-1) + del_R_kt;
                del_s_k = -s_kt(i,t-1)*(dt/tau_si);
                s_kt(i,t) = s_kt(i,t-1) + del_s_k;
            end
            
            %% post-synaptic loop
            for j = 1:N
                if (rec_identity(i,j) ~= 0 || ff_identity(i,j) ~= 0 || hebb_identity(i,j) ~= 0) && (t > D) && (l < num_trials + 1) && (t< t_total/dt) %only looks for traces at synapses with allowed connections and times
                    %% Hebbian and Trace dynamics
                    if trace_refractory(i,j) == 0
                        if rec_identity(i,j) ~= 0 %selects recurrent synapses
                            if R_it(i,t-D) > .01 && R_it(j,t-1) > .01
                                a_temp = 0.1;
                                b_temp = .01;
                                c_temp = 0;
                                H_d = eta_d*phi_4(R_it(i,t-D),a_temp,b_temp,c_temp)*phi_4(R_it(j,t-dt),a_temp,b_temp,c_temp)/T_max_d; %Hebbian learning term for depression
                                H_p = eta_p*phi_4(R_it(i,t-D),a_temp,b_temp,c_temp)*phi_4(R_it(j,t-dt),a_temp,b_temp,c_temp)/T_max_p; %Hebbian learning term for potentiation
                            else
                                H_d = 0; %Hebbian learning term for depression
                                H_p = 0; %Hebbian learning term for potentiation
                            end
                            albert = 100;
                            del_T_ijp = (-T_ijp(i,j) + (H_p/(1+albert*dopa_rec2(t-1)))*(T_max_p - T_ijp(i,j)))*(dt/tau_p); %change in LTP eligibility trace
                            del_T_ijd = (-T_ijd(i,j) + (H_d/(1+albert*dopa_rec2(t-1)))*(T_max_d - T_ijd(i,j)))*(dt/tau_d); %change in LTD eligibility trace
                            
                            T_ijp(i,j) = T_ijp(i,j) + del_T_ijp - hepp*dopa_rec2(t-1); %update LTP eligibility trace
                            T_ijd(i,j) = T_ijd(i,j) + del_T_ijd - hepp*dopa_rec2(t-1); %update LTD eligibility trace
%                             if t == 300 || t == 1000 || t == 1700
%                                 T_ijp(i,j) = 0;
%                                 T_ijd(i,j) = 0;
%                                 trace_refractory(i,j) = 50/dt;
%                             end
                            if T_ijp(i,j) < 0
                                T_ijp(i,j) = 0;
                            end
                            if T_ijd(i,j) < 0
                                T_ijd(i,j) = 0;
                            end
                        elseif fin_identity(i,j) ~= 0 %selects ff synapses
                            if R_it(i,t-D) > .002 && R_yt(j,t-1) > .002
%                                 disp('yes')
                                H_d = eta_d1*R_it(i,t-D)*R_yt(j,t-dt)/T_max_d1; %Hebbian learning term for depression
                                H_p = eta_p1*R_it(i,t-D)*R_yt(j,t-dt)/T_max_p1; %Hebbian learning term for potentiation
                            else
                                H_d = 0; %Hebbian learning term for depression
                                H_p = 0; %Hebbian learning term for potentiation
                            end
                            del_T_ijp = (-T_ijp(i,j) + H_p*(T_max_p1 - T_ijp(i,j)))*(dt/tau_p1); %change in LTP eligibility trace
                            del_T_ijd = (-T_ijd(i,j) + H_d*(T_max_d1 - T_ijd(i,j)))*(dt/tau_d1); %change in LTD eligibility trace
                            T_ijp(i,j) = T_ijp(i,j) + del_T_ijp; %update LTP eligibility trace
                            T_ijd(i,j) = T_ijd(i,j) + del_T_ijd; %update LTD eligibility trace
                        end
                    else
                        T_ijp(i,j) = 0;
                        T_ijd(i,j) = 0;
                        trace_refractory(i,j) = trace_refractory(i,j) - 1/dt;
                    end
                    
                    %% Learning dynamics at time of reward
                    if l>1 && rec_identity(i,j) ~= 0 %selects recurrent synapses
                        del_W_ji(i,j) = del_W_ji(i,j) + eta_rec*(T_ijp(i,j)*(dopa_rec(t-1)/num_VTA)-T_ijd(i,j)*(dopa_rec(t-1)/num_VTA))*dt;
%                         disp('enter')
                        
                    elseif l>1 && fin_identity(i,j)~= 0 %selects ff synapses
                        del_W_in(i,j) = del_W_in(i,j) + eta_fin(i,j)*(T_ijp(i,j)*(dopa(t-1)/num_VTA)-T_ijd(i,j)*(dopa_ff(t-1)/num_VTA))*dt; %change in ff weights
                    elseif l>1 && hebb_identity(i,j) ~=0
                        del_M_ki(i,j) = del_M_ki(i,j) + eta_hebb(i)*((dopa(t-1) - dopa_ff(t-1))/num_VTA)*R_kt(j,t);% -decay_on*M_ki(i,j)*lam_decay_2;
                    elseif l>1 && ff_identity(i,j) ~=0
                        del_W_ji(i,j) = del_W_ji(i,j) + eta_ff*(R_it(i,t)*R_it(j,t) -W_ji(i,j)*lam_decay_2);
%                         del_M_ki(i,j) = del_M_ki(i,j) + eta_hebb(i)*(sum(temp)/num_VTA -4)*R_kt(j,t);% -decay_on*M_ki(i,j)*lam_decay_2;
%                         R_kt(i,t)*(R_it(j,t)-0)
%                         if del_M_ki(j,i) > 0
% %                             disp(del_M_ki(j,i))
%                             disp(i)
%                         else
%                         end
%                         disp('enter')
                    else
                        del_W_ji(i,j) = del_W_ji(i,j) + 0;
                    end
                   
                end
            end
        end
        %% updating conductances
        g_Ey(:,t) = W_in(:,:)*s_yt(:,t); %input conductance
        g_Eyi(:,t) = W_inin(:,:)*s_yt(:,t); %input conductance to inhibitory cells
        g_Ei(:,t) = W_ji(:,:)*s_it(:,t); %recurrent excitatory conductance
        g_Ii(:,t) = M_ki(:,:)*s_kt(:,t); %I to E conductance
        g_Ek(:,t) = P_ik(:,:)*s_it(:,t); %E to I conductance
        
        
        temp_t(:,t) = R_it(N-num_VTA+1:N,t-1).*1000;
        temp = temp_t(:,t);
        
%         a1 = 160;%65;
%         b1 = 12;%8;
%         c1 = 140;%5;
%         d1 = 0;%0.3;
        

%         a1 = 15;
%         b1 = 5.6;
%         c1 = 0.9;
%         d1 = 25;
%         e1 = -.9;
%         f1 = .73;
%         g1 = 4.5;
%         lam = 1.2;

        a1 = 20;
        b1 = 5;
        c1 = 1.3;
        d1 = 19.2;
        e1 = -.3;
        f1 = 1.1;
        g1 = 7.1;
        lam = 2.2;
        
        a_rec = 20;
        b_rec = 20;
        c_rec = 0.4;
        
        
        dopa(t) = LTP_D_func(sum(temp)/num_VTA,a1,b1,c1,d1,e1,f1,g1,lam,1);
        dopa_temp = LTP_D_func(sum(temp)/num_VTA,a_rec,b_rec,c_rec,d1,e1,f1,g1,lam,1);
        dopa_rec(t) = dopa_temp;
        hepa = 1;
        dopa_rec2(t) = dopa_rec2(t-1) + (-dopa_rec2(t-1) + hepa*dopa_temp)/tau_dopa;
        d_der(t) = dopa_rec2(t)-dopa_rec2(t-1);
        dopa_ff(t) = LTP_D_func(sum(temp)/num_VTA,a1,b1,c1,d1,e1,f1,g1,lam,0);
        temp_tracker(t) = sum(temp)/num_VTA;
        T_maxie_d(t) = T_maxie_d(t-1) + (T_max_d-T_maxie_d(t-1) - hepc*dopa_rec2(t))/tau_max;
        if T_maxie_d(t) <=0
            T_maxie_d(t) = 0;
        end
        T_maxie_p(t) = T_maxie_p(t-1) + (T_max_p-T_maxie_p(t-1) - hepc*dopa_rec2(t))/tau_max;
        if T_maxie_p(t) <=0
            T_maxie_p(t) = 0;
        end
        
        
        
        if select_rec == 1
%             T_pt(t) = mean(T_ijp(2*npp+1:3*npp,2*npp+1:3*npp),'all')*100000; %recurrent
%             T_dt(t) = mean(T_ijd(2*npp+1:3*npp,2*npp+1:3*npp),'all')*100000; %recurrent
            T_pt(t) = mean(T_ijp(1:npp,1:npp),'all')*100000; %recurrent
            T_dt(t) = mean(T_ijd(1:npp,1:npp),'all')*100000; %recurrent
            num_thresh = num_thresh_rec;
        else
            T_pt(t) = mean(T_ijp(N-num_VTA+1:N,2*npp+1:3*npp),'all')*5000000; %ff
            T_dt(t) = mean(T_ijd(N-num_VTA+1:N,2*npp+1:3*npp),'all')*5000000; %ff

            num_thresh = num_thresh_ff;
        end
        
    end
    
    if select_rec == 1
        dopa_plot1 = dopa_rec;
        dopa_plot = dopa_rec;
    else
        dopa_plot1 = dopa;
        dopa_plot = dopa_ff;
    end
    %% update weights/plotting
    W_ji = W_ji + del_W_ji;
    W_ji = W_ji.*(W_ji > 0) + 0.*(W_ji <= 0); %cuts off weights <0
    
    M_ki = M_ki + del_M_ki;
    M_ki = M_ki.*(M_ki > 0) + 0.*(M_ki <= 0); %cuts off weights <0

    W_in = W_in + del_W_in;
    W_in = W_in.*(W_in > 0) + 0.*(W_in <= 0); %cuts off weights <0
    
%     figure
%     imagesc(del_M_ki)
    
    
    sc_R_it(:,:,l) = sc_R; %spiking for plotting
    R_it_l(:,:,l) = R_it;
    for o = 1:pop+1
        temp = (o-1)*npp;
        plot_R_it(o,:,l) = mean(R_it(temp+1:temp+npp,:),1); %population average firing rates for plotting
    end
    %rescaling firing rates
    R_it = R_it*1000;
    R_kt = R_kt*1000;
    R_yt = R_yt*1000;
    
    
    mean_VTA_fr = mean(mean(sc_R(N-num_VTA+1:N,:)))*1000;
    %tracking mean ff and recurrent weights
    mean_temp1 = mean(W_ji((2*npp+1):3*npp,npp+1:2*npp)*(npp^2),'all');
    
    rec_vect(l) = mean_temp1;
    mean_temp1 = mean(M_ki(N-num_VTA+1:N,npp+1:(2*npp))*(npp^2),'all');
    m_vect(l) = mean_temp1;
    mean_temp1 = mean(W_in(N-num_VTA+1:N,1:npp)*(npp^2),'all');
    ff_vect(l) = mean_temp1;
    
    if num_columns == 2
        mean_temp2 = mean(W_ji(1:npp,1:npp)*(npp^2),'all');
        rec_vect1(l) = mean_temp2;
        mean_temp2 = mean(M_ki(N-num_VTA+1:N,(3*npp+1):(4*npp))*(npp^2),'all');
        m_vect1(l) = mean_temp2;
        mean_temp2 = mean(W_in(N-num_VTA+1:N,(npp+1):2*npp)*(npp^2),'all');
        ff_vect1(l) = mean_temp2;
    end
    
    for t = 0:length(1:ROC_dt:t_total-2*ROC_dt)
        if l > 1
            hist_vect(:,t+1,l) = sum(sc_R(N-num_VTA+1:N,(t*ROC_dt+1):(t*ROC_dt + ROC_dt)),2);
        elseif l == 1
            hist_vect1(:,t+1) = sum(sc_R(N-num_VTA+1:N,(t*ROC_dt+1):(t*ROC_dt + ROC_dt)),2);
        end
    end
%     if l < 25
%         running_mean_R_it = mean(R_it_l(:,:,1:l),3);
%     else
%         running_mean_R_it = mean(R_it_l(:,:,l-25:l),3);
%     end
    
    sprintf('Trial %d complete',l)
    %plotting
    if l == 1
        old_R_it = plot_R_it(:,:,l);
        plot_func(T,delta,l,num_columns,0:dt:t_total,t_total,dt,npp,N,num_VTA,R_it,R_kt,T_pt,T_dt,W_ji,'Before Learning',t_stim,dopa_plot1,dopa_plot)
    elseif l == num_trials
        tit = sprintf('Final Learning Trial %d (Full Stim)',num_trials);
        plot_func(T,delta,l,num_columns,0:dt:t_total,t_total,dt,npp,N,num_VTA,R_it,R_kt,T_pt,T_dt,W_ji,tit,t_stim,dopa_plot1,dopa_plot)
    elseif l == num_trials+1
        new_R_it = plot_R_it(:,:,l);
        plot_func(T,delta,l,num_columns,0:dt:t_total,t_total,dt,npp,N,num_VTA,R_it,R_kt,T_pt,T_dt,W_ji,'After Learning (one stim)',t_stim,dopa_plot1,dopa_plot)
    else
        tit = sprintf('During Learning Trial %d',l);
        plot_func(T,delta,l,num_columns,0:dt:t_total,t_total,dt,npp,N,num_VTA,R_it,R_kt,T_pt,T_dt,W_ji,tit,t_stim,dopa_plot1,dopa_plot)
    end
    drawnow
end
%%
figure('Position',[1200 300 500 500])
subplot(3,1,1)
plot(ff_vect(1,:),'x');
if num_columns == 2
    hold on
    plot(ff_vect1(1,:),'x');
end
xlabel('Trial Number');
ylabel('Mean FF Weight');
subplot(3,1,2)
plot(rec_vect(1,:),'x');
if num_columns == 2
    hold on
    plot(rec_vect1(1,:),'x');
end
xlabel('Trial Number');
ylabel('Mean rec Weight');
subplot(3,1,3)
plot(m_vect(1,:),'x');
if num_columns == 2
    hold on
    plot(m_vect1(1,:),'x');
end
xlabel('Trial Number');
ylabel('Mean inh Weight');
%%

lengthie = 15;
minus_trial = 82; 


mean_VTA_plot = mean(R_it(N-num_VTA+1:N,:));
mean_VTA_plot_minus = mean(1000*R_it_l(N-num_VTA+1:N,:,minus_trial),1);
% mean_R_it_after = mean(R_it_l(:,:,(num_trials-lengthie + 1):num_trials),3);
mean_R_it_after = mean(R_it_l(:,:,num_trials-lengthie+1:num_trials),3);
mean_R_it_after_learning = mean(R_it_l(:,:,3*num_trials/4-lengthie+1:3*num_trials/4),3);
mean_R_it_before = mean(R_it_l(:,:,1:lengthie),3);
% mean_R_it_before = mean(R_it_l(:,:,2*num_trials/4 - lengthie + 1:2*num_trials/4),3);
mean_mean_R_it_after = mean(mean_R_it_after(N-num_VTA+1:N,:));
mean_mean_R_it_after_learning = mean(mean_R_it_after_learning(N-num_VTA+1:N,:));
mean_mean_R_it_before = mean(mean_R_it_before(N-num_VTA+1:N,:));
mean_R_it = mean(R_it_l,3);

%%


figure
plot(dopa)
hold on
plot(dopa_ff)
figure
plot(sum(temp_t,1)/num_VTA)
figure
plot(dopa.*T_pt)
hold on
plot(dopa_ff.*T_dt)

%%

figure('Position',[300 300 1000 1000])
subplot(2,1,1)
LineFormat = struct();
LineFormat.Color = 'yellow';
plotSpikeRaster(sc_R_it(:,:,num_trials) == 1,'LineFormat',LineFormat);
set(gca,'color',[0 0 0])
ylabel('Neuron number')
xlabel('Time(ms)')
title('spike raster  (200-300 are VTA neurons)')
subplot(2,1,2)
plot(1:dt:t_total,mean_VTA_plot)
ylabel('Firing rate')
xlabel('Time(ms)')
title('Mean VTA neuron firing rate, last trial')
xlim([0.01 t_total]);
ylim([0.01 max(mean_VTA_plot) + 5]);

figure('Position',[300 300 1000 1000])
subplot(2,1,1)
LineFormat = struct();
LineFormat.Color = 'yellow';
plotSpikeRaster(sc_R_it(:,:,minus_trial) == 1,'LineFormat',LineFormat);
set(gca,'color',[0 0 0])
ylabel('Neuron number')
xlabel('Time(ms)')
title('spike raster  (200-300 are VTA neurons)')
subplot(2,1,2)
plot(1:dt:t_total,mean_VTA_plot_minus)
ylabel('Firing rate')
xlabel('Time(ms)')
title('Mean VTA neuron firing rate, last trial')
xlim([0.01 t_total]);
ylim([0.01 max(mean_VTA_plot_minus) + 5]);


%%

figure('Position',[50 300 1800 1000])
subplot(1,3,1)
hold on
for i = (N-num_VTA+1):N
    plot3(1:dt:t_total,i*ones(size(1:dt:t_total)),mean_R_it_before(i,:)*1000)
end
view(0,85)
xlim([0 t_total])
xlabel('Time(ms)')
ylabel('Neuron number')
zlabel('Firing rate (Hz)')
title(['VTA neurons, averaged for ' num2str(lengthie) ' trials before learning'])

colororder(white(100)-1)

subplot(1,3,2)
hold on
for i = (N-num_VTA+1):N
    plot3(1:dt:t_total,i*ones(size(1:dt:t_total)),mean_R_it_after_learning(i,:)*1000)
end
view(0,85)
xlim([0 t_total])
xlabel('Time(ms)')
ylabel('Neuron number')
zlabel('Firing rate (Hz)')
title(['VTA neurons, averaged for ' num2str(lengthie) ' trials after learning (blocking)'])

colororder(white(100)-1)

subplot(1,3,3)
hold on
for i = (N-num_VTA+1):N
    plot3(1:dt:t_total,i*ones(size(1:dt:t_total)),mean_R_it_after(i,:)*1000)
end
view(0,85)
xlim([0 t_total])
xlabel('Time(ms)')
ylabel('Neuron number')
zlabel('Firing rate (Hz)')
title(['VTA neurons, averaged for ' num2str(lengthie) ' trials after unblocking'])

colororder(white(100)-1)


%%
% figure
% title('VTA neurons')
% subplot(1,3,1)
% imagesc(mean_R_it_before(N-num_VTA+1:N,:))
% caxis([0 max(max(mean_R_it_after_learning(N-num_VTA+1:N,:)))])
% xlabel('time')
% ylabel('neuron number')
% title('VTA neurons before learning (mean over 20 trials)')
% subplot(1,3,2)
% imagesc(mean_R_it_after_learning(N-num_VTA+1:N,:))
% caxis([0 max(max(mean_R_it_after_learning(N-num_VTA+1:N,:)))])
% xlabel('time')
% ylabel('neuron number')
% title('VTA neurons after learning (mean over 20 trials)')
% subplot(1,3,3)
% imagesc(mean_R_it_after(N-num_VTA+1:N,:))
% caxis([0 max(max(mean_R_it_after_learning(N-num_VTA+1:N,:)))])
% xlabel('time')
% ylabel('neuron number')
% title('VTA neurons after CS removal (mean over 20 trials)')
% % disp(mean_VTA_fr)


figure
subplot(3,1,1)
plot(1:dt:t_total,mean_mean_R_it_before*1000)
title(['mean over all neurons and'  num2str(lengthie) 'trials before learning'])
xlabel('time')
ylabel('Firing rate')
subplot(3,1,2)
plot(1:dt:t_total,mean_mean_R_it_after_learning*1000)
title(['mean over all neurons and last' num2str(lengthie) 'trials of learning'])
xlabel('time')
ylabel('Firing rate')
subplot(3,1,3)
plot(1:dt:t_total,mean_mean_R_it_after*1000)
title(['mean over all neurons and' num2str(lengthie) 'trials of no CS'])
xlabel('time')
ylabel('Firing rate')




%%


baseline = hist_vect1(:,1:lengthie);

figure('Position',[50 300 1800 1000])
for k = 1:3
    for t = 0:length(1:ROC_dt:t_total-2*ROC_dt)
        if k == 1
            r1_vect = reshape(hist_vect(:,t+1,(2*num_trials/4 -lengthie+ 1):2*num_trials/4),[length(N-num_VTA+1:N) lengthie]);
            title_1 = ['AUC for first ' num2str(lengthie) ' trials (before learning)'];
        elseif k == 2
            r1_vect = reshape(hist_vect(:,t+1,(3*num_trials/4 -lengthie+ 1):3*num_trials/4 ),[length(N-num_VTA+1:N) lengthie]);
            title_1 = ['AUC for ' num2str(lengthie) ' trials after first learning (blocking)'];
        elseif k == 3
            r1_vect = reshape(hist_vect(:,t+1,(num_trials-lengthie + 1):num_trials),[length(N-num_VTA+1:N) lengthie]);
            title_1 = ['AUC for ' num2str(lengthie) ' trials after learning via unblocking'];
        end
        for i = 1:length(N-num_VTA+1:N)
            b_line = baseline(i,:)';
            r1_v = r1_vect(i,:)';
            bins = min([b_line r1_v]):max([b_line r1_v])+1;
            prob_pos = zeros(1,length(bins));
            prob_neg = zeros(1,length(bins));
            for j = 1:length(bins)
                thresh = bins(j);
                prob_pos(j) = sum(r1_v>=thresh)/length(r1_v);
                prob_neg(j) = sum(b_line>=thresh)/length(b_line);
            end
            [prob_neg_sorted,idx] = sort(prob_neg);
            prob_pos_sorted = prob_pos(idx);
            
            auc_plot(i,t+1) = trapz(prob_neg_sorted, prob_pos_sorted);
        end
    end
    subplot(1,3,k)
    imagesc(auc_plot)
    caxis([0 1])
    title(title_1)
    xlabel('Time bin # (50ms time bins)')
    ylabel('Neuron #')
    mycolormap = customcolormap([0 0.45 0.5 0.55 1], [1 1 0;0 0 0;0 0 0;0 0 0;0 1 1]);
    colorbar;
    colormap(mycolormap)
end


% mycolormap = customcolormap([0 0.45 0.5 0.55 1], [1 1 0;0 0 0;0 0 0;0 0 0;0 1 1]);
% colorbar;
% colormap(mycolormap)

