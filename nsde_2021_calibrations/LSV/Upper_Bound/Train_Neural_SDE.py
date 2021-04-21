import sys
import os
sys.path.append(os.path.dirname('__file__'))

import torch
import torch.nn as nn
import numpy as np
import math
import time
import argparse

from py_vollib.black_scholes.implied_volatility import implied_volatility 
from py_vollib.black_scholes import black_scholes as price_black_scholes 

torch.manual_seed(901) # fixed for reproducibility

from networks import *
         
class Net_LSV(nn.Module):
    """
    Calibration of LSV model to vanilla prices at different maturities
    """
    def __init__(self, timegrid, strikes_call, device, n_maturities):
        
        super(Net_LSV, self).__init__()
        self.timegrid = timegrid
        self.device = device
        self.strikes_call = strikes_call
        
        # initialise price diffusion neural network (different neural network for each maturity)
        self.net_S_vol =  Net_timegrid(dim=3, nOut=1, n_layers=3, vNetWidth=100, n_maturities=n_maturities, activation_output="softplus")
        
        # initialise vanilla hedging strategy neural networks 
        """
        network for each maturity is used to hedge only options for that maturity, for example network corresponding to final maturity
        is used to simulate hedging strategy (from time 0) for vanilla options at the final maturity
        """
        self.net_CV = Net_timegrid(dim=2, nOut=len(strikes_call), n_layers=2, vNetWidth=20, n_maturities=n_maturities, activation_output="softplus")

        # initialise stochastic volatility drift and diffusion neural networks
        self.V_drift = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_maturities=n_maturities)
        self.V_vol = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_maturities=n_maturities, activation_output="softplus")
        
        # initialise stochastic volatility correlation and initial value parameters
        self.v0 = torch.nn.Parameter(torch.rand(1)*0.1)
        self.rho = torch.nn.Parameter(2*torch.rand(1)-1)
        
        # initialise exotic hedging strategy neural networks 
        """
        network for each maturity is used only to simulate hedging strategy for the timesteps in between previous and current maturity
        so that "n_maturities" networks are used to simulate hedging strategy for exotic option with maturity at the end of considered time horizon
        """
        self.exotic_hedge = Net_timegrid(dim=2, nOut=1, n_layers=2, vNetWidth=20, n_maturities=n_maturities)


    def forward(self, S0, rate, z, MC_samples, ind_T, period_length,n_maturities): 
        """
        pv_h:=price vanilla hedged pv_var_h:=variance of hedged vanilla price
        pe_h:=price exotic hedged; pe_var_h:=variance of hedged exotic price
        pe_u:=price exotic unhedged; pe_var_u:=variance of unhedged exotic price
        cv_vanilla_fwd-stores output of vanilla hedging strategies at each timestep
        """
        ones = torch.ones(MC_samples, 1, device=self.device)
        S_old = ones * S0
        cv_vanilla = torch.zeros(S_old.shape[0], len(self.strikes_call),n_maturities, device=self.device)
        cv_exotic = torch.zeros(S_old.shape[0], 1 , device=self.device)
        pv_h = torch.zeros(len(self.strikes_call), device=self.device)
        pv_var_h = torch.zeros_like(pv_h)
        pe_u = torch.zeros(1, device=self.device)
        pe_h = torch.zeros(1, device=self.device)
        pe_var_h = torch.zeros(1, device=self.device)
        pe_var_u = torch.zeros(1, device=self.device)
        cv_vanilla_fwd = torch.zeros(S_old.shape[0], len(self.strikes_call),n_maturities, device=self.device)
        pv_h_out = torch.Tensor().to(device)  # used to store model vanilla prices for all maturities
        pv_var_h_out = torch.Tensor().to(device) # used to store variances of vanilla options for all maturities
        martingale_test = torch.zeros(1, device=self.device)
        put_atm = torch.zeros(1, device=self.device)
        call_atm = torch.zeros(1, device=self.device)
        V_old = ones * torch.sigmoid(self.v0)*0.5
        rho = torch.tanh(self.rho)
        running_max = S_old # initialisation of running_max for calculation of lookback price
        
        with open("rho_v0.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(rho.item(),V_old[0,0].item()))


        for i in range(1, ind_T+1):
            idx_net = (i-1)//period_length # assume maturities are evenly distributed, i.e. 0, 16, 32, ..., 96
            t = torch.ones_like(S_old) * self.timegrid[i-1]
            h = self.timegrid[i]-self.timegrid[i-1]   
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            zz = torch.randn_like(dW)
            dB = rho * dW + torch.sqrt(1-rho**2)*torch.sqrt(h)*zz
            
            Slog_old=torch.log(S_old)
            price_diffusion = self.net_S_vol.forward_idx(idx_net, torch.cat([t,Slog_old,V_old],1))
            
            # Evaluate vanilla hedging strategies at particular timestep
            for mat in range(n_maturities):
                if mat>=idx_net:
                    cv_vanilla_fwd[:,:,mat] = self.net_CV.forward_idx(mat, torch.cat([t,Slog_old.detach()],1))
           
            exotic_hedge = self.exotic_hedge.forward_idx(idx_net, torch.cat([t,Slog_old.detach()],1))
            V_new = V_old + self.V_drift.forward_idx(idx_net,V_old).reshape(MC_samples,1)*h + self.V_vol.forward_idx(idx_net,V_old).reshape(MC_samples,1)*dB
            price_diffusion_const = price_diffusion.detach()

            drift =  (rate-0.5*(price_diffusion**2).reshape(MC_samples,1))
            diff = price_diffusion.reshape(MC_samples,1) * dW

            # Drift normalisations in tamed Euler scheme have gradient detached
            drift_c = (1+torch.abs(drift.detach())*torch.sqrt(h))
            diff_c = (1+torch.abs(price_diffusion_const.reshape(MC_samples,1))*torch.sqrt(h))
           
            # Tamed Euler step
            Slog_new = Slog_old + drift*h/drift_c + diff/diff_c 
            S_new=torch.exp(Slog_new)
            
            # Discounted price change between timesteps; used for control variate training       
            dS = torch.exp(-rate*self.timegrid[i])*S_new - torch.exp(-rate*self.timegrid[i-1])*S_old
           
            # Evaluate stoch. integrals (w.r.t. discounted price process) corresponding to control variates
            for mat in range(n_maturities):
                 if mat>=idx_net:
                     cv_vanilla[:,:,mat] += cv_vanilla_fwd[:,:,mat] * dS.detach().repeat(1,len(self.strikes_call))  
            cv_exotic +=  exotic_hedge.reshape(MC_samples,1) * dS.detach()
            
            # Evaluate exotic payoff 
            running_max = torch.max(running_max, S_new) 
            
            # Update values of asset prc processes for next Tamed Euler step
            S_old = S_new
            V_old = torch.clamp(V_new,0)
                  
            # Evaluate vanilla option prices and variances of price estimate if timestep corresponds to maturity  
            if int(i) in maturities: 
               dis_fact = torch.exp(-rate*self.timegrid[(idx_net + 1)*period_length])
               for idx, strike in enumerate(self.strikes_call):
                  pv_h_temp = dis_fact*torch.clamp(S_old-strike,0).squeeze(1).detach()-cv_vanilla[:,idx,idx_net]
                  pv_h_detach_cv = dis_fact*torch.clamp(S_old-strike,0).squeeze(1)-cv_vanilla[:,idx,idx_net].detach()
                  pv_h[idx] = pv_h_detach_cv.mean() # gradient of cv not applied when evaluating price of vanilla options 
                  pv_var_h[idx] = pv_h_temp.var() # gradient of price not applied when evaluating variance of vanilla options
               pv_h_out = torch.cat([pv_h_out,pv_h.T],0) 
               pv_var_h_out = torch.cat([pv_var_h_out,pv_var_h.T],0)
                
            # Evaluate exotic option price and variance of exotic price estimates at final maturity (and martingale test)
            if int(i)==n_maturities*period_length:  
                  martingale_test = dis_fact*S_old.detach()
                  martingale_test = martingale_test.mean()/S0
                  put_atm = dis_fact*torch.clamp(S0-S_old.detach(),0).squeeze(1)
                  put_atm = put_atm.mean().detach()
                  call_atm = dis_fact*torch.clamp(S_old.detach()-S0,0).squeeze(1)
                  call_atm = call_atm.mean()
                  put_call_parity_error = call_atm-put_atm - S0 + dis_fact*S0
                  exotic_payoff = running_max - S_old
                  exotic_payoff_detach = exotic_payoff.detach() # detach dependence on price for estimating variance of exotic
                  pe_h_temp = dis_fact*exotic_payoff_detach.squeeze(1).reshape(MC_samples,1)-cv_exotic.reshape(MC_samples,1)
                  pe_u = dis_fact*exotic_payoff.squeeze(1)
                  pe_var_h = pe_h_temp.var()
                  pe_var_u = pe_u.var()
                  pe_u = pe_u.mean() 
                  pe_h = dis_fact*exotic_payoff.squeeze(1).reshape(MC_samples,1)-cv_exotic.detach().reshape(MC_samples,1)
                  pe_h = pe_h.mean()

        return pv_h_out, pv_var_h_out, pe_u, pe_h, pe_var_u, pe_var_h, martingale_test, put_atm, call_atm, put_call_parity_error
        
def save_file(to_save, filename):
    with open(filename, "a") as f:
        f.write("{}\n".format(to_save.item()))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)


def train_nsde(args):
    
    loss_fn = nn.MSELoss() 
    maturities = range(16, 97, 16)
    n_maturities = len(maturities)
    model = Net_LSV(timegrid=timegrid, strikes_call=strikes_call, device=device, n_maturities=n_maturities)
    model = model.to(device)
    itercount = 0
    itercount_2 = 0
    itercount_3 = 0
    itercount_4 = 0
    n_epochs = 200
    batch_size = 50000
    batch_size_hedge = 10000
    T=n_maturities*16
    base_learing_rate = 0.0005

    params_SDE = list(model.net_S_vol.parameters()) + [model.rho, model.v0] + list(model.V_drift.parameters())+list(model.V_vol.parameters())
    params_CV = list(model.net_CV.parameters())
    params_hedge = list(model.exotic_hedge.parameters())

    optimizer_SDE = torch.optim.Adam(params_CV,lr=100*base_learing_rate)
    optimizer_SDE_2 = torch.optim.Adam(params_SDE,lr=base_learing_rate)
    optimizer_SDE_4 = torch.optim.Adam(params_hedge,lr=100*base_learing_rate)

    loss_val_best = 10
    best_IV_error = 10
    best_hedge_error = 10
    exotic_hedge_error = torch.tensor(10, device=device).float()
    target_mat_T = torch.tensor(data, device=device).float()
       
    MC_samples_gen=1000000 
    MC_samples_var=400000

    batch_steps= 96
    LAMBDA = args.LAMBDA
    LAMBDA_2 = args.LAMBDA
    c = args.c
    c_2 = args.c
        
    # Number of paths used for validation
    MC_samples_gen = args.MC_samples_gen
    MC_samples_var = args.MC_samples_var
       
    for epoch in range(n_epochs):
        # evaluate model at initialisation and save model statistics at initialisation
        if epoch==0:

            # calculate target IVs
            K = strikes_call
            iv_target_1 = np.zeros_like(K)
            iv_target_1_out = torch.Tensor().to(device)
            with torch.no_grad():
                try:
                   for idxt, t in enumerate(maturities):
                        van_target_1 = target_mat_T[idxt,::].cpu().numpy() 
                        for idxx, ( target, k) in enumerate(zip(van_target_1, K)):
                            iv_target_1[idxx] = implied_volatility(target,  S=S0, K=k, r=0.0, t=t/(2*96), flag="c") 
                        iv_target_1_out = torch.cat([iv_target_1_out,torch.from_numpy(iv_target_1).float().to(device).T],0)  
                except:
                   pass
            print('iv_target_1',iv_target_1_out)
                
            # calculate vega weights 
            vega_target_1 = np.ones_like(K)
            vega_out = torch.Tensor().to(device)
            constant_shift=0.01
                
            with torch.no_grad():
                       
                try:
                    for idxt, t in enumerate(maturities):
                        van_target_1 = target_mat_T[idxt,::].cpu().numpy() 
                        for idxx, (target, k) in enumerate(zip(van_target_1, K)):
                            iv_target_11 = implied_volatility(target, S=S0, K=k, r=0.0, t=t/(2*96), flag="c") 
                            iv_1 = iv_target_11+constant_shift
                            vega_target_1[idxx] = constant_shift/(price_black_scholes(flag='c',S=S0,K=k,t=t/(2*96),r=0.0,sigma=iv_1)-target)
                        vega_out = torch.cat([vega_out,torch.from_numpy(vega_target_1).float().to(device).T],0)                    
                except:
                    pass
            vega_target_1=torch.tensor(vega_target_1).to(device)
            print('vega_out', vega_out)
            vega=torch.reshape(vega_out,(6,21))
            #normalise the weights by the number of options 6x21=126
            vega=(21*6)/torch.sum(vega)*vega
            print('vegas of target options:',vega)
                
            with torch.no_grad():
                pv_h, _,exotic_prc_no_cv,exotic_prc_cv,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z, MC_samples_gen, T, period_length=16,n_maturities=n_maturities)
                _, var_pv_h_val,_,_,exotic_prc_var_no_cv_val, exotic_prc_var_cv_val,_,_,_,_ = model(S0, rate, z_var,MC_samples_var, T, period_length=16,n_maturities=n_maturities)
                 
                print("pred:",pv_h)
                print("target:", target_mat_T )
            pred=torch.reshape(pv_h,(6,21))
            loss_val = torch.sqrt(loss_fn(pred,target_mat_T))
            loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
            loss_vega = torch.sqrt(loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega)))
            print('validation {}, loss={:.10f}'.format(itercount, loss_val.item()))
            print('validation_rel {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
            print('validation_vega {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
            strloss='loss_'+str(T)+'.txt'
            strloss_rel='loss_rel'+str(T)+'.txt'
            strloss_vega='loss_vega'+str(T)+'.txt'
            strloss_martingale='martingale_test_'+str(T)+'.txt'
            strloss_call_atm='call_atm_'+str(T)+'.txt'
            strloss_put_atm='put_atm_'+str(T)+'.txt'
            strloss_put_call_parity_error = 'put_call_parity_error'+str(T)+'.txt'
            strvan_1_var = 'variance_van_1_'+str(T)+'.txt'
            strexotic_var_cv = 'variance_exotic_cv'+str(T)+'.txt'
            strexotic_var_no_cv = 'variance_exotic_no_cv'+str(T)+'.txt'
            strexotic_prc_cv = 'prc_exotic_cv'+str(T)+'.txt'
            strexotic_prc_no_cv = 'prc_exotic_no_cv'+str(T)+'.txt'
            print('Variance_V1:', torch.sum(var_pv_h_val))
            print('Variance of exotic option with cv:', exotic_prc_var_cv_val)
            print('Variance of exotic option without cv:', exotic_prc_var_no_cv_val)
            print('Price of exotic option with cv:', exotic_prc_cv)
            print('Price of exotic option with cv:', exotic_prc_no_cv)
            save_file(loss_val.cpu(), strloss)
            save_file(loss_val_rel.cpu(), strloss_rel)
            save_file(loss_vega.cpu(), strloss_vega)
            save_file(torch.sum(var_pv_h_val).cpu(), strvan_1_var)
            save_file(martingale_test.cpu(), strloss_martingale) 
            save_file(call_atm.cpu(), strloss_call_atm)
            save_file(put_atm.cpu(), strloss_put_atm) 
            save_file(put_call_parity_error.cpu(), strloss_put_call_parity_error) 
            save_file(exotic_prc_var_cv_val.cpu(), strexotic_var_cv)
            save_file(exotic_prc_cv.cpu(), strexotic_prc_cv)
            save_file(exotic_prc_var_no_cv_val.cpu(), strexotic_var_no_cv)
            save_file(exotic_prc_no_cv.cpu(), strexotic_prc_no_cv)
            print('Martingale test at initialisation:', martingale_test)
            print('Call price at initialisation:', call_atm)
            print('Put price at initialisation:', put_atm)
            print('Put call parity error at initialisation:', put_call_parity_error)
                        
        print('Epoch, batch size:', epoch, batch_size)
        for i in range(0,20*batch_size, batch_size):
            # simulate paths over entire training horizon 

            optimizer_SDE.zero_grad()
            optimizer_SDE_2.zero_grad()
            optimizer_SDE_4.zero_grad()
                
            # Start training CV networks from 1st epoch, at initial (0th) epoch SDE networks are trained  
            if epoch%10==1: 
                model.net_CV.unfreeze()
                model.net_S_vol.freeze()
                model.V_drift.freeze()
                model.V_vol.freeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size_hedge, batch_steps, device=device)
                _, var_pv_h, _,_,_, _, _, _, _, _ = model(S0, rate, batch_x1, batch_size_hedge,T, period_length=16,n_maturities=n_maturities)
                time_forward = time.time() - init_time
                sum_variances = torch.sum(var_pv_h)
                loss=sum_variances
                init_time = time.time()
                itercount += 1
                loss.backward()
                nn.utils.clip_grad_norm_(params_SDE, 1)
                time_backward = time.time() - init_time
                if (itercount % 20 == 0):
                    print('training statistics')
                    print('iteration {}, sum_variance_van={:.4f}, variance_year={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount, sum_variances,var_pv_h[0].item(), time_forward, time_backward))
                optimizer_SDE.step()
                               
            if epoch%10!=1 and epoch%10!=2: 
                model.net_S_vol.unfreeze()
                model.V_drift.unfreeze()
                model.V_vol.unfreeze()
                model.net_CV.freeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size, batch_steps, device=device)
                pv_h, _, _,exotic_prc_cv,_, _, _, put_atm, call_atm, put_call_parity_error = model(S0, rate, batch_x1, batch_size,T, period_length=16,n_maturities=n_maturities)
                time_forward = time.time() - init_time
                pred = torch.reshape(pv_h,(6,21))
                MSE = loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega))
                loss= -exotic_prc_cv + LAMBDA * MSE + c_2/2 * MSE**2  # we use exotic price without control variate in optimising payoff bound 
                init_time = time.time()
                itercount_2 +=1
                loss.backward()
                if itercount_2%20==0:
                    LAMBDA += c_2*MSE.detach() if LAMBDA<1e6 else LAMBDA
                    c_2 = 2*c_2 if c_2<1e10 else c_2
                time_backward = time.time() - init_time
                nn.utils.clip_grad_norm_(params_SDE, 1)
                if (itercount_2 % 20 == 0):
                    print('training statistics')
                    print('iteration {}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount, time_forward, time_backward))
                optimizer_SDE_2.step()
                    
            if epoch%10==2: 
                model.net_S_vol.freeze()
                model.V_drift.freeze()
                model.V_vol.freeze()
                model.net_CV.freeze()
                model.exotic_hedge.unfreeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size_hedge, batch_steps, device=device)
                _, _, _,_, _,exotic_prc_var_cv,_, _, _, _ = model(S0, rate, batch_x1, batch_size_hedge,T, period_length=16,n_maturities=n_maturities)
                time_forward = time.time() - init_time
                loss= exotic_prc_var_cv
                init_time = t
                itercount_4 +=1
                loss.backward()
                time_backward = time.time() - init_time
                if (itercount_4 % 20 == 0):
                        print('training exotic control variate')
                optimizer_SDE_4.step()                                    
         
        with torch.no_grad():
                pv_h, _,exotic_prc_no_cv, exotic_prc_cv,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z, MC_samples_gen, T, period_length=16,n_maturities=n_maturities)
                _, var_pv_h_val,_,_, exotic_prc_var_no_cv_val,exotic_prc_var_cv_val, _,_,_,_ = model(S0, rate, z_var,  MC_samples_var, T, period_length=16,n_maturities=n_maturities)
        print("pred_1:",pv_h)
        print("target:", target_mat_T )
        pred=torch.reshape(pv_h,(6,21))
        print('pred', pred)
        loss_val = torch.sqrt(loss_fn(pred,target_mat_T))
        loss_vega = torch.sqrt(loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega)))
        loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
        print('validation {}, loss={:.10f}'.format(itercount, loss_val.item()))
        print('validation_rel {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
        print('validation_vega {}, loss={:.10f}'.format(itercount, loss_vega.item())) 
        exotic_hedge_error_change =torch.abs(1-exotic_hedge_error/exotic_prc_var_cv_val)  
        exotic_hedge_error = exotic_prc_var_cv_val
        strloss='loss_'+str(T)+'.txt'
        strvan_1_var = 'variance_van_1_'+str(T)+'.txt'
        strexotic_var_cv = 'variance_exotic_cv'+str(T)+'.txt'
        strexotic_var_no_cv = 'variance_exotic_no_cv'+str(T)+'.txt'
        strexotic_prc_cv = 'prc_exotic_cv'+str(T)+'.txt'
        strexotic_prc_no_cv = 'prc_exotic_no_cv'+str(T)+'.txt'
        strloss_rel='loss_rel'+str(T)+'.txt'
        strloss_vega='loss_vega'+str(T)+'.txt'
        strloss_martingale='martingale_test_'+str(T)+'.txt'
        strloss_call_atm='call_atm_'+str(T)+'.txt'
        strloss_put_atm='put_atm_'+str(T)+'.txt'
        strloss_put_call_parity_error = 'put_call_parity_error'+str(T)+'.txt'
        strIVerror='IV_error_'+str(T)+'.txt'
        strIVerror_infinity='IV_error_infinity'+str(T)+'.txt'
        print('Variance_V1:', torch.sum(var_pv_h_val))
        print('Variance_exotic with cv:', exotic_prc_var_cv_val)
        print('Variance_exotic without cv:', exotic_prc_var_no_cv_val)
        print('Price_exotic with cv:', exotic_prc_cv)
        print('Price_exotic without cv:', exotic_prc_no_cv)
        save_file(loss_val.cpu(), strloss)
        save_file(loss_val_rel.cpu(), strloss_rel)
        save_file(loss_vega.cpu(), strloss_vega)
        save_file(torch.sum(var_pv_h_val).cpu(), strvan_1_var)
        save_file(martingale_test.cpu(), strloss_martingale) 
        save_file(call_atm.cpu(), strloss_call_atm)
        save_file(put_atm.cpu(), strloss_put_atm) 
        save_file(put_call_parity_error.cpu(), strloss_put_call_parity_error)
        save_file(exotic_prc_var_cv_val.cpu(), strexotic_var_cv)
        save_file(exotic_prc_cv.cpu(), strexotic_prc_cv)
        save_file(exotic_prc_var_no_cv_val.cpu(), strexotic_var_no_cv)
        save_file(exotic_prc_no_cv.cpu(), strexotic_prc_no_cv)
        print('Martingale test:', martingale_test)
        print('Call price:', call_atm)
        print('Put price:', put_atm)
        print('Put call parity error:', put_call_parity_error)

        iv_1 = np.ones_like(K)
        iv_1_out = torch.Tensor().to(device)
        iv_1_error = np.max(np.absolute(iv_target_1-iv_1))
           
        with torch.no_grad():
                for idxt, t in enumerate(maturities):
                    van_pred_11 = pred[idxt,::].cpu().numpy()
                    for idxx, (pred_1, k) in enumerate(zip(van_pred_11, K)):
                        try:
                            iv_1[idxx] = implied_volatility(pred_1,  S=S0, K=k, r=0.0, t=t/(2*96), flag="c")
                        except:
                            pass
                    iv_1_out = torch.cat([iv_1_out,torch.from_numpy(iv_1).float().to(device).T],0)
        iv_max_error= loss_fn(iv_1_out,iv_target_1_out).cpu().numpy() #np.max(np.absolute(iv_1_out.cpu().numpy()-iv_target_1_out.cpu().numpy()))
        iv_infinity_error = np.max(np.absolute(iv_1_out.cpu().numpy()-iv_target_1_out.cpu().numpy()))
            
        print('iv_out', iv_1_out)
        print('iv_target', iv_target_1_out)
        print('iv_max_error', iv_max_error)
        print('iv_infinity_error', iv_infinity_error)
        save_file(iv_max_error, strIVerror)
        save_file(iv_infinity_error, strIVerror_infinity)
            
        if iv_max_error < best_IV_error and epoch<200:
            model_best=model
            best_IV_error=iv_max_error
            print('current_loss', loss_val)
            print('IV best error',iv_max_error)
            filename = "Neural_SDE_maturity{}.pth.tar".format(T)
            checkpoint = {"state_dict":model.state_dict(),
                         "T":T,
                         "pred":pv_h,
                         "MSE": loss_val,
                         "exotic_hedge_error": best_hedge_error,
                         "iv_MSE_error": iv_max_error,
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_val_rel,
                         "Exotic_prc": exotic_prc_cv,
                         "target_mat_T": target_mat_T}
            torch.save(checkpoint, filename)
                
        if  epoch==200:
            checkpoint_str= 'Neural_SDE_maturity'+str(96)+'.pth.tar'
            checkpoint=torch.load(checkpoint_str)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device) 
            filename = "Neural_SDE_test_maturity{}.pth.tar".format(T)
            np.random.seed(1234567)
            MC_samples_gen=1000000 # this is generated once and used to validate trained model after each epoch
            MC_samples_var=10000
            z_test = np.random.normal(size=(MC_samples_gen, n_steps))
            z_test_var = np.random.normal(size=(MC_samples_var, n_steps))
            z_test = torch.tensor(z).to(device=device).float()
            z_test_var = torch.tensor(z_var).to(device=device).float()
            with torch.no_grad():
                pv_h,_,exotic_prc_no_cv, exotic_prc_cv,_, _,martingale_test, put_atm, call_atm, put_call_parity_error = model(S0, rate, z_test, MC_samples_gen, T, period_length=16,n_maturities=n_maturities)
                _, var_pv_h_val,_,_, exotic_prc_var_no_cv_val,exotic_prc_var_cv_val, _,_,_,_ = model(S0, rate, z_test_var,  MC_samples_var, T, period_length=16,n_maturities=n_maturities)
            print("pred_1:",pv_h)
            print("target:", target_mat_T )
            pred=torch.reshape(pv_h,(6,21))
            print('pred', pred)
            loss_val = torch.sqrt(loss_fn(pred,target_mat_T))
            loss_vega = torch.sqrt(loss_fn(torch.mul(pred,vega),torch.mul(target_mat_T,vega)))
            loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_mat_T), torch.ones_like(pred)))
            print('validation {}, loss={:.10f}'.format(itercount, loss_val.item()))
            print('validation_rel {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
            print('validation_vega {}, loss={:.10f}'.format(itercount, loss_vega.item())) 
            exotic_hedge_error_change =torch.abs(1-exotic_hedge_error/exotic_prc_var_cv_val)  
            exotic_hedge_error = exotic_prc_var_cv_val
            strloss='loss_test_'+str(T)+'.txt'
            strvan_1_var = 'variance_test_van_1_'+str(T)+'.txt'
            strexotic_var_cv = 'variance_test_exotic_cv'+str(T)+'.txt'
            strexotic_var_no_cv = 'variance_test_exotic_no_cv'+str(T)+'.txt'
            strexotic_prc_cv = 'prc_test_exotic_cv'+str(T)+'.txt'
            strexotic_prc_no_cv = 'prc_test_exotic_no_cv'+str(T)+'.txt'
            strloss_rel='loss_test_rel'+str(T)+'.txt'
            strloss_vega='loss_test_vega'+str(T)+'.txt'
            strloss_martingale='martingale_test_set_test_'+str(T)+'.txt'
            strloss_call_atm='call_test_atm_'+str(T)+'.txt'
            strloss_put_atm='put_test_atm_'+str(T)+'.txt'
            strloss_put_call_parity_error = 'put_call_parity_error_test_'+str(T)+'.txt'
            strIVerror='IV_error_test_'+str(T)+'.txt'
            strIVerror_infinity='IV_error_infinity_test_'+str(T)+'.txt'
            print('Variance_V1:', torch.sum(var_pv_h_val))
            print('Variance_exotic with cv:', exotic_prc_var_cv_val)
            print('Variance_exotic without cv:', exotic_prc_var_no_cv_val)
            print('Price_exotic with cv:', exotic_prc_cv)
            print('Price_exotic without cv:', exotic_prc_no_cv)
            save_file(loss_val.cpu(), strloss)
            save_file(loss_val_rel.cpu(), strloss_rel)
            save_file(loss_vega.cpu(), strloss_vega)
            save_file(torch.sum(var_pv_h_val).cpu(), strvan_1_var)
            save_file(martingale_test.cpu(), strloss_martingale) 
            save_file(call_atm.cpu(), strloss_call_atm)
            save_file(put_atm.cpu(), strloss_put_atm) 
            save_file(put_call_parity_error.cpu(), strloss_put_call_parity_error)
            save_file(exotic_prc_var_cv_val.cpu(), strexotic_var_cv)
            save_file(exotic_prc_cv.cpu(), strexotic_prc_cv)
            save_file(exotic_prc_var_no_cv_val.cpu(), strexotic_var_no_cv)
            save_file(exotic_prc_no_cv.cpu(), strexotic_prc_no_cv)
            print('Martingale test:', martingale_test)
            print('Call price:', call_atm)
            print('Put price:', put_atm)
            print('Put call parity error:', put_call_parity_error)
            iv_1 = np.ones_like(K)
            iv_1_out = torch.Tensor().to(device)
            iv_1_error = np.max(np.absolute(iv_target_1-iv_1))
            with torch.no_grad():
                for idxt, t in enumerate(maturities):
                    van_pred_11 = pred[idxt,::].cpu().numpy()
                    for idxx, (pred_1, k) in enumerate(zip(van_pred_11, K)):
                        try:
                            iv_1[idxx] = implied_volatility(pred_1,  S=S0, K=k, r=0.0, t=t/(2*96), flag="c")
                        except:
                            pass
                    iv_1_out = torch.cat([iv_1_out,torch.from_numpy(iv_1).float().to(device).T],0)
            iv_max_error= loss_fn(iv_1_out,iv_target_1_out).cpu().numpy() #np.max(np.absolute(iv_1_out.cpu().numpy()-iv_target_1_out.cpu().numpy()))
            iv_infinity_error = np.max(np.absolute(iv_1_out.cpu().numpy()-iv_target_1_out.cpu().numpy()))
            print('iv_out_test', iv_1_out)
            print('iv_target', iv_target_1_out)
            print('iv_max_error_test', iv_max_error)
            print('iv_infinity_error_test', iv_infinity_error)
            save_file(iv_max_error, strIVerror)
            save_file(iv_infinity_error, strIVerror_infinity)
            checkpoint = {"state_dict":model.state_dict(),
                         "T":T,
                         "pred":pv_h,
                         "MSE": loss_val,
                         "exotic_hedge_error": best_hedge_error,
                         "iv_MSE_error": iv_max_error,
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_val_rel,
                         "Exotic_prc": exotic_prc_cv,
                         "target_mat_T": target_mat_T}
            torch.save(checkpoint, filename)
                  
    return model_best   

if __name__ == '__main__':

    MC_samples_gen=1000000 # this is generated once and used to validate trained model after each epoch
    MC_samples_var=400000
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--LAMBDA', type=int, default=10000)
    parser.add_argument('--c', type=int, default=20000)
    parser.add_argument('--MC_samples_gen',type=int,default=MC_samples_gen)
    parser.add_argument('--MC_samples_var',type=int,default=MC_samples_var)
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device='cuda:{}'.format(args.device)
        torch.cuda.set_device(args.device)
    else:
        device="cpu"
       
    data = np.load("call_prices_mat_6_21_uniform.npy")/1000
    print('data',data) 
    strikes_call = np.load("strikes_mat_6_21_uniform.npy")/1000
    print('strikes_call', strikes_call)
    
    # Set up training
    S0 = 3.221 # initial asset price 
    rate = 0.0
    maturities = range(16, 97, 16)
    n_steps = 96
    timegrid = torch.linspace(0,0.5,n_steps+1).to(device)
    
    np.random.seed(901) # fix seed for reproducibility
    z = np.random.normal(size=(MC_samples_gen, n_steps))
    z_var = np.random.normal(size=(MC_samples_var, n_steps))
    
    #  Random samples used to generate paths used for testing
    z = torch.tensor(z).to(device=device).float()
    z_var = torch.tensor(z_var).to(device=device).float()
 
    # Train nsde model
    train_nsde(args)  

    


      