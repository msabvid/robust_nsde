import sys
import os
sys.path.append(os.path.dirname('__file__'))

import torch
import torch.nn as nn
import numpy as np
import math
import time
import copy
import argparse
import random

from py_vollib.black_scholes.implied_volatility import implied_volatility 
from py_vollib.black_scholes import black_scholes as price_black_scholes 

from networks import *
         
class Net_LSV(nn.Module):
    """
    Calibration of LSV model to vanilla prices at different maturities
    """
    def __init__(self, device,n_strikes, n_networks):
        
        super(Net_LSV, self).__init__()
        self.device = device

        
        # initialise price diffusion neural network (different neural network for each maturity)
        self.S_vol =  Net_timegrid(dim=3, nOut=1, n_layers=3, vNetWidth=100, n_networks=1, activation_output="softplus")
        
        # initialise vanilla hedging strategy neural networks 
        """
        network for each maturity is used to hedge only options for that maturity, for example network corresponding to final maturity
        is used to simulate hedging strategy (from time 0) for vanilla options at the final maturity
        """
        self.vanilla_hedge = Net_timegrid(dim=2, nOut=n_strikes, n_layers=2, vNetWidth=20, n_networks=n_maturities, activation_output="softplus")

        # initialise stochastic volatility drift and diffusion neural networks
        self.V_drift = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_networks=1)
        self.V_vol = Net_timegrid(dim=1, nOut=1, n_layers=2, vNetWidth=20, n_networks=1, activation_output="softplus")
        
        # initialise stochastic volatility correlation and initial value parameters
        self.v0 = torch.nn.Parameter(torch.rand(1)*0.1)
        self.rho = torch.nn.Parameter(torch.ones(1)*(-2))
        
        # initialise exotic hedging strategy neural networks 
        """
        network for each maturity is used only to simulate hedging strategy for the timesteps in between previous and current maturity
        so that "n_maturities" networks are used to simulate hedging strategy for exotic option with maturity at the end of considered time horizon
        """
        self.exotic_hedge_straddle = Net_timegrid(dim=2, nOut=1, n_layers=3, vNetWidth=100, n_networks=1)
        self.exotic_hedge_lookback = Net_timegrid(dim=2, nOut=1, n_layers=3, vNetWidth=100, n_networks=1)

    def forward(self, S0, realised_prices,past_hedges, rate, z, MC_samples, ind_T,n_maturities, maturities,maturity_exotic, strikes, timegrid,gradient_count): 
        """
        pv_h:=price vanilla hedged pv_var_h:=variance of hedged vanilla price
        pe_h:=price exotic hedged; pe_var_h:=variance of hedged exotic price
        pe_u:=price exotic unhedged; pe_var_u:=variance of unhedged exotic price
        cv_vanilla_fwd-stores output of vanilla hedging strategies at each timestep
        """
        ones = torch.ones(MC_samples, 1, device=self.device)
        S_old = ones * S0
        if args.hedge_exotic:
            S_old_exotic = ones * realised_prices[0]
            cv_exotic = torch.zeros(S_old.shape[0], 1 , device=self.device)
        cv_vanilla = torch.zeros(S_old.shape[0], n_strikes,n_maturities, device=self.device)  
        pv_h = torch.zeros(n_strikes, device=self.device)
        pv_var_h = torch.zeros_like(pv_h)
        pe_u = torch.zeros(1, device=self.device)
        pe_h = torch.zeros(1, device=self.device)
        pe_var_h = torch.zeros(1, device=self.device)
        pe_var_u = torch.zeros(1, device=self.device)
        cv_vanilla_fwd = torch.zeros(S_old.shape[0], n_strikes,n_maturities, device=self.device)
        pv_h_out = torch.Tensor().to(device)  # used to store model vanilla prices for all maturities
        pv_var_h_out = torch.Tensor().to(device) # used to store variances of vanilla options for all maturities
        martingale_test = torch.zeros(1, device=self.device)
        put_atm = torch.zeros(1, device=self.device)
        call_atm = torch.zeros(1, device=self.device)
        V_old = ones * torch.sigmoid(self.v0)*0.5
        V_old_exotic = ones * torch.sigmoid(self.v0)*0.5
        rho = torch.tanh(self.rho)
        exotic_hedge_value = torch.zeros(1, device=self.device)
        
        if args.lookback_exotic:
            running_max = S_old_exotic # initialisation of running_max for calculation of lookback price
                 
        irand = random.sample(range(1,ind_T+1), gradient_count-1) # indices of timesteps used in gradient evaluation at each training epoch
    
        for i in range(1, ind_T+1):
            # assume that last timestep corresponds to the final vanilla maturity
            idx_net = [k for k in range(n_maturities) if int(timegrid[i]*365+0.001) <= maturities[k]] # maturities not necessarily uniform
            idx_net = idx_net[0]

            t = torch.ones_like(S_old) * timegrid[i-1]
            if args.hedge_exotic and i>cal_day:
                t_shift= torch.ones_like(S_old) * (timegrid[i-1]-timegrid[cal_day])
            h = timegrid[i]-timegrid[i-1]
            dW = (torch.sqrt(h) * z[:,i-1]).reshape(MC_samples,1)
            zz = torch.randn_like(dW)
            dB = rho[0] * dW + torch.sqrt(1-rho[0]**2)*torch.sqrt(h)*zz
            
            Slog_old=torch.log(S_old)
            if args.hedge_exotic:
                Slog_old_exotic=torch.log(S_old_exotic)
            price_diffusion = self.S_vol.forward_idx(0, torch.cat([t,Slog_old,V_old],1))
            price_diffusion_const = price_diffusion.detach()
           # price_diffusion_exotic = self.S_vol.forward_idx(0, torch.cat([t,Slog_old_exotic,V_old],1))
            if cal_day<i and args.hedge_exotic:
                price_diffusion_exotic = self.S_vol.forward_idx(0, torch.cat([t_shift,Slog_old_exotic,V_old_exotic],1))
                price_diffusion_exotic_const = price_diffusion_exotic.detach()
            # Evaluate vanilla hedging strategies at particular timestep
            for mat in range(n_maturities):
                if mat>=idx_net:
                    cv_vanilla_fwd[:,:,mat] = self.vanilla_hedge.forward_idx(mat, torch.cat([t,Slog_old.detach()],1))
                    
            if args.hedge_exotic and args.lookback_exotic:
                exotic_hedge = self.exotic_hedge_lookback.forward_idx(0, torch.cat([t,Slog_old_exotic.detach()],1))
                
            if args.hedge_exotic and args.straddle_exotic:
                exotic_hedge = self.exotic_hedge_straddle.forward_idx(0, torch.cat([t,Slog_old_exotic.detach()],1))    
            
            if i in irand:
                V_new = V_old + self.V_drift.forward_idx(0,torch.cat([V_old],1)).reshape(MC_samples,1)*h + self.V_vol.forward_idx(0,torch.cat([V_old],1)).reshape(MC_samples,1)*dB
                if cal_day<i and args.hedge_exotic:
                    V_new_exotic = V_old_exotic + self.V_drift.forward_idx(0,torch.cat([V_old_exotic],1)).reshape(MC_samples,1)*h + self.V_vol.forward_idx(0,torch.cat([V_old_exotic],1)).reshape(MC_samples,1)*dB
            else:
                V_new = V_old + self.V_drift.forward_idx(0,torch.cat([V_old],1)).detach().reshape(MC_samples,1)*h + self.V_vol.forward_idx(0,torch.cat([V_old],1)).detach().reshape(MC_samples,1)*dB
                if cal_day<i and args.hedge_exotic:
                    V_new_exotic = V_old_exotic + self.V_drift.forward_idx(0,torch.cat([V_old_exotic],1)).reshape(MC_samples,1)*h + self.V_vol.forward_idx(0,torch.cat([V_old_exotic],1)).reshape(MC_samples,1)*dB
                            
            drift =  (rate-0.5*(price_diffusion**2).reshape(MC_samples,1))
            diff = price_diffusion.reshape(MC_samples,1) * dW
            if cal_day<i and args.hedge_exotic:
                drift_exotic =  (rate-0.5*(price_diffusion_exotic**2).reshape(MC_samples,1))
                diff_exotic = price_diffusion_exotic.reshape(MC_samples,1) * dW
                drift_exotic_c =  (1+torch.abs(drift_exotic.detach())*torch.sqrt(h))
                diff_exotic_c = (1+torch.abs(price_diffusion_exotic_const.reshape(MC_samples,1))*torch.sqrt(h))

            # Drift normalisations in tamed Euler scheme have gradient detached
            drift_c = (1+torch.abs(drift.detach())*torch.sqrt(h))
            diff_c = (1+torch.abs(price_diffusion_const.reshape(MC_samples,1))*torch.sqrt(h))

         
            # Tamed Euler step
            if i in irand:
                if args.tamed_Euler:
                    Slog_new = Slog_old + drift*h/drift_c + diff/diff_c
                else:
                    Slog_new = Slog_old + drift*h + diff
            else:
                if args.tamed_Euler:
                    Slog_new = Slog_old + drift.detach()*h/drift_c + diff.detach()/diff_c
                else:
                    Slog_new = Slog_old + drift.detach()*h + diff.detach()
                    
            if i>cal_day and i in irand and args.hedge_exotic:
                if args.tamed_Euler:
                    Slog_new_exotic = Slog_old_exotic + drift_exotic*h/drift_exotic_c + diff_exotic/diff_exotic_c
                else:
                    Slog_new_exotic = Slog_old_exotic + drift_exotic*h + diff_exotic
            if i>cal_day and (i not in irand) and args.hedge_exotic:
                if args.tamed_Euler:
                    Slog_new_exotic = Slog_old_exotic + drift_exotic.detach()*h/drift_exoitc_c + diff_exoitc.detach()/diff_exotic_c
                else:
                    Slog_new_exotic = Slog_old_exotic + drift_exoitc.detach()*h + diff_exotic.detach()        
                    
            if i<=cal_day and args.hedge_exotic:
                S_new_exotic=ones*realised_prices[i]
            if i>cal_day and args.hedge_exotic:
                S_new_exotic=torch.exp(Slog_new_exotic)
            if i-1 ==cal_day and args.lookback_exotic:
                t_temp=torch.ones(1, 1, device=self.device)*timegrid[i-1]
                S_temp=torch.log(torch.ones(1, 1, device=self.device)*realised_prices[i-1]).detach()
                exotic_hedge_value =  self.exotic_hedge_lookback.forward_idx(0, torch.cat([t_temp,S_temp],1)).detach()
            if i-1 ==cal_day and args.straddle_exotic:
                t_temp=torch.ones(1, 1, device=self.device)*timegrid[i-1]
                S_temp=torch.log(torch.ones(1, 1, device=self.device)*realised_prices[i-1]).detach()
                exotic_hedge_value =  self.exotic_hedge_straddle.forward_idx(0, torch.cat([t_temp,S_temp],1)).detach()
            S_new=torch.exp(Slog_new)
            
            # Discounted price change between timesteps; used for control variate training 
            dS = torch.exp(-rate*timegrid[i])*S_new - torch.exp(-rate*timegrid[i-1])*S_old
            if args.hedge_exotic:
                dS_exotic = torch.exp(-rate*(timegrid[i]-timegrid[cal_day]))*S_new_exotic - torch.exp(-rate*(timegrid[i-1]-timegrid[cal_day]))*S_old_exotic

            # Evaluate stoch. integrals (w.r.t. discounted price process) corresponding to control variates
            for mat in range(n_maturities):
                if mat>=idx_net:
                    cv_vanilla[:,:,mat] += cv_vanilla_fwd[:,:,mat] * dS.detach().repeat(1,n_strikes)  
            if args.hedge_exotic and i>cal_day:
                cv_exotic +=  exotic_hedge.reshape(MC_samples,1) * dS_exotic.detach()
            if args.hedge_exotic and i<=cal_day:
                cv_exotic += past_hedges[i-1]*ones* dS_exotic.detach()
            # Evaluate exotic payoff
            if args.lookback_exotic:
                running_max = torch.max(running_max, S_new_exotic)  
                
            
            # Update values of asset prc processes for next Tamed Euler step
            S_old = S_new
            V_old = torch.clamp(V_new,0)
            
            if args.hedge_exotic:
                S_old_exotic = S_new_exotic
            if args.hedge_exotic and i>cal_day:    
                V_old_exotic = torch.clamp(V_new_exotic,0)
                  
            # Evaluate vanilla option prices and variances of price estimate if timestep corresponds to maturity  
            #print('int_i',int(i))
            #print('timegrid_i', timegrid[i])
            #print('maturities',maturities)
            if int(timegrid[i]*365+0.0001) in maturities: 
                dis_fact = torch.exp(-rate*timegrid[i])
                for idx, strike in enumerate(strikes[idx_net,:]):
                    if strike>S0:
                        pv_h_temp = dis_fact*torch.clamp(S_old-strike,0).squeeze(1).detach()-cv_vanilla[:,idx,idx_net]
                        pv_h_detach_cv = dis_fact*torch.clamp(S_old-strike,0).squeeze(1)-cv_vanilla[:,idx,idx_net].detach()
                        pv_h[idx] = pv_h_detach_cv.mean() # gradient of cv not applied when evaluating price of vanilla options 
                        pv_var_h[idx] = pv_h_temp.var() # gradient of price not applied when evaluating variance of vanilla options
                        if idx>0 and strike==strikes[idx_net,idx-1]:
                        #  some strikes are repeated if there is not enough strikes traded with non-zero volume
                        # repeated strikes are not considered in backpropagation 
                            pv_h[idx]=pv_h[idx].detach()
                          #  pv_var_h[idx]=pv_var_h[idx].detach()
                    if strike<S0:  
                        pv_h_temp = dis_fact*torch.clamp(-S_old+strike,0).squeeze(1).detach()-cv_vanilla[:,idx,idx_net]
                        pv_h_detach_cv = dis_fact*torch.clamp(-S_old+strike,0).squeeze(1)-cv_vanilla[:,idx,idx_net].detach()
                        pv_h[idx] = pv_h_detach_cv.mean() # gradient of cv not applied when evaluating price of vanilla options 
                        pv_var_h[idx] = pv_h_temp.var() # gradient of price not applied when evaluating variance of vanilla options
                        if idx>0 and strike==strikes[idx_net,idx-1]:
                        #  some strikes are repeated if there is not enough strikes traded with non-zero volume
                        # repeated strikes are not considered in backpropagation 
                            pv_h[idx]=pv_h[idx].detach()
                         #   pv_var_h[idx]=pv_var_h[idx].detach()
                pv_h_out = torch.cat([pv_h_out,pv_h.T],0) 
                pv_var_h_out = torch.cat([pv_var_h_out,pv_var_h.T],0)
                
            
            # Evaluate exotic option price and variance of exotic price estimates at final maturity (and martingale test)
            if (int(timegrid[i]*365+0.0001)==maturities[-1]) or int(timegrid[i]*365+0.0001)==maturity_exotic: 
                dis_fact = torch.exp(-rate*timegrid[i])
                martingale_test = dis_fact*S_old.detach()
                martingale_test = martingale_test.mean()/S0
                put_atm = dis_fact*torch.clamp(S0-S_old.detach(),0).squeeze(1)
                put_atm = put_atm.mean().detach()
                call_atm = dis_fact*torch.clamp(S_old.detach()-S0,0).squeeze(1)
                call_atm = call_atm.mean()
                put_call_parity_error = call_atm-put_atm - S0 + dis_fact*S0
                  
            if args.hedge_exotic and int(timegrid[i]*365+0.0001)==maturity_exotic :
                    dis_fact = torch.exp(-rate*(timegrid[i]-timegrid[cal_day]))
                    if args.lookback_exotic:
                        exotic_payoff = running_max - S_old_exotic
                    elif args.straddle_exotic:    
                        exotic_payoff = torch.clamp(S_old_exotic.detach()-realised_prices[0],0)+torch.clamp(-S_old_exotic.detach()+realised_prices[0],0)
                    exotic_payoff_detach = exotic_payoff.detach() # detach dependence on price for estimating variance of exotic                    
                    pe_h_temp = dis_fact*exotic_payoff_detach.squeeze(1).reshape(MC_samples,1)-cv_exotic.reshape(MC_samples,1)
                    pe_u = dis_fact*exotic_payoff.squeeze(1)
                    pe_var_h = pe_h_temp.var()
                    pe_var_u = pe_u.var()
                    pe_u = pe_u.mean() 
                    pe_h = dis_fact*exotic_payoff.squeeze(1).reshape(MC_samples,1)-cv_exotic.detach().reshape(MC_samples,1)
                    pe_h = pe_h.mean()

        return pv_h_out, pv_var_h_out, pe_u, pe_h, pe_var_u, pe_var_h, martingale_test, put_atm, call_atm, put_call_parity_error, exotic_hedge_value
        
def init_weights(m): 
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1.5)

def train_nsde(model,z_val,z_val_var,z_test,z_test_var,config):
    
    itercount = 0
    loss_fn = nn.MSELoss() 
    
    cal_day=config["cal_day"]
    maturities_daily = config["maturities_daily"]
    maturity_exotic_daily = config["maturity_exotic_daily"]
    maturities = config["maturities"]
    n_maturities = config["n_maturities"]
    n_strikes = config["n_strikes"]
    model = model.to(device)
              
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    maturity_values = config["maturity_values"]
    timegrid_daily = config["timegrid_daily"]
    timegrid = config["timegrid"]
    timegrid_exotic = config["timegrid_exotic"]
    hedges_lookback = config["hedges_lookback"]
    hedges_straddle = config["hedges_straddle"]
    gradient_count = config["gradient_count"]
    gradient_count_exotic = config["gradient_count_exotic"]
    batch_size_hedge = config["batch_size_hedge"]
    batch_steps_daily = len(timegrid_daily)-1 
    batch_steps_exotic = len(timegrid_exotic)-1
    batch_steps = len(timegrid)-1 
    T =batch_steps
    T_daily=batch_steps_daily
    T_exotic=batch_steps_exotic
    learning_rate =config["learning_rate"]
    K = config["strikes"]
    rate = config["interest_rate"][0]
    S0 = config["initial_price"]
    seed = config["seed"]
    n_strikes = len(K[0,:])
    
    params_SDE = list(model.S_vol.parameters()) + [model.rho, model.v0] + list(model.V_drift.parameters())+list(model.V_vol.parameters())
    params_vanilla_hedge = list(model.vanilla_hedge.parameters())
    
    past_hedges = hedges_lookback.to(device=device).float()
    if args.hedge_exotic and args.lookback_exotic:
        params_exotic_hedge = list(model.exotic_hedge_lookback.parameters())
        
    if args.hedge_exotic and args.straddle_exotic:
        params_exotic_hedge = list(model.exotic_hedge_straddle.parameters())
        past_hedges = hedges_straddle.to(device=device).float()

    optimizer_vanilla_hedge = torch.optim.Adam(params_vanilla_hedge,lr=25*learning_rate)
    optimizer_SDE = torch.optim.Adam(params_SDE,lr=25*learning_rate)
    
    if args.hedge_exotic:
        optimizer_exotic_hedge = torch.optim.Adam(params_exotic_hedge,lr=1*learning_rate)

    best_IV_mean_error = 10
    best_hedge_error = 9999999999
    target_option_prices = torch.tensor(config["target_data"][:len(config["maturity_values"]),:len(config["strikes"][0,:])], device=device).float()
       

    # Number of paths used for validation
    MC_samples_price = args.MC_samples_price
    MC_samples_var = args.MC_samples_var
    
        
    for epoch in range(n_epochs):
        
        # evaluate model at initialisation and save error, exotic price and other statistics at initialisation
        if epoch==0:

            # calculate IV of target options
            iv_target = np.zeros_like(K[0,:])
            iv_target_out = torch.Tensor().to(device)
            try:
                for idxt, t in enumerate(maturities_daily):
                    van_target = target_option_prices[idxt,::].cpu().numpy() 
                    for idxx, ( target, k) in enumerate(zip(van_target, K[idxt,:])):
                        if k>S0: 
                            iv_target[idxx] = implied_volatility(target,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="c")
                        else:
                            iv_target[idxx] = implied_volatility(target,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="p")
                    #print('iv_target',iv_target[idxx])
                    iv_target_out = torch.cat([iv_target_out,torch.from_numpy(iv_target).float().to(device).T],0)  
            except:
                pass
            print('Target Implied Volatility Surface',iv_target_out)
                
            # for each of the target options calculate inverse of the vega (used as calibration weighting scheme) 
            inverse_vega_target = np.ones_like(K[0,:])
            inverse_vega_out = torch.Tensor().to(device)
            constant_shift=0.01 # vega of each option is calculated by shifting corresponding IV by 1% 
                                    
            try:
                for idxt, t in enumerate(maturities_daily):
                    van_target = target_option_prices[idxt,::].cpu().numpy() 
                    for idxx, (target, k) in enumerate(zip(van_target, K[idxt,:])):
                        if k>S0:
                            iv_target_temp = implied_volatility(target,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="c")
                        else:
                            iv_target_temp = implied_volatility(target,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="p")
                        iv = iv_target_temp+constant_shift
                        if k>S0:
                            inverse_vega_target[idxx] = constant_shift/(price_black_scholes(flag='c',S=S0,K=k,t=maturity_values[idxt]-1/365,r=rate.cpu().numpy(),sigma=iv)-target)
                        else:
                            inverse_vega_target[idxx] = constant_shift/(price_black_scholes(flag='p',S=S0,K=k,t=maturity_values[idxt]-1/365,r=rate.cpu().numpy(),sigma=iv)-target)
                    inverse_vega_out = torch.cat([inverse_vega_out,torch.from_numpy(inverse_vega_target).float().to(device).T],0)                    
            except:
                    pass

            inverse_vega=torch.reshape(inverse_vega_out,(n_maturities,n_strikes))
            
            #normalise the weights 
            inverse_vega=(n_strikes*n_maturities)*inverse_vega/torch.sum(inverse_vega)
            
            print('Weighting scheme for target options:',inverse_vega)
            
            with torch.no_grad():
                if args.hedge_exotic: 
                    _, _,pe_u, pe_h,_, _,_, _, _, _,_ =   model(S0, realised_prices,past_hedges, rate, z_val, MC_samples_price, batch_steps_exotic,n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_exotic,gradient_count=gradient_count_exotic)
                    _, _,_,_, pe_var_u,pe_var_h, _,_,_,_,_ = model(S0, realised_prices,past_hedges, rate, z_val_var,MC_samples_var, batch_steps_exotic,n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_exotic,gradient_count=gradient_count_exotic)
                pv_h, _,_, _,_, _,martingale_test, put_atm, call_atm, put_call_parity_error,_ =   model(S0, realised_prices,past_hedges, rate, z_val, MC_samples_price, batch_steps_daily,n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_daily,gradient_count=gradient_count)
                _, var_pv_h,_,_, _, _, _,_,_,_,_ = model(S0, realised_prices,past_hedges, rate, z_val_var,MC_samples_var, batch_steps_daily,n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_daily,gradient_count=gradient_count)                 
            
            pred=torch.reshape(pv_h,(n_maturities,n_strikes))
            print("Model Option Prices at Initialisation:",pred)
            print("Target Option OTM Prices:", target_option_prices )
            loss_val = torch.sqrt(loss_fn(pred,target_option_prices))
            loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_option_prices), torch.ones_like(pred)))
            loss_vega = torch.sqrt(loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_option_prices,inverse_vega)))
            
            if args.hedge_exotic==False:
                filename = "NSDE_model_initial_prices_LSV_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
            if args.lookback_exotic==True and args.hedge_exotic==True:
                filename = "NSDE_initial_error_hedge_lookback_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
            if args.straddle_exotic==True and args.hedge_exotic==True:
                filename = "NSDE_initial_error_hedge_straddle_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)  

            if args.hedge_exotic==False:
                checkpoint = {"T":maturity_exotic_daily,
                         "pred":pred,
                         "rho": torch.tanh(model.rho).detach().cpu().numpy(),
                         "target_option_prices": target_option_prices}
                torch.save(checkpoint, filename)
            else:
                checkpoint = {"T":maturity_exotic_daily,
                         "pred":pred,
                         "exotic_hedge_error": pe_var_h,
                         "exotic_variance_unhedged": pe_var_u,
                         "rho": torch.tanh(model.rho).detach().cpu().numpy(),
                         "Exotic_prc": pe_h,
                         "Exotic_prc unhedged": pe_u,     
                         "target_option_prices": target_option_prices}
                torch.save(checkpoint, filename)
            
            
            with open("Price_Call_and_Price_Put_6_Month_ATM.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(call_atm.item(),put_atm.item() ))
            if args.hedge_exotic and args.lookback_exotic:
                with open("Price_Lookback_Hedged_and_Price_Lookback_Unhedged.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_h.item(),pe_u.item() ))
                with open("Lookback_Hedged_Variance_Lookback_Unhedged_Variance.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_var_h.item(),pe_var_u.item() ))  
           
            if args.hedge_exotic and args.straddle_exotic:
                with open("Price_Straddle_Hedged_and_Price_Straddle_Unhedged.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_h.item(),pe_u.item() ))
                with open("Straddle_Hedged_Variance_Straddle_Unhedged_Variance.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_var_h.item(),pe_var_u.item() )) 
                    
                    
            V_initial = torch.sigmoid(model.v0)*0.5
            rhos = torch.tanh(model.rho)        
                    
            with open("rho_and_v0.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(rhos[0].item(),V_initial.item()))        
            
            with open("Sum_Variances_Vanilla_Options.txt","a") as f:
                f.write("{:.4f}\n".format(torch.sum(var_pv_h).item() ))
            
            with open("Loss_MSE_Loss_REL_Loss_Vega_Weighted.txt","a") as f:
                f.write("{:.4f},{:.4f},{:.4f}\n".format(loss_val.item(),loss_val_rel.item(),loss_vega.item() ))
            
            with open("Put_Call_Parity_Error_and_Martingale_Test.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(put_call_parity_error.item(),martingale_test.item() ))
            
            print('Validation Mean Square Error At Initialisation {}, loss={:.10f}'.format(itercount, loss_val.item()))
            print('Validation Relative Error At Initialisation {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
            print('Validation Iverse Vega Weighted MSE At Initialisation {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
            print('Sum of Variances of Vanilla OTM Options At Initialisation:', torch.sum(var_pv_h))
            
            if args.hedge_exotic:
                print('Variance of Hedged Exotic Option At Initialisation:', pe_var_h)
                print('Variance of Unhedged Exotic Option At Initialisation:', pe_var_u)
                print('Price of Hedged Exotic Option At Initialisation:', pe_h)
                print('Price of Unhedged Exotic Option At Initialisation:', pe_u)
            print('Martingale Test At Initialisation:', martingale_test)
            print('ATM 6-Month Call Price At Initialisation:', call_atm)
            print('ATM 6-Month Put Price At initialisation:', put_atm)
            print('ATM 6-Month Put-Call Parity Error At Initialisation:', put_call_parity_error)
            print('Rho at initialisation:', torch.tanh(model.rho))
          
                        
        print('Epoch, batch size:', epoch, batch_size)
        for i in range(0,20*batch_size, batch_size):
            # simulate paths over entire training horizon 

            if args.hedge_exotic==False:
                optimizer_SDE.zero_grad()
                optimizer_vanilla_hedge.zero_grad()
            
            if args.hedge_exotic:
                optimizer_exotic_hedge.zero_grad()
                
            # Start training CV networks from 1st epoch, at initial (0th) epoch neural networks corresponding to SDE parameters are trained  
            
            # Train Vanilla Hedging Strategy 
            if epoch%10==1 and args.hedge_exotic==False: 
                model.vanilla_hedge.unfreeze()
                model.S_vol.freeze()
                model.V_drift.freeze()
                model.V_vol.freeze()
                model.exotic_hedge_straddle.freeze()
                model.exotic_hedge_lookback.freeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size_hedge, batch_steps, device=device)
                _, var_pv_h, _,_,_, _, _, _, _, _,_ = model(S0, realised_prices,past_hedges, rate, batch_x1, batch_size_hedge,T,n_maturities=n_maturities,maturities=maturities,maturity_exotic=maturity_exotic,strikes=strikes,timegrid=timegrid,gradient_count=gradient_count)
                time_forward = time.time() - init_time
                loss = var_pv_h.sum()
                init_time = time.time()
                itercount += 1
                loss.backward()
                time_backward = time.time() - init_time
                if (itercount % 20 == 0):
                    print('Training Vanilla Hedging Strategy')
                    print('iteration {}, sum_variance_vanilla={:.4f},time_forward={:.4f}, time_backward={:.4f}'.format(itercount, loss.item(), time_forward, time_backward))
                optimizer_vanilla_hedge.step()
                  
            # Train Neural Networks Corresponding to SDE Parameters to vanilla target with path-dependent objective
            if epoch%10!=1 and args.hedge_exotic==False: 
                model.S_vol.unfreeze()
                model.V_drift.unfreeze()
                model.V_vol.unfreeze()
                model.vanilla_hedge.freeze()
                model.exotic_hedge_lookback.freeze()
                model.exotic_hedge_straddle.freeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size, batch_steps_daily, device=device)
                pv_h, _, _,_,_, _, _, put_atm, call_atm, put_call_parity_error,_ = model(S0, realised_prices, past_hedges, rate, batch_x1, batch_size,T,n_maturities=n_maturities, maturities=maturities,maturity_exotic=maturity_exotic,strikes=strikes,timegrid=timegrid,gradient_count=gradient_count)
                time_forward = time.time() - init_time
                pred = torch.reshape(pv_h,(n_maturities,n_strikes))
                MSE = loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_option_prices,inverse_vega))
                loss= MSE #opt_constant*pe_h + LAMBDA_2 * MSE + c_2/2 * MSE**2  
                init_time = time.time()
                itercount +=1
                loss.backward()
                time_backward = time.time() - init_time
                if (itercount % 20 == 0):
                    print('Training SDE Parameters')
                    print('iteration {}, Loss={:.4f}, time_forward={:.4f}, time_backward={:.4f}'.format(itercount,loss.item(), time_forward, time_backward))
                optimizer_SDE.step()

            # Train Neural Network Corresponding to Exotic Hedging Strategy
            if args.hedge_exotic: 
                model.S_vol.freeze()
                model.V_drift.freeze()
                model.V_vol.freeze()
                model.vanilla_hedge.freeze()
                if args.lookback_exotic:
                    model.exotic_hedge_lookback.unfreeze()
                    model.exotic_hedge_straddle.freeze()
                if args.straddle_exotic:
                    model.exotic_hedge_straddle.unfreeze()
                    model.exotic_hedge_lookback.freeze()
                init_time = time.time()
                batch_x1 = torch.randn(batch_size_hedge, batch_steps, device=device)
                _, _, _,pe_h, _,pe_var_h,_, _, _, _, _ = model(S0, realised_prices, past_hedges, rate, batch_x1, batch_size_hedge,T_exotic,n_maturities=n_maturities, maturities=maturities,maturity_exotic=maturity_exotic,strikes=strikes,timegrid=timegrid_exotic,gradient_count=gradient_count_exotic)
                time_forward = time.time() - init_time
                loss= pe_var_h
                init_time = t
                itercount +=1
                loss.backward()
                time_backward = time.time() - init_time
                if (itercount % 20 == 0):
                    print('Training Exotic Hedging Strategy')
                    print('iteration {}, variance_exotic_option={:.4f}, time_backward={:.4f}'.format(itercount, loss.item(), time_forward, time_backward))
                optimizer_exotic_hedge.step()                                    
         
        with torch.no_grad():
            if args.hedge_exotic: 
                _, _,pe_u_val, pe_h_val,_, _,_, _, _, _, _ = model(S0, realised_prices, past_hedges, rate, z_val, MC_samples_price, T_exotic, n_maturities=n_maturities, maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_exotic,gradient_count=gradient_count_exotic)
                _, _,_,_, pe_var_u_val,pe_var_h_val, _,_,_,_,_ = model(S0, realised_prices, past_hedges, rate, z_val_var,  MC_samples_var, T_exotic, n_maturities=n_maturities, maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_exotic,gradient_count=gradient_count_exotic)
            pv_h_val, _,_, _,_, _,martingale_test, put_atm, call_atm, put_call_parity_error, _ = model(S0, realised_prices, past_hedges, rate, z_val, MC_samples_price, T_daily, n_maturities=n_maturities,maturity_exotic=maturity_exotic_daily, maturities=maturities_daily,strikes=strikes,timegrid=timegrid_daily,gradient_count=gradient_count)
            _, var_pv_h_val,_,_, _,_, _,_,_,_,_ = model(S0, realised_prices, past_hedges, rate, z_val_var,  MC_samples_var, T_daily, n_maturities=n_maturities, maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_daily,gradient_count=gradient_count)
                
        
        pred=torch.reshape(pv_h_val,(n_maturities,n_strikes))
        print("Model Option Prices:",pred)
        print("Target Option OTM Prices:", target_option_prices )
        print("Strikes:", K)
        loss_val = torch.sqrt(loss_fn(pred,target_option_prices))
        loss_val_rel  = torch.sqrt(loss_fn(torch.div(pred,target_option_prices), torch.ones_like(pred)))
        loss_vega = torch.sqrt(loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_option_prices,inverse_vega)))
            
        with open("Price_Call_and_Price_Put_6_Month_ATM.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(call_atm.item(),put_atm.item() ))
        if args.hedge_exotic and args.lookback_exotic:
            with open("Price_Lookback_Hedged_and_Price_Lookback_Unhedged.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(pe_h_val.item(),pe_u_val.item() ))
            with open("Lookback_Hedged_Variance_Lookback_Unhedged_Variance.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(pe_var_h_val.item(),pe_var_u_val.item() )) 
        if args.hedge_exotic and args.straddle_exotic:
            with open("Price_Straddle_Hedged_and_Price_Straddle_Unhedged.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(pe_h_val.item(),pe_u_val.item() ))
            with open("Straddle_Hedged_Variance_Straddle_Unhedged_Variance.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(pe_var_h_val.item(),pe_var_u_val.item() ))         
                
                
        V_initial = torch.sigmoid(model.v0)*0.5
        rhos = torch.tanh(model.rho)        
                    
        with open("rho_and_v0.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(rhos[0].item(),V_initial.item()))             
                       
        with open("Loss_MSE_Loss_REL_Loss_Vega_Weighted.txt","a") as f:
            f.write("{:.4f},{:.4f},{:.4f}\n".format(loss_val.item(),loss_val_rel.item(),loss_vega.item() ))
            
        with open("Put_Call_Parity_Error_and_Martingale_Test.txt","a") as f:
            f.write("{:.4f},{:.4f}\n".format(put_call_parity_error.item(),martingale_test.item() ))
            
        with open("Sum_Variances_Vanilla_Options.txt","a") as f:
                f.write("{:.4f}\n".format(torch.sum(var_pv_h_val).item() ))    
            
        print('Validation Mean Square Error {}, loss={:.10f}'.format(itercount, loss_val.item()))
        print('Validation Relative Error {}, loss={:.10f}'.format(itercount, loss_val_rel.item()))
        print('Validation Iverse Vega Weighted MSE {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
        print('Sum of Variances of Vanilla OTM Options:', torch.sum(var_pv_h_val))
        if args.hedge_exotic:
            print('Variance of Hedged Exotic Option:', pe_var_h_val)
            print('Variance of Unhedged Exotic Option:', pe_var_u_val)
            print('Price of Hedged Exotic Option:', pe_h_val)
            print('Price of Unhedged Exotic Option:', pe_u_val)
        print('Martingale Test:', martingale_test)
        print('ATM 6-Month Call Price:', call_atm)
        print('ATM 6-Month Put Price:', put_atm)
        print('ATM 6-Month Put-Call Parity Error:', put_call_parity_error)
        print('Rho:', torch.tanh(model.rho))
          
        iv = np.ones_like(K[0,:])*1
        iv_out = torch.Tensor().to(device)         
        with torch.no_grad():
                for idxt, t in enumerate(maturities_daily):
                    van_pred_temp = pred[idxt,::].cpu().numpy()
                    for idxx, (pred_temp, k) in enumerate(zip(van_pred_temp, K[idxt,:])):
                        try:
                            if k>S0:
                                iv[idxx] = implied_volatility(pred_temp,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="c")
                            else:
                                iv[idxx] = implied_volatility(pred_temp,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="p")
                        except:
                            pass
                    iv_out = torch.cat([iv_out,torch.from_numpy(iv).float().to(device).T],0)
        if  args.dynamic_weight_update:
             inverse_vega += (n_strikes*n_maturities)*torch.clamp(torch.abs(iv_out-iv_target_out).reshape(n_maturities,n_strikes), 0.001,0.5)
             inverse_vega = (n_strikes*n_maturities)*inverse_vega/torch.sum(inverse_vega)
        print('Vega weights', inverse_vega)
                                      
        iv_mean_error= loss_fn(iv_out,iv_target_out).cpu().numpy() 
        iv_infinity_error = np.max(np.absolute(iv_out.cpu().numpy()-iv_target_out.cpu().numpy()))
        with open("IV_Mean_Infinity_Errors.txt","a") as f:
             f.write("{:.4f},{:.4f}\n".format(iv_mean_error.item(),iv_infinity_error.item())) 
            
        print('Implied Volatility Model', iv_out)
        print('Implied Volatility Target', iv_target_out)
        print('Implied Volatility Mean Square Error', iv_mean_error)
        print('Implied Volatility Infinity_error', iv_infinity_error)
          
        if iv_mean_error < best_IV_mean_error and args.hedge_exotic==False:
            model_best=model
            best_IV_mean_error=iv_mean_error
            print('current_loss', loss_val)
            print('IV best error',iv_mean_error)
            filename = "NSDE_val_LSV_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
            checkpoint = {"state_dict":model.state_dict(),
                         "T":maturity_exotic_daily,
                         "pred":pred,
                         "MSE": loss_val,
                         "rho": torch.tanh(model.rho).detach().cpu().numpy(),
                         "iv_MSE_error": iv_mean_error,
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_val_rel,
                         "target_option_prices": target_option_prices}
            torch.save(checkpoint, filename)             
            
        if  args.hedge_exotic and pe_var_h_val < best_hedge_error:
            model_best=model
            best_hedge_error=pe_var_h_val
            print('current_loss', loss_val)
            print('IV best error',iv_mean_error)
            if args.lookback_exotic:
                filename = "NSDE_val_hedge_lookback_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
                checkpoint = {"state_dict":model.state_dict(),
                         "T":maturity_exotic_daily,
                         "pred":pred,
                         "MSE": loss_val,
                         "exotic_hedge_error": pe_var_h_val,
                         "rho": torch.tanh(model.rho).detach().cpu().numpy(),
                         "iv_MSE_error": iv_mean_error,
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_val_rel,
                         "Exotic_prc": pe_h_val,
                         "target_option_prices": target_option_prices}
                torch.save(checkpoint, filename)    
                
            if args.straddle_exotic:
                filename = "NSDE_val_hedge_straddle_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
                checkpoint = {"state_dict":model.state_dict(),
                         "T":maturity_exotic_daily,
                         "pred":pred,
                         "MSE": loss_val,
                         "exotic_hedge_error": pe_var_h_val,
                         "rho": torch.tanh(model.rho).detach().cpu().numpy(),
                         "iv_MSE_error": iv_mean_error,
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_val_rel,
                         "Exotic_prc": pe_h_val,
                         "target_option_prices": target_option_prices}
                torch.save(checkpoint, filename)        
            
         
        if  epoch==(n_epochs-1):
            
            if args.hedge_exotic==False:
                checkpoint_str= "NSDE_val_LSV_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
            if args.lookback_exotic==True and args.hedge_exotic==True:
                checkpoint_str= "NSDE_val_hedge_lookback_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
            if args.straddle_exotic==True and args.hedge_exotic==True:
                checkpoint_str= "NSDE_val_hedge_straddle_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)   

            checkpoint=torch.load(checkpoint_str)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device) 
            
            if args.hedge_exotic==False:
                filename = "NSDE_test_LSV_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
            if args.lookback_exotic==True and args.hedge_exotic==True:
                filename = "NSDE_test_hedge_lookback_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)
            if args.straddle_exotic==True and args.hedge_exotic==True:
                filename = "NSDE_test_hedge_straddle_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day)   

            with torch.no_grad():
                if args.hedge_exotic: 
                    _, _,pe_u_test, pe_h_test,_, _,_, _, _, _, _ =   model(S0, realised_prices, past_hedges, rate, z_test, MC_samples_price, T_exotic, n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_exotic,gradient_count=gradient_count_exotic)
                    _, _,_,_, pe_var_u_test,pe_var_h_test, _,_,_,_, exotic_hedge_value = model(S0, realised_prices, past_hedges, rate, z_test_var,  MC_samples_var, T_exotic, n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_exotic,gradient_count=gradient_count_exotic)
                pv_h_test, _,_, _,_, _,martingale_test, put_atm, call_atm, put_call_parity_error, _ =  model(S0,realised_prices, past_hedges, rate, z_test, MC_samples_price, T_daily, n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_daily,gradient_count=gradient_count)
                _, var_pv_h_test,_,_, _, _, _,_,_,_, _ = model(S0,realised_prices, past_hedges, rate, z_test_var,  MC_samples_var, T_daily, n_maturities=n_maturities,maturities=maturities_daily,maturity_exotic=maturity_exotic_daily,strikes=strikes,timegrid=timegrid_daily,gradient_count=gradient_count)                 
            if args.hedge_exotic:
                past_hedges[cal_day]=exotic_hedge_value
            print('past_hedges', past_hedges)
            pred=torch.reshape(pv_h_test,(n_maturities,n_strikes))
            print("Model Option Prices:",pred)
            print("Target Option Call Prices:", target_option_prices )
        
            loss_test = torch.sqrt(loss_fn(pred,target_option_prices))
            loss_test_rel  = torch.sqrt(loss_fn(torch.div(pred,target_option_prices), torch.ones_like(pred)))
            loss_vega = torch.sqrt(loss_fn(torch.mul(pred,inverse_vega),torch.mul(target_option_prices,inverse_vega)))
            
            with open("Price_Call_and_Price_Put_6_Month_ATM.txt","a") as f:
                 f.write("{:.4f},{:.4f}\n".format(call_atm.item(),put_atm.item() ))
                    
            if args.hedge_exotic and args.lookback_exotic:
                with open("Price_Lookback_Hedged_and_Price_Lookback_Unhedged.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_h_test.item(),pe_u_test.item() ))
                with open("Lookback_Hedged_Variance_Lookback_Unhedged_Variance.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_var_h_test.item(),pe_var_u_test.item() ))
                    
            if args.hedge_exotic and args.straddle_exotic:
                with open("Price_Straddle_Hedged_and_Price_Straddle_Unhedged.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_h_test.item(),pe_u_test.item() ))
                with open("Straddle_Hedged_Variance_Straddle_Unhedged_Variance.txt","a") as f:
                    f.write("{:.4f},{:.4f}\n".format(pe_var_h_test.item(),pe_var_u_test.item() ))        
                    
            
            V_initial = torch.sigmoid(model.v0)*0.5
            rhos = torch.tanh(model.rho)        
                    
            with open("rho_and_v0.txt","a") as f:
                f.write("{:.4f},{:.4f},{:.4f}\n".format(rhos[0].item(),rhos[-1].item(),V_initial.item()))                      
                              
            with open("Loss_MSE_Loss_REL_Loss_Vega_Weighted.txt","a") as f:
                 f.write("{:.4f},{:.4f},{:.4f}\n".format(loss_test.item(),loss_test_rel.item(),loss_vega.item() ))
            
            with open("Put_Call_Parity_Error_and_Martingale_Test.txt","a") as f:
                 f.write("{:.4f},{:.4f}\n".format(put_call_parity_error.item(),martingale_test.item() ))
                    
            with open("Sum_Variances_Vanilla_Options.txt","a") as f:
                f.write("{:.4f}\n".format(torch.sum(var_pv_h_test).item() ))        
            
            print('Test Mean Square Error {}, loss={:.10f}'.format(itercount, loss_test.item()))
            print('Validation Relative Error {}, loss={:.10f}'.format(itercount, loss_test_rel.item()))
            print('Validation Iverse Vega Weighted MSE {}, loss={:.10f}'.format(itercount, loss_vega.item()))   
            print('Sum of Variances of Vanilla OTM Options:', torch.sum(var_pv_h_test))
            if args.hedge_exotic:
                print('Variance of Hedged Exotic Option:', pe_var_h_test)
                print('Variance of Unhedged Exotic Option:', pe_var_u_test)
                print('Price of Hedged Exotic Option:', pe_h_test)
                print('Price of Unhedged Exotic Option:', pe_u_test)
            print('Martingale Test:', martingale_test)
            print('ATM 6-Month Call Price:', call_atm)
            print('ATM 6-Month Put Price:', put_atm)
            print('ATM 6-Month Put-Call Parity Error:', put_call_parity_error)
            print('Rho:', torch.tanh(model.rho))
         
                     
            iv = np.ones_like(K[0,:])
            iv_out = torch.Tensor().to(device)
            
            with torch.no_grad():
                for idxt, t in enumerate(maturities_daily):
                    van_pred_temp = pred[idxt,::].cpu().numpy()
                    for idxx, (pred_temp, k) in enumerate(zip(van_pred_temp, K[idxt,:])):
                        try:
                            if k>S0:
                                iv[idxx] = implied_volatility(pred_temp,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="c")
                            else:
                                iv[idxx] = implied_volatility(pred_temp,  S=S0, K=k, r=rate.cpu().numpy(), t=maturity_values[idxt]-1/365, flag="p")
                        except:
                            pass
                    iv_out = torch.cat([iv_out,torch.from_numpy(iv).float().to(device).T],0)
                    
            iv_mean_error= loss_fn(iv_out,iv_target_out).cpu().numpy() 
            iv_infinity_error = np.max(np.absolute(iv_out.cpu().numpy()-iv_target_out.cpu().numpy()))
            
            print('iv_out_test', iv_out)
            print('iv_target', iv_target_out)
            print('iv_mean_error_test', iv_mean_error)
            print('iv_infinity_error_test', iv_infinity_error)
            with open("IV_Mean_Infinity_Errors.txt","a") as f:
                f.write("{:.4f},{:.4f}\n".format(iv_mean_error.item(),iv_infinity_error.item())) 
            if args.hedge_exotic==False:
                checkpoint = {"state_dict":model.state_dict(),
                         "T":maturity_exotic_daily,
                         "pred":pred,
                         "MSE": loss_test,
                         "iv_MSE_error": iv_mean_error,
                         "rho": torch.tanh(model.rho).detach().cpu().numpy(),
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_test_rel,
                         "target_option_prices": target_option_prices}
                torch.save(checkpoint, filename)
            else:
                checkpoint = {"state_dict":model.state_dict(),
                         "T":maturity_exotic_daily,
                         "pred":pred,
                         "MSE": loss_test,
                         "exotic_hedge_error": pe_var_h_test,
                         "past_hedges": past_hedges,     
                         "iv_MSE_error": iv_mean_error,
                         "rho": torch.tanh(model.rho).detach().cpu().numpy(),
                         "iv_infinity_error": iv_infinity_error,
                         "MSE_rel": loss_test_rel,
                         "Exotic_prc": pe_h_test,
                         "target_option_prices": target_option_prices}
                torch.save(checkpoint, filename)
                     
                  
    return model_best   

if __name__ == '__main__':

    MC_samples_price=1000000 # this is generated once and used to validate trained model after each epoch
    MC_samples_var=400000

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--MC_samples_price',type=int,default=MC_samples_price)
    parser.add_argument('--MC_samples_var',type=int,default=MC_samples_var)
    parser.add_argument('--tamed_Euler', action='store_true', default=False)
    parser.add_argument('--hedge_exotic',action='store_true', default=False) # initially we only train model to IV data
    parser.add_argument('--lookback_exotic',action='store_true', default=False)
    parser.add_argument('--straddle_exotic',action='store_true', default=False)
    parser.add_argument('--dynamic_weight_update',action='store_true', default=True)
    args = parser.parse_args()  

    if torch.cuda.is_available():
        device='cuda:{}'.format(args.device)
        torch.cuda.set_device(args.device)
    else:
        device="cpu"
        
    # Set up training   
    data= np.load("target_prices.npy")
    print('data',data) 
    strikes= np.load("strikes.npy")
    rates = np.load("rates.npy")
    print('strikes', strikes)
    
    maturity_values = np.load("maturities.npy")
    cal_day = np.load("cal_day.npy")
    realised_prices = torch.tensor(np.load("GOOG_ALL.npy").astype('float32'))[0:cal_day+1].to(device)
    print('realised_prices',realised_prices)
    maturity_value_exotic = np.load("maturity_exotic.npy")
    n_maturities = len(maturity_values)
    n_strikes = len(strikes[0,:])
    maturities_daily = [int(round(item)) for item in 365*maturity_values]
    maturity_exotic_daily = int(round(365*maturity_value_exotic))
    maturity_exotic = maturity_exotic_daily # model is calibrated and validate on the same daily timegrid
    print('maturities_daily',maturities_daily)
    S0 = realised_prices[cal_day] # initial asset price 
    print('initial price', S0)

    timegrid_daily = np.load("timegrid.npy")
    timegrid_exotic = np.load("timegrid_exotic.npy")
    n_steps_daily =  len(timegrid_daily)
    n_steps_exotic = len(timegrid_exotic)
    timegrid_daily = torch.tensor(timegrid_daily).to(device)
    timegrid_exotic = torch.tensor(timegrid_exotic).to(device)
    rate = (torch.tensor(rates).to(device)).type(torch.float32)
    seed=456


    timegrid = timegrid_daily # use daily timegrid to train the model (otherwise daily timegrid is used to validate the model)
    n_steps = n_steps_daily
    maturities = maturities_daily # indices of maturities that are used to train the model correspond to daily timegrid
    gradient_count = int(round(1.0*(n_steps))) # use 100% of timesteps to evaluate gradient for backpropagation for maturities beyond first
    gradient_count_exotic = int(round(1.0*(n_steps_exotic)))
    
    torch.manual_seed(seed) # fixed for reproducibility
    model = Net_LSV( device=device,n_strikes=n_strikes, n_networks=1)
    model.to(device)
   # model.apply(init_weights) # default uniform Pytorch initialisation of weights tends to work better
   # for name, param in model.named_parameters():
    #    if param.requires_grad:
     #      print(name, param.data)
        
    if cal_day==0:
        for idx in range(1,n_maturities):
            model.vanilla_hedge.net_t[idx].h_o = copy.deepcopy(model.vanilla_hedge.net_t[0].h_o)
            model.vanilla_hedge.net_t[idx].i_h = copy.deepcopy(model.vanilla_hedge.net_t[0].i_h)
            model.vanilla_hedge.net_t[idx].h_h = copy.deepcopy(model.vanilla_hedge.net_t[0].h_h)
        checkpoint_str= "NSDE_test_LSV_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day) 
        checkpoint=torch.load(checkpoint_str)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)  
        hedges_straddle = torch.zeros_like(realised_prices).to(device=device).float()
        hedges_lookback = torch.zeros_like(realised_prices).to(device=device).float()
    if cal_day>0:
        checkpoint_str= "NSDE_test_hedge_straddle_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day-1) 
        checkpoint=torch.load(checkpoint_str)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        if torch.abs(torch.tanh(model.rho.detach().cpu()))>0.99:
        	model.rho=torch.nn.Parameter(model.rho.detach().cpu()*0.25)
        hedges_straddle = checkpoint['past_hedges']
        checkpoint_str= "NSDE_test_hedge_lookback_seed_{}_cal_day_{}.pth.tar".format(seed,cal_day-1) 
        checkpoint=torch.load(checkpoint_str)
        hedges_lookback = checkpoint['past_hedges']

    print('hedges_straddle', hedges_straddle)
    print('hedges_lookback', hedges_lookback)
    np.random.seed(seed) # fix seed for reproducibility
    z_val = np.random.normal(size=(MC_samples_price, max(n_steps,n_steps_exotic)))
    z_val_var = np.random.normal(size=(MC_samples_var, max(n_steps,n_steps_exotic)))
    np.random.seed(1234567) # fixed seed for test set model is evaluated after calibration completes 
    z_test = np.random.normal(size=(MC_samples_price, max(n_steps,n_steps_exotic)))
    z_test_var = np.random.normal(size=(MC_samples_var, max(n_steps,n_steps_exotic)))
    
    z_val = torch.tensor(z_val).to(device=device).float()
    z_val_var = torch.tensor(z_val_var).to(device=device).float()
    
    #  Random samples used to generate paths used for testing
    z_test = torch.tensor(z_test).to(device=device).float()
    z_test_var = torch.tensor(z_test_var).to(device=device).float()

    CONFIG_SDE = {"batch_size":50000,
              "batch_size_hedge":10000,
              "n_epochs":100,
              "cal_day":cal_day,
              "gradient_count":gradient_count,
              "gradient_count_exotic":gradient_count_exotic,    
              "initial_price":S0,
              "maturities_daily":maturities_daily,
              "maturities":maturities,
              "maturity_exotic_daily": maturity_exotic_daily,      
              "maturity_exotic": maturity_exotic,    
              "maturity_values":maturity_values,
              "hedges_straddle": hedges_straddle,
              "hedges_lookback": hedges_lookback,
              "learning_rate": 0.0001,
              "interest_rate":rate,
              "n_maturities":n_maturities,
              "n_strikes":n_strikes,
              "strikes":strikes,
              "timegrid_daily":timegrid_daily,
              "timegrid_exotic":timegrid_exotic,
              "timegrid":timegrid,
              "target_data":data,
              "seed":seed}
                                        
    model_SDE = train_nsde(model, z_val,z_val_var,z_test,z_test_var, CONFIG_SDE)
  #  model_SDE = model # if model has been trained already 
    
    CONFIG_Exotic_Hedge = {"batch_size":50000,
              "batch_size_hedge":10000,
              "n_epochs":50,
              "cal_day":cal_day,
              "gradient_count":gradient_count,
              "gradient_count_exotic":gradient_count_exotic,             
              "initial_price":S0,
              "maturities_daily":maturities_daily,
              "maturities":maturities,
              "maturity_exotic_daily": maturity_exotic_daily,      
              "maturity_exotic": maturity_exotic,   
              "maturity_values":maturity_values,
              "hedges_straddle": hedges_straddle,
              "hedges_lookback": hedges_lookback,             
              "learning_rate": 0.0001,
              "interest_rate":rate,
              "n_maturities":n_maturities,
              "n_strikes":n_strikes,
              "strikes":strikes,
              "timegrid_daily":timegrid_daily,
              "timegrid":timegrid,
              "timegrid_exotic":timegrid_exotic,             
              "target_data":data,
              "seed":seed}
    
    # After SDE is trained to IV data, learn hedging strategies for lookback and ATM straddle
    
    parser = argparse.ArgumentParser()                     
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--MC_samples_price',type=int,default=MC_samples_price)
    parser.add_argument('--MC_samples_var',type=int,default=MC_samples_var)
    parser.add_argument('--tamed_Euler', action='store_true', default=False)
    parser.add_argument('--hedge_exotic',action='store_true', default=True)
    parser.add_argument('--lookback_exotic',action='store_true', default=True)
    parser.add_argument('--straddle_exotic',action='store_true', default=False)
    parser.add_argument('--dynamic_weight_update',action='store_true', default=True)
    args = parser.parse_args()                    
    
    model_SDE_hedge_1 = train_nsde(model_SDE, z_val,z_val_var,z_test,z_test_var, CONFIG_Exotic_Hedge)
     
    parser = argparse.ArgumentParser()    
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--MC_samples_price',type=int,default=MC_samples_price)
    parser.add_argument('--MC_samples_var',type=int,default=MC_samples_var)
    parser.add_argument('--tamed_Euler', action='store_true', default=False)
    parser.add_argument('--hedge_exotic',action='store_true', default=True) 
    parser.add_argument('--lookback_exotic',action='store_true', default=False)
    parser.add_argument('--straddle_exotic',action='store_true', default=True)
    parser.add_argument('--dynamic_weight_update',action='store_true', default=True)
    args = parser.parse_args() 
                         
    model_SDE_hedge_2 = train_nsde(model_SDE_hedge_1, z_val,z_val_var,z_test,z_test_var, CONFIG_Exotic_Hedge)
    