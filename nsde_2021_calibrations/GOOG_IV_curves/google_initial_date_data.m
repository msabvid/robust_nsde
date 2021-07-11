%load('date_1621-1650.mat')
[I,~]=find(data.secid==121812); % find ID of Google
GOOG_IV=data.impl_volatility(I);
GOOG_VOL=data.volume(I);
GOOG_K=data.strike_price(I)/1000;
GOOG_BID=data.best_bid(I);
GOOG_ASK=data.best_offer(I);
GOOG_mat=data.maturity(I);
GOOG_date=data.date(I);
GOOG_day=data.d(I);
GOOG_rate=itrate;
GOOG_month=data.m(I);
GOOG_year=data.y(I);
GOOG_delta=data.delta(I);
%load('allS_new.mat')
[I,~]=find(S(:,2)==121812);
GOOG_Sday=S(I,1);
GOOG_S=S(I,3);
Day_idx=735342:1:(735342+30);
[GOOG_S0entry,GOOG_cal_days] = find(GOOG_Sday==Day_idx);
GOOG_S0=GOOG_S(GOOG_S0entry);
GOOG_cal_idx=GOOG_cal_days+735342-1;
size(GOOG_S0)

number_maturities_for_each_cal_day=5;
n = length(GOOG_cal_days);
maturities = zeros(n,number_maturities_for_each_cal_day);
rates = zeros(n,1);

number_strikes_for_each_maturity=25;

IV=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
IV_blend=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
IV_blend_smooth=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
VOL=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
K=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
Delta=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
ASK=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
BID=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);

for i = 1:n
    [maturities(i,:),rates(i)]= get_data_under_30days(GOOG_cal_idx(i),GOOG_date,GOOG_mat,GOOG_rate,number_maturities_for_each_cal_day);
end

GOOG_timegrids=zeros(n,max(max(maturities)));
GOOG_timegrids(1,1:length(GOOG_cal_days))=GOOG_cal_days;

for i=2:n
    Day_idx_temp=735342+GOOG_cal_days(i):1:(735342++GOOG_cal_days(i)+maturities(i,number_maturities_for_each_cal_day));
    [~,GOOG_cal_days_temp] = find(GOOG_Sday==Day_idx_temp);
    GOOG_timegrids(i,1:length(GOOG_cal_days_temp))=GOOG_cal_days_temp;
end

for i = 1:n
 [IV(i,:,:),IV_blend(i,:,:),IV_blend_smooth(i,:,:),VOL(i,:,:),K(i,:,:),Delta(i,:,:),ASK(i,:,:),BID(i,:,:)] = get_price_data(GOOG_cal_idx(i),GOOG_date,maturities(i,:),GOOG_S0(i),GOOG_mat,GOOG_IV,GOOG_VOL,GOOG_K,GOOG_BID,GOOG_ASK,GOOG_delta,number_strikes_for_each_maturity,GOOG_day,GOOG_month,GOOG_year,number_maturities_for_each_cal_day);
end

save('GOOG_data.mat')
save('IV_target.mat','IV_blend_smooth')
save('IV_raw.mat','IV')
save('ASK_price.mat','ASK')
save('BID_price.mat','BID')
save('GOOG_S0.mat','GOOG_S0')
save('strikes.mat','K')
save('maturities.mat','maturities')
save('rates.mat','rates')
save('voltume.mat','VOL')


save('MyArray.mat','MyArray')
function [Kmin,Kmax] = get_maxminK(strikes_sorted,S0)
idx_min=1;
idx_max=0;
strike_count=length(strikes_sorted);
    while abs(strikes_sorted(idx_min)-S0)>0.02*S0 && idx_min<strike_count
            idx_min=idx_min+1;
    end
    while abs(strikes_sorted(strike_count-idx_max)-S0)>0.02*S0 && idx_max<strike_count-1
            idx_max=idx_max+1;
    end

Kmin=strikes_sorted(idx_min);
Kmax=strikes_sorted(strike_count-idx_max);
    
end

function [IV,IV_blend,IV_blend_smooth,VOL,K,Delta,ASK,BID] = get_price_data(day,GOOG_date,maturities,GOOG_S0,GOOG_mat,GOOG_IV,GOOG_VOL,GOOG_K,GOOG_BID,GOOG_ASK,GOOG_delta,number_strikes_for_each_maturity,GOOG_day,GOOG_month,GOOG_year,number_maturities_for_each_cal_day)

IV=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
IV_blend=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
IV_blend_smooth=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
VOL=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
K=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
Delta=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
ASK=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
BID=zeros(number_maturities_for_each_cal_day,number_strikes_for_each_maturity);

for i=1:length(maturities)
[Iday,~]=find(GOOG_date==day);
[nonzero_volume_idx,~]=find(GOOG_VOL>0);
[Call_OTM_idx_1,~] = find(GOOG_delta>0); 
[Put_OTM_idx_1,~] = find(GOOG_delta<0);
[Call_OTM_idx_2,~] = find(GOOG_K>GOOG_S0);
[Put_OTM_idx_2,~] = find(GOOG_K<GOOG_S0);
[Call_OTM_idx,~,~] = intersect(Call_OTM_idx_1,Call_OTM_idx_2);
[Put_OTM_idx,~,~] = intersect(Put_OTM_idx_1,Put_OTM_idx_2);

[Call_ITM_idx_1,~] = find(GOOG_delta>0); 
[Put_ITM_idx_1,~] = find(GOOG_delta<0);
[Call_ITM_idx_2,~] = find(GOOG_K<GOOG_S0);
[Put_ITM_idx_2,~] = find(GOOG_K>GOOG_S0);
[Call_ITM_idx,~,~] = intersect(Call_ITM_idx_1,Call_ITM_idx_2);
[Put_ITM_idx,~,~] = intersect(Put_ITM_idx_1,Put_ITM_idx_2);
    
   % first select OTM options only
   [Imat_temp,~]=find(GOOG_mat==maturities(i));
   [Imat,~,~]=intersect(Imat_temp,Iday);
   [Imat_idx,~,~] = intersect(nonzero_volume_idx,Imat);
   [Call_idx_OTM,~,~] = intersect(Call_OTM_idx,Imat_idx);
   [Put_idx_OTM,~,~] = intersect(Put_OTM_idx,Imat_idx);
   OTM_idx=[Call_idx_OTM;Put_idx_OTM];
   [Call_idx_ITM,~,~] = intersect(Call_ITM_idx,Imat_idx);
   [Put_idx_ITM,~,~] = intersect(Put_ITM_idx,Imat_idx);
   ITM_idx=[Call_idx_ITM;Put_idx_ITM];
   
   GOOG_delta_temp = GOOG_delta(OTM_idx); 
   GOOG_K_temp = GOOG_K(OTM_idx);
   GOOG_VOL_temp = GOOG_VOL(OTM_idx);
   GOOG_IV_temp = GOOG_IV(OTM_idx);
   GOOG_BID_temp = GOOG_BID(OTM_idx);
   GOOG_ASK_temp = GOOG_ASK(OTM_idx);
   [~,idx_volume_sorted]=sort(GOOG_VOL_temp,'descend');
   while length(idx_volume_sorted)<number_strikes_for_each_maturity
       idx_volume_sorted = [idx_volume_sorted; idx_volume_sorted];
   end
   idx_volume_sorted = idx_volume_sorted(1:number_strikes_for_each_maturity);   
   delta = GOOG_delta_temp(idx_volume_sorted);
   strikes = GOOG_K_temp(idx_volume_sorted);
   vol = GOOG_VOL_temp(idx_volume_sorted);
   iv = GOOG_IV_temp(idx_volume_sorted);
   iv_blend = GOOG_IV_temp(idx_volume_sorted);
   bid = GOOG_BID_temp(idx_volume_sorted);
   ask = GOOG_ASK_temp(idx_volume_sorted);
   [strikes_sorted,idx_sorted]=sort(strikes,'ascend');
   iv_sorted=iv(idx_sorted);
   iv_blend_sorted=iv_blend(idx_sorted);
   delta_sorted=delta(idx_sorted);
   vol_sorted=vol(idx_sorted);
   ask_sorted=ask(idx_sorted);
   bid_sorted=bid(idx_sorted);
   [Kmin,Kmax]=get_maxminK(strikes_sorted,GOOG_S0);
   GOOG_K_temp = GOOG_K(ITM_idx);
   GOOG_IV_temp = GOOG_IV(ITM_idx);
   for j=1:length(strikes_sorted)
       if strikes_sorted(j)<Kmax && strikes_sorted(j)>Kmin
          [IV_ITM_idx,~] = find(GOOG_K_temp==strikes_sorted(j));
          if length(IV_ITM_idx)==1
             iv_itm_tmp = GOOG_IV_temp(IV_ITM_idx);
          else
             iv_itm_tmp = iv_sorted(j);
          end
          if Kmin>strikes_sorted(1) && Kmax<strikes_sorted(end)
            if strikes_sorted(j)>GOOG_S0 
                iv_blend_sorted(j)=(Kmax-strikes_sorted(j))/(Kmax-Kmin)*iv_itm_tmp+(strikes_sorted(j)-Kmin)/(Kmax-Kmin)*iv_sorted(j);
            elseif strikes_sorted(j)<GOOG_S0
                iv_blend_sorted(j)=(Kmax-strikes_sorted(j))/(Kmax-Kmin)*iv_sorted(j)+(strikes_sorted(j)-Kmin)/(Kmax-Kmin)*iv_itm_tmp;
            end 
          end  
       end   
   end
   strikes_sorted_unique = unique(strikes_sorted);
   iv_blend_smooth=iv_blend_sorted;
   [~,iv_spaps] = spaps(strikes_sorted, iv_blend_sorted,0.0005);
   for k=1:length(iv_spaps)
       [idx_strike,~]=find(strikes_sorted==strikes_sorted_unique(k));
       iv_blend_smooth(idx_strike)=iv_spaps(k);
   end    
   IV(i,:)=iv_sorted;
   IV_blend(i,:)=iv_blend_sorted;
   IV_blend_smooth(i,:)=iv_blend_smooth;
   hold off
   plot(strikes_sorted,iv_sorted, strikes_sorted,iv_blend_sorted)
   hold on
   xline(GOOG_S0,'--b');
   plot(strikes_sorted,iv_blend_smooth,'--rs','LineWidth',2,...
                       'MarkerEdgeColor','k',...
                       'MarkerFaceColor','g',...
                       'MarkerSize',10)
   
   xlabel('Strike')
   ylabel('Implied Vol.')
   legend('IV Raw(Market)','IV Blend','ATM','IV Blend+Smooth','Location','Best')
   plot_title = strcat('Maturity: T=', string(maturities(i)) , 'days, ',' Date:  ', string(GOOG_day(Iday(1))),'.', string(GOOG_month(Iday(1))),'.' ,string(GOOG_year(Iday(1))));
   save_string = strcat('Date',string(GOOG_day(Iday(1))),string(GOOG_month(Iday(1))),string(GOOG_year(Iday(1))),'Maturity',string(maturities(i)));
   title(plot_title);
   saveas(gcf,save_string,'png');
   delete(gcf);
   hold off
   VOL(i,:)=vol_sorted;
   BID(i,:)=bid_sorted;
   ASK(i,:)=ask_sorted;
   K(i,:)=strikes_sorted;
   Delta(i,:)=delta_sorted;
end    
end 

function [maturities,rates] = get_data_under_30days(day,GOOG_date,GOOG_mat,GOOG_rate,number_maturities_for_each_cal_day)
[Iday,~]=find(GOOG_date==day);
[rate_day,~]=find(GOOG_rate(:,1)==day);
rates = GOOG_rate(rate_day,2);
maturities_temp=unique(GOOG_mat(Iday));
maturities = zeros(number_maturities_for_each_cal_day,1); 
for i=1:length(maturities)
    maturities(i)=maturities_temp(i);
end
end