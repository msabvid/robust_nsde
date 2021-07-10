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
Day_idx=735342:1:(735342+29);
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
VOL=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
K=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
Delta=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
ASK=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);
BID=zeros(n,number_maturities_for_each_cal_day,number_strikes_for_each_maturity);

for i = 1:n
    [maturities(i,:),rates(i)]= get_data_under_30days(GOOG_cal_idx(i),GOOG_date,GOOG_mat,GOOG_rate);
    [IV(i,:,:),VOL(i,:,:),K(i,:,:),Delta(i,:,:),ASK(i,:,:),BID(i,:,:)] = get_price_data(GOOG_cal_idx(i),GOOG_date,maturities(i,:),GOOG_S0(i),GOOG_mat,GOOG_IV,GOOG_VOL,GOOG_K,GOOG_BID,GOOG_ASK,GOOG_delta,number_strikes_for_each_maturity,GOOG_day,GOOG_month,GOOG_year,number_maturities_for_each_cal_day);
end

function [IV,VOL,K,Delta,ASK,BID] = get_price_data(day,GOOG_date,maturities,GOOG_S0,GOOG_mat,GOOG_IV,GOOG_VOL,GOOG_K,GOOG_BID,GOOG_ASK,GOOG_delta,number_strikes_for_each_maturity,GOOG_day,GOOG_month,GOOG_year,number_maturities_for_each_cal_day)

IV=zeros(5,number_strikes_for_each_maturity);
VOL=zeros(5,number_strikes_for_each_maturity);
K=zeros(5,number_strikes_for_each_maturity);
Delta=zeros(5,number_strikes_for_each_maturity);
ASK=zeros(5,number_strikes_for_each_maturity);
BID=zeros(5,number_strikes_for_each_maturity);

for i=1:length(maturities)
%GOOG_S0
%GOOG_K(1:100)
[Iday,~]=find(GOOG_date==day);
[nonzero_volume_idx,~]=find(GOOG_VOL>0);
[Call_OTM_idx_1,~] = find(GOOG_delta>0); 
[Put_OTM_idx_1,~] = find(GOOG_delta<0);
[Call_OTM_idx_2,~] = find(GOOG_K>GOOG_S0);
[Put_OTM_idx_2,~] = find(GOOG_K<GOOG_S0);
[Call_OTM_idx,~,~] = intersect(Call_OTM_idx_1,Call_OTM_idx_2);
[Put_OTM_idx,~,~] = intersect(Put_OTM_idx_1,Put_OTM_idx_2);
    
   % first select OTM options only
   [Imat_temp,~]=find(GOOG_mat==maturities(i));
   [Imat,~,~]=intersect(Imat_temp,Iday);
  % GOOG_date(Iday)
  % [Imat,~]=find(GOOG_date(Iday)==maturities(i))
   [Imat_idx,~,~] = intersect(nonzero_volume_idx,Imat);
   [Call_idx,~,~] = intersect(Call_OTM_idx,Imat_idx);
   [Put_idx,~,~] = intersect(Put_OTM_idx,Imat_idx);
   OTM_idx=[Call_idx;Put_idx];
   GOOG_delta_temp = GOOG_delta(OTM_idx); % something needs fixed here its picking wrong data 
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
   bid = GOOG_BID_temp(idx_volume_sorted);
   ask = GOOG_ASK_temp(idx_volume_sorted);
   [strikes_sorted,idx_sorted]=sort(strikes,'ascend');
   iv_sorted=iv(idx_sorted);
   delta_sorted=delta(idx_sorted);
   vol_sorted=vol(idx_sorted);
   ask_sorted=ask(idx_sorted);
   bid_sorted=bid(idx_sorted);
   IV(i,:)=iv_sorted;
   plot(iv_sorted)
   plot(strikes_sorted,iv_sorted,'--rs','LineWidth',2,...
                       'MarkerEdgeColor','k',...
                       'MarkerFaceColor','g',...
                       'MarkerSize',10)
   xlabel('Strike')
   ylabel('Implied Vol.')
   legend('Market IV')
   plot_title = strcat('Maturity: T=', string(maturities(i)) , 'days, ',' Date:  ', string(GOOG_day(Iday(1))),'.', string(GOOG_month(Iday(1))),'.' ,string(GOOG_year(Iday(1))));
   save_string = strcat('Maturity',string(maturities(i)),'Date',string(GOOG_day(Iday(1))),string(GOOG_month(Iday(1))),string(GOOG_year(Iday(1))));
   title(plot_title);
   saveas(gcf,save_string,'png');
   VOL(i,:)=vol_sorted;
   BID(i,:)=bid_sorted;
   ASK(i,:)=ask_sorted;
   K(i,:)=strikes_sorted;
   Delta(i,:)=delta_sorted;
end    
end 

function [maturities,rates] = get_data_under_30days(day,GOOG_date,GOOG_mat,GOOG_rate)
%GOOG_day
%day
[Iday,~]=find(GOOG_date==day);
[rate_day,~]=find(GOOG_rate(:,1)==day);
rates = GOOG_rate(rate_day,2);
maturities_temp=unique(GOOG_mat(Iday));
maturities = zeros(5,1); 
for i=1:length(maturities)
    maturities(i)=maturities_temp(i);
end
end