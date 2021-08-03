%load('date_1621-1650.mat')
%load('allS_new.mat')
load('model_IV_fit_previous.mat')
load('model_IV_fit.mat')


plot_neural_SDE_fit_day_zero(GOOG_cal_idx(1),GOOG_date,maturities(1,:),GOOG_S0(1),IV_blend_smooth(1,:,:),model_IV_fit(1,:,:), K(1,:,:), GOOG_day,GOOG_month,GOOG_year, number_strikes_for_each_maturity)
for i = 2:n
 maturities_temp =maturities(i,:);   
 if i>10
    maturities_temp(2)=maturities_temp(2)+1; 
 end    
 plot_neural_SDE_fit(GOOG_cal_idx(i),GOOG_date,maturities_temp,GOOG_S0(i),IV_blend_smooth(i,:,:),model_IV_fit(i,:,:),model_IV_fit_previous(i,:,:), K(i,:,:), GOOG_day,GOOG_month,GOOG_year,number_strikes_for_each_maturity)
end


function [] = plot_neural_SDE_fit(day,GOOG_date,maturities,GOOG_S0,IV_blend,IV_model,IV_model_previous, Strikes, GOOG_day,GOOG_month,GOOG_year,number_strikes_for_each_maturity)

for i=1:length(maturities)
[Iday,~]=find(GOOG_date==day);



iv_target = zeros(1,number_strikes_for_each_maturity);
iv_model = zeros(1,number_strikes_for_each_maturity);
iv_model_previous = zeros(1,number_strikes_for_each_maturity);
strikes_sorted = zeros(1,number_strikes_for_each_maturity);

iv_target(1,:) = IV_blend(1,i,:);
iv_model(1,:) = IV_model(1,i,:);
iv_model_previous(1,:) = IV_model_previous(1,i,:);
strikes_sorted(1,:) = Strikes(1,i,:);


            
   hold off
   plot(strikes_sorted,iv_model, strikes_sorted,iv_model_previous,'LineWidth',3,'LineWidth',3)
   hold on
   xline(GOOG_S0,'--b');
   plot(strikes_sorted,iv_target,'--rs','LineWidth',3,...
                       'MarkerEdgeColor','k',...
                       'MarkerFaceColor','g',...
                       'MarkerSize',10)
   
   xlabel('Strike')
   ylabel('Implied Vol.')
   legend('IV Model','IV Previous Date Model','ATM','IV Target','Location','Best')
   plot_title = strcat('Maturity: T=', string(maturities(i)) , ' (d), ',' Date:  ', string(GOOG_day(Iday(1))),'.', string(GOOG_month(Iday(1))),'.' ,string(GOOG_year(Iday(1))));
   save_string = strcat('Model_Fit_Date',string(GOOG_day(Iday(1))),string(GOOG_month(Iday(1))),string(GOOG_year(Iday(1))),'Maturity',string(maturities(i)));
   title(plot_title);
   saveas(gcf,save_string,'png');
   delete(gcf);
   hold off

end
end

function [] = plot_neural_SDE_fit_day_zero(day,GOOG_date,maturities,GOOG_S0,IV_blend,IV_model, Strikes, GOOG_day,GOOG_month,GOOG_year,number_strikes_for_each_maturity)

for i=1:length(maturities)
[Iday,~]=find(GOOG_date==day);

iv_target = zeros(1,number_strikes_for_each_maturity);
iv_model = zeros(1,number_strikes_for_each_maturity);
strikes_sorted = zeros(1,number_strikes_for_each_maturity);

iv_target(1,:) = IV_blend(1,i,:);
iv_model(1,:) = IV_model(1,i,:);
strikes_sorted(1,:) = Strikes(1,i,:);


            
   hold off
   plot(strikes_sorted,iv_model,'LineWidth',3)
   hold on
   xline(GOOG_S0,'--b');
   plot(strikes_sorted,iv_target,'--rs','LineWidth',3,...
                       'MarkerEdgeColor','k',...
                       'MarkerFaceColor','g',...
                       'MarkerSize',10)
   
   xlabel('Strike')
   ylabel('Implied Vol.')
   legend('IV Model','ATM','IV Target','Location','Best')
   plot_title = strcat('Maturity: T=', string(maturities(i)) , ' (d), ',' Date:  ', string(GOOG_day(Iday(1))),'.', string(GOOG_month(Iday(1))),'.' ,string(GOOG_year(Iday(1))));
   save_string = strcat('Model_Fit_Date',string(GOOG_day(Iday(1))),string(GOOG_month(Iday(1))),string(GOOG_year(Iday(1))),'Maturity',string(maturities(i)));
   title(plot_title);
   saveas(gcf,save_string,'png');
   delete(gcf);
   hold off

end
end
