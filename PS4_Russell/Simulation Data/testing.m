%% My version of lognormal draws from alpha_i
% draws = 1+lognrnd(0,1, [length(alphas(:)),1]);
draws = 1+lognrnd(0,1, [length(alphas(:)),1]);

[fp, xfp] = kde(draws); 
plot(xfp,fp,"-")
%% 
[fpa, xfpa] = kde(alphas(:)); 

hold on
plot(xfpa,fpa,"-")

shares_flat=shares(:);