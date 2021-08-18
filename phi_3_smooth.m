function y = phi_3_smooth(u,a)
 y = (u-a).*((u>a) & ~isnan(u));   
end