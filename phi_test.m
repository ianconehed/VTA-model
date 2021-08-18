clear all
close all


u = -20:.01:20;

u_c1 = -18;
v1 = 1;
theta1 = -20;

u_c2 = -4.5;
v2 = 1;
theta2 = -5;

a = 1;


y = u;
old_y = u;
new_y = u;
for i = 1:length(u)
    old_y(i) = phi_2(u(i),v1,theta1,u_c1);
    new_y(i) = phi_3(u(i),1);
    y(i) = phi_3_smooth(u(i),1);
end

plot(u,y)
hold on
plot(u,new_y,'r--')