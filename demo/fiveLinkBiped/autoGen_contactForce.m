function [Fx,Fy] = autoGen_contactForce(q1,q2,q3,q4,q5,dq1,dq2,dq3,dq4,dq5,ddq1,ddq2,ddq3,ddq4,ddq5,m1,m2,m3,m4,m5,l1,l2,l3,l4,c1,c2,c3,c4,c5,g)
%AUTOGEN_CONTACTFORCE
%    [FX,FY] = AUTOGEN_CONTACTFORCE(Q1,Q2,Q3,Q4,Q5,DQ1,DQ2,DQ3,DQ4,DQ5,DDQ1,DDQ2,DDQ3,DDQ4,DDQ5,M1,M2,M3,M4,M5,L1,L2,L3,L4,C1,C2,C3,C4,C5,G)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    12-Jul-2021 22:27:33

t2 = cos(q1);
t3 = cos(q2);
t4 = cos(q3);
t5 = cos(q4);
t6 = cos(q5);
t7 = sin(q1);
t8 = sin(q2);
t9 = sin(q3);
t10 = sin(q4);
t11 = sin(q5);
t12 = dq1.^2;
t13 = dq2.^2;
t14 = dq3.^2;
t15 = dq4.^2;
t16 = dq5.^2;
Fx = c1.*ddq1.*m1.*t2+c2.*ddq2.*m2.*t3+c3.*ddq3.*m3.*t4+c4.*ddq4.*m4.*t5+c5.*ddq5.*m5.*t6-ddq1.*l1.*m1.*t2-ddq1.*l1.*m2.*t2-ddq1.*l1.*m3.*t2-ddq1.*l1.*m4.*t2-ddq1.*l1.*m5.*t2-ddq2.*l2.*m2.*t3-ddq2.*l2.*m3.*t3-ddq2.*l2.*m4.*t3-ddq2.*l2.*m5.*t3-ddq3.*l3.*m3.*t4+ddq4.*l4.*m5.*t5-c1.*m1.*t7.*t12-c2.*m2.*t8.*t13-c3.*m3.*t9.*t14-c4.*m4.*t10.*t15-c5.*m5.*t11.*t16+l1.*m1.*t7.*t12+l1.*m2.*t7.*t12+l1.*m3.*t7.*t12+l1.*m4.*t7.*t12+l1.*m5.*t7.*t12+l2.*m2.*t8.*t13+l2.*m3.*t8.*t13+l2.*m4.*t8.*t13+l2.*m5.*t8.*t13+l3.*m3.*t9.*t14-l4.*m5.*t10.*t15;
if nargout > 1
    et1 = g.*m1+g.*m2+g.*m3+g.*m4+g.*m5+c1.*ddq1.*m1.*t7+c2.*ddq2.*m2.*t8+c3.*ddq3.*m3.*t9+c4.*ddq4.*m4.*t10+c5.*ddq5.*m5.*t11-ddq1.*l1.*m1.*t7-ddq1.*l1.*m2.*t7-ddq1.*l1.*m3.*t7-ddq1.*l1.*m4.*t7-ddq1.*l1.*m5.*t7-ddq2.*l2.*m2.*t8-ddq2.*l2.*m3.*t8-ddq2.*l2.*m4.*t8-ddq2.*l2.*m5.*t8-ddq3.*l3.*m3.*t9+ddq4.*l4.*m5.*t10+c1.*m1.*t2.*t12+c2.*m2.*t3.*t13+c3.*m3.*t4.*t14+c4.*m4.*t5.*t15+c5.*m5.*t6.*t16-l1.*m1.*t2.*t12-l1.*m2.*t2.*t12-l1.*m3.*t2.*t12-l1.*m4.*t2.*t12-l1.*m5.*t2.*t12-l2.*m2.*t3.*t13-l2.*m3.*t3.*t13-l2.*m4.*t3.*t13;
    et2 = -l2.*m5.*t3.*t13-l3.*m3.*t4.*t14+l4.*m5.*t5.*t15;
    Fy = et1+et2;
end
