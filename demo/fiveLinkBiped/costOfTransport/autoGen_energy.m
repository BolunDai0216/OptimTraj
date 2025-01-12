function [KE,PE] = autoGen_energy(q1,q2,q3,q4,q5,dq1,dq2,dq3,dq4,dq5,m1,m2,m3,m4,m5,I1,I2,I3,I4,I5,l1,l2,l3,l4,c1,c2,c3,c4,c5,g)
%AUTOGEN_ENERGY
%    [KE,PE] = AUTOGEN_ENERGY(Q1,Q2,Q3,Q4,Q5,DQ1,DQ2,DQ3,DQ4,DQ5,M1,M2,M3,M4,M5,I1,I2,I3,I4,I5,L1,L2,L3,L4,C1,C2,C3,C4,C5,G)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    12-Jul-2021 22:21:17

t2 = cos(q1);
t3 = cos(q2);
t4 = cos(q3);
t5 = cos(q4);
t6 = cos(q5);
t7 = sin(q1);
t8 = sin(q2);
t9 = sin(q3);
t10 = sin(q4);
t11 = dq1.^2;
t12 = c1.*t2;
t13 = c2.*t3;
t14 = c3.*t4;
t15 = l1.*t2;
t16 = l2.*t3;
t17 = l3.*t4;
t20 = dq1.*l1.*t7;
t21 = dq2.*l2.*t8;
t18 = dq1.*t15;
t19 = dq2.*t16;
t22 = -t15;
t23 = -t16;
t24 = -t20;
t25 = t12+t22;
et1 = (I1.*t11)./2.0+(m2.*((t18-dq2.*(t13+t23)).^2+(t20-dq2.*(c2.*t8-l2.*t8)).^2))./2.0+(m5.*((t20+t21-c5.*dq5.*sin(q5)-dq4.*l4.*t10).^2+(t18+t19-c5.*dq5.*t6-dq4.*l4.*t5).^2))./2.0+(I2.*dq2.^2)./2.0+(I3.*dq3.^2)./2.0+(I4.*dq4.^2)./2.0+(I5.*dq5.^2)./2.0+(m3.*((t18+t19-dq3.*(t14-t17)).^2+(t20+t21-dq3.*(c3.*t9-l3.*t9)).^2))./2.0+(m4.*((t18+t19-c4.*dq4.*t5).^2+(t20+t21-c4.*dq4.*t10).^2))./2.0;
et2 = (m1.*(t11.*(c1.*t7-l1.*t7).^2+t11.*t25.^2))./2.0;
KE = et1+et2;
if nargout > 1
    PE = g.*m2.*(-t13+t15+t16)-g.*m1.*t25+g.*m4.*(t15+t16-c4.*t5)+g.*m3.*(-t14+t15+t16+t17)+g.*m5.*(t15+t16-c5.*t6-l4.*t5);
end
