function [P,Gvec] = autoGen_getPoints(q1,q2,q3,q4,q5,l1,l2,l3,l4,l5,c1,c2,c3,c4,c5)
%AUTOGEN_GETPOINTS
%    [P,GVEC] = AUTOGEN_GETPOINTS(Q1,Q2,Q3,Q4,Q5,L1,L2,L3,L4,L5,C1,C2,C3,C4,C5)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    12-Jul-2021 22:21:18

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
t12 = l1.*t2;
t13 = l2.*t3;
t14 = l3.*t4;
t15 = l4.*t5;
t16 = l1.*t7;
t17 = l2.*t8;
t18 = l3.*t9;
t19 = l4.*t10;
t20 = -t15;
t21 = -t16;
t22 = -t17;
t23 = -t18;
P = [t21;t12;t21+t22;t12+t13;t21+t22+t23;t12+t13+t14;t19+t21+t22;t12+t13+t20;t19+t21+t22+l5.*t11;t12+t13+t20-l5.*t6];
if nargout > 1
    Gvec = [t21+c1.*t7;t12-c1.*t2;t21+t22+c2.*t8;t12+t13-c2.*t3;t21+t22+t23+c3.*t9;t12+t13+t14-c3.*t4;t21+t22+c4.*t10;t12+t13-c4.*t5;t19+t21+t22+c5.*t11;t12+t13+t20-c5.*t6];
end
