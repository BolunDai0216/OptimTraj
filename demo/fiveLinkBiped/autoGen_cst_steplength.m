function [ceq,ceqJac] = autoGen_cst_steplength(q1m,q2m,q4m,q5m,l1,l2,l4,l5,stepLength)
%AUTOGEN_CST_STEPLENGTH
%    [CEQ,CEQJAC] = AUTOGEN_CST_STEPLENGTH(Q1M,Q2M,Q4M,Q5M,L1,L2,L4,L5,STEPLENGTH)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    12-Jul-2021 22:27:33

t2 = cos(q1m);
t3 = cos(q2m);
t4 = cos(q4m);
t5 = cos(q5m);
t6 = sin(q1m);
t7 = sin(q2m);
t8 = sin(q4m);
t9 = sin(q5m);
t10 = l1.*t2;
t11 = l2.*t3;
t12 = l4.*t4;
t13 = l5.*t5;
t14 = l1.*t6;
t15 = l2.*t7;
t16 = l4.*t8;
t17 = l5.*t9;
t18 = -t14;
t19 = -t15;
ceq = [-stepLength+t16+t17+t18+t19;t10+t11-t12-t13];
if nargout > 1
    ceqJac = reshape([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-t10,t18,-t11,t19,0.0,0.0,t12,t16,t13,t17,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[2,22]);
end
