clc; clear;
disp('Creating variables and derivatives...')

%%%% Absolute orientation (angle) of each link
q1 = sym('q1', 'real');
q2 = sym('q2','real');
q3 = sym('q3','real');
q4 = sym('q4','real');
q5 = sym('q5','real');

%%%% Absolute angular rate of each link
dq1 = sym('dq1','real');
dq2 = sym('dq2','real');
dq3 = sym('dq3','real');
dq4 = sym('dq4','real');
dq5 = sym('dq5','real');

%%%% Absolute angular acceleration of each linke
ddq1 = sym('ddq1','real');
ddq2 = sym('ddq2','real');
ddq3 = sym('ddq3','real');
ddq4 = sym('ddq4','real');
ddq5 = sym('ddq5','real');

%%%% Torques at each joint
u1 = sym('u1','real');  %Stance foot
u2 = sym('u2','real');   %Stance knee
u3 = sym('u3','real');   %Stance hip
u4 = sym('u4','real');   %Swing hip
u5 = sym('u5','real');   %Swing knee

%%%% Torques rate at each joint
du1 = sym('du1','real');  %Stance foot
du2 = sym('du2','real');   %Stance knee
du3 = sym('du3','real');   %Stance hip
du4 = sym('du4','real');   %Swing hip
du5 = sym('du5','real');   %Swing knee

%%%% Slack variables -- negative component of power
sn1 = sym('sn1','real');  %Stance foot
sn2 = sym('sn2','real');   %Stance knee
sn3 = sym('sn3','real');   %Stance hip
sn4 = sym('sn4','real');   %Swing hip
sn5 = sym('sn5','real');   %Swing knee

%%%% Slack variables -- positive component of power
sp1 = sym('sp1','real');  %Stance foot
sp2 = sym('sp2','real');   %Stance knee
sp3 = sym('sp3','real');   %Stance hip
sp4 = sym('sp4','real');   %Swing hip
sp5 = sym('sp5','real');   %Swing knee

%%%% Mass of each link
m1 = sym('m1','real');
m2 = sym('m2','real');
m3 = sym('m3','real');
m4 = sym('m4','real');
m5 = sym('m5','real');

%%%% Distance between parent joint and link center of mass
c1 = sym('c1','real');
c2 = sym('c2','real');
c3 = sym('c3','real');
c4 = sym('c4','real');
c5 = sym('c5','real');

%%%% Length of each link
l1 = sym('l1','real');
l2 = sym('l2','real');
l3 = sym('l3','real');
l4 = sym('l4','real');
l5 = sym('l5','real');

%%%% Moment of inertia of each link about its own center of mass
I1 = sym('I1','real');
I2 = sym('I2','real');
I3 = sym('I3','real');
I4 = sym('I4','real');
I5 = sym('I5','real');

g = sym('g','real'); % Gravity
Fx = sym('Fx','real');   %Horizontal contact force at stance foot
Fy = sym('Fy','real');   %Vertical contact force at stance foot
empty = sym('empty','real');   %Used for vectorization, user should pass a vector of zeros
t = sym('t','real');  %dummy continuous time

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                Set up coordinate system and unit vectors                %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

i = sym([1;0]);   %Horizontal axis
j = sym([0;1]);   %Vertical axis

e1 = cos(q1)*(j) + sin(q1)*(-i);  %unit vector from P0 -> P1, (contact point to stance knee)
e2 = cos(q2)*(j) + sin(q2)*(-i);  %unit vector from P1 -> P2, (stance knee to hip)
e3 = cos(q3)*(j) + sin(q3)*(-i);  %unit vector from P2 -> P3, (hip to shoulders);
e4 = -cos(q4)*(j) - sin(q4)*(-i);  %unit vector from P2 -> P4, (hip to swing knee);
e5 = -cos(q5)*(j) - sin(q5)*(-i);  %unit vector from P4 -> P5, (swing knee to swing foot);

P0 = 0*i + 0*j;   %stance foot = Contact point = origin
P1 = P0 + l1*e1;  %stance knee
P2 = P1 + l2*e2;  %hip
P3 = P2 + l3*e3;  %shoulders
P4 = P2 + l4*e4;  %swing knee
P5 = P4 + l5*e5;  %swing foot

G1 = P1 - c1*e1;  % CoM stance leg tibia
G2 = P2 - c2*e2;  % CoM stance leg febur
G3 = P3 - c3*e3;  % CoM torso
G4 = P2 + c4*e4;  % CoM swing leg femur
G5 = P4 + c5*e5;  % CoM swing leg tibia
G = (m1*G1 + m2*G2 + m3*G3 + m4*G4 + m5*G5)/(m1+m2+m3+m4+m5);  %Center of mass for entire robot

%%%% Define a function for doing '2d' cross product: dot(a x b, k)
cross2d = @(a,b)(a(1)*b(2) - a(2)*b(1));

%%%% Weight of each link:
w1 = -m1*g*j;
w2 = -m2*g*j;
w3 = -m3*g*j;
w4 = -m4*g*j;
w5 = -m5*g*j;

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                             Derivatives                                 %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

q = [q1;q2;q3;q4;q5];
dq = [dq1;dq2;dq3;dq4;dq5];
ddq = [ddq1;ddq2;ddq3;ddq4;ddq5];
u = [u1;u2;u3;u4;u5];
du = [du1;du2;du3;du4;du5];
sn = [sn1;sn2;sn3;sn4;sn5];
sp = [sp1;sp2;sp3;sp4;sp5];
z = [t;q;dq;u;du;sn;sp];   % time-varying vector of inputs

% Neat trick to compute derivatives using the chain rule
derivative = @(in)( jacobian(in,[q;dq;u])*[dq;ddq;du] );

% Velocity of the swing foot (used for step constraints)
dP5 = derivative(P5);

% Compute derivatives for the CoM of each link:
dG1 = derivative(G1);  ddG1 = derivative(dG1);
dG2 = derivative(G2);  ddG2 = derivative(dG2);
dG3 = derivative(G3);  ddG3 = derivative(dG3);
dG4 = derivative(G4);  ddG4 = derivative(dG4);
dG5 = derivative(G5);  ddG5 = derivative(dG5);
dG = derivative(G);  ddG = derivative(dG);