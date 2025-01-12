function [m,mi,f,fi,mz,mzi,mzd,fz,fzi,fzd] = autoGen_cst_heelStrike(q1p,q2p,q3p,q4p,q5p,q1m,q2m,q3m,q4m,q5m,dq1m,dq2m,dq3m,dq4m,dq5m,m1,m2,m3,m4,m5,I1,I2,I3,I4,I5,l1,l2,l3,l4,l5,c1,c2,c3,c4,c5,empty)
%AUTOGEN_CST_HEELSTRIKE
%    [M,MI,F,FI,MZ,MZI,MZD,FZ,FZI,FZD] = AUTOGEN_CST_HEELSTRIKE(Q1P,Q2P,Q3P,Q4P,Q5P,Q1M,Q2M,Q3M,Q4M,Q5M,DQ1M,DQ2M,DQ3M,DQ4M,DQ5M,M1,M2,M3,M4,M5,I1,I2,I3,I4,I5,L1,L2,L3,L4,L5,C1,C2,C3,C4,C5,EMPTY)

%    This function was generated by the Symbolic Math Toolbox version 8.7.
%    12-Jul-2021 22:27:30

t2 = cos(q1m);
t3 = cos(q2m);
t4 = cos(q3m);
t5 = cos(q4m);
t6 = cos(q5m);
t7 = cos(q1p);
t8 = cos(q2p);
t9 = cos(q3p);
t10 = cos(q4p);
t11 = cos(q5p);
t12 = sin(q1m);
t13 = sin(q2m);
t14 = sin(q3m);
t15 = sin(q4m);
t16 = sin(q5m);
t17 = sin(q1p);
t18 = sin(q2p);
t19 = sin(q3p);
t20 = sin(q4p);
t21 = sin(q5p);
t22 = I1.*dq1m;
t23 = I2.*dq2m;
t24 = I3.*dq3m;
t25 = I4.*dq4m;
t26 = c2.*m2;
t27 = c4.*m4;
t28 = l2.*m2;
t29 = l2.*m3;
t30 = l2.*m4;
t31 = l2.*m5;
t32 = l4.*m5;
t33 = c1.^2;
t34 = c2.^2;
t35 = c3.^2;
t36 = c4.^2;
t37 = c5.^2;
t38 = l1.^2;
t39 = l2.^2;
t40 = l3.^2;
t41 = l4.^2;
t48 = c1.*l1.*m1;
t49 = c1.*l2.*m1;
t61 = l1.*l2.*m1;
t68 = -I1;
t69 = -I2;
t70 = -I3;
t71 = -I4;
t72 = -I5;
t81 = -l1;
t82 = -l2;
t83 = -l3;
t84 = -l4;
t85 = -l5;
t86 = -q2m;
t87 = -q3m;
t88 = -q4m;
t89 = -q5m;
t90 = -q2p;
t91 = -q3p;
t92 = -q4p;
t93 = -q5p;
t115 = c3.*l3.*m3.*2.0;
t42 = c1.*t2;
t43 = c2.*t3;
t44 = c3.*t4;
t45 = c3.*t9;
t46 = c4.*t10;
t47 = c5.*t11;
t50 = l1.*t26;
t51 = l2.*t26;
t52 = l4.*t27;
t53 = l2.*t3;
t54 = l3.*t4;
t55 = l4.*t5;
t56 = l5.*t6;
t57 = l1.*t7;
t58 = l2.*t8;
t59 = l3.*t9;
t60 = l4.*t10;
t62 = c1.*t12;
t63 = c2.*t13;
t64 = c3.*t14;
t65 = c3.*t19;
t66 = c4.*t20;
t67 = c5.*t21;
t73 = l2.*t13;
t74 = l3.*t14;
t75 = l4.*t15;
t76 = l5.*t16;
t77 = l1.*t17;
t78 = l2.*t18;
t79 = l3.*t19;
t80 = l4.*t20;
t94 = dq1m.*l1.*t2;
t99 = c4.*dq4m.*t15;
t100 = -t22;
t101 = -t23;
t102 = -t24;
t103 = -t25;
t104 = dq1m.*l1.*t12;
t107 = -t26;
t108 = m1.*t33;
t109 = c2.*t26;
t110 = m3.*t35;
t111 = c4.*t27;
t112 = m5.*t37;
t113 = m3.*t40;
t114 = l4.*t32;
t118 = c4.*dq4m.*t5;
t119 = c1+t81;
t120 = c2+t82;
t121 = c3+t83;
t122 = c4+t84;
t123 = c5+t85;
t127 = -t49;
t131 = q1m+t86;
t132 = q1m+t87;
t133 = q1m+t88;
t134 = q2m+t87;
t135 = q1m+t89;
t136 = q2m+t88;
t137 = q2m+t89;
t138 = q3m+t88;
t139 = q3m+t89;
t140 = q4m+t89;
t141 = q1p+t90;
t142 = q1p+t91;
t143 = q1p+t92;
t144 = q2p+t91;
t145 = q1p+t93;
t146 = q2p+t92;
t147 = q2p+t93;
t148 = q4p+t93;
t198 = t32.*t84;
t199 = t27+t32;
t95 = dq2m.*t53;
t96 = dq3m.*t54;
t97 = dq2m.*t63;
t98 = dq3m.*t64;
t105 = dq2m.*t73;
t106 = dq3m.*t74;
t116 = dq2m.*t43;
t117 = dq3m.*t44;
t124 = -t44;
t125 = -t45;
t126 = -t46;
t128 = -t55;
t129 = -t57;
t130 = -t58;
t149 = -t64;
t150 = -t65;
t151 = -t66;
t152 = -t75;
t153 = -t77;
t154 = -t78;
t155 = cos(t131);
t156 = cos(t132);
t157 = cos(t133);
t158 = cos(t134);
t159 = cos(t135);
t160 = cos(t136);
t161 = cos(t137);
t162 = cos(t140);
t163 = cos(t141);
t164 = cos(t142);
t165 = cos(t143);
t166 = cos(t144);
t167 = cos(t145);
t168 = cos(t146);
t169 = cos(t147);
t170 = cos(t148);
t173 = -t99;
t174 = sin(t131);
t175 = sin(t132);
t176 = sin(t133);
t177 = sin(t134);
t178 = sin(t135);
t179 = sin(t136);
t180 = sin(t137);
t181 = sin(t138);
t182 = sin(t139);
t183 = sin(t140);
t184 = sin(t141);
t185 = sin(t142);
t186 = sin(t143);
t187 = sin(t144);
t188 = sin(t145);
t189 = sin(t146);
t190 = sin(t147);
t191 = sin(t148);
t192 = -t108;
t193 = c2.*t107;
t194 = -t110;
t195 = -t111;
t196 = -t112;
t197 = -t113;
t202 = -t118;
t203 = t42+t53;
t204 = t62+t73;
t338 = t50+t61+t127;
t348 = t28+t29+t30+t31+t107;
t171 = -t97;
t172 = -t98;
t200 = -t116;
t201 = -t117;
t205 = c3.*m3.*t164;
t206 = t27.*t165;
t207 = c5.*m5.*t167;
t208 = l3.*m3.*t164;
t209 = t32.*t165;
t210 = t26.*t184;
t211 = c3.*m3.*t185;
t212 = c3.*m3.*t187;
t213 = t27.*t186;
t214 = t27.*t189;
t215 = c5.*m5.*t188;
t216 = c5.*m5.*t190;
t217 = dq1m.*l1.*t175;
t218 = dq2m.*l2.*t177;
t219 = dq3m.*l4.*t181;
t220 = t28.*t184;
t221 = t29.*t184;
t222 = t30.*t184;
t223 = t31.*t184;
t224 = l3.*m3.*t185;
t225 = l3.*m3.*t187;
t226 = t32.*t186;
t227 = t32.*t189;
t228 = t49.*t155;
t229 = t50.*t155;
t230 = c1.*l4.*m1.*t157;
t231 = c3.*l1.*m3.*t156;
t232 = c3.*t29.*t158;
t233 = l4.*t26.*t160;
t234 = l1.*t27.*t157;
t235 = l2.*t27.*t160;
t237 = l2.*t27.*t168;
t239 = c5.*t31.*t169;
t240 = c5.*t32.*t170;
t241 = t61.*t155;
t242 = l1.*l3.*m3.*t156;
t243 = l1.*l4.*m1.*t157;
t244 = l1.*l4.*m2.*t157;
t245 = l1.*l4.*m3.*t157;
t246 = l3.*t29.*t158;
t247 = l1.*l4.*m4.*t157;
t248 = l4.*t28.*t160;
t249 = l4.*t29.*t160;
t250 = l4.*t30.*t160;
t252 = l4.*t31.*t168;
t253 = t49.*t174;
t254 = t50.*t174;
t255 = c1.*l4.*m1.*t176;
t256 = c3.*l1.*m3.*t175;
t257 = l1.*t27.*t176;
t260 = c5.*t31.*t190;
t261 = c5.*t32.*t191;
t262 = t61.*t174;
t263 = l1.*l3.*m3.*t175;
t264 = l1.*l4.*m1.*t176;
t265 = l1.*l4.*m2.*t176;
t266 = l1.*l4.*m3.*t176;
t267 = l1.*l4.*m4.*t176;
t268 = l4.*t31.*t189;
t272 = c3.*dq2m.*t29.*t177;
t274 = dq2m.*l4.*t26.*t179;
t275 = dq2m.*l5.*t26.*t180;
t276 = dq2m.*l2.*t27.*t179;
t278 = c5.*dq2m.*t31.*t180;
t279 = dq4m.*l5.*t27.*t183;
t280 = c5.*dq4m.*t32.*t183;
t286 = dq2m.*l3.*t29.*t177;
t287 = dq2m.*l4.*t28.*t179;
t288 = dq2m.*l4.*t29.*t179;
t289 = dq2m.*l4.*t30.*t179;
t290 = dq2m.*l5.*t28.*t180;
t291 = dq2m.*l5.*t29.*t180;
t292 = dq2m.*l5.*t30.*t180;
t294 = dq2m.*l5.*t31.*t180;
t295 = dq4m.*l5.*t32.*t183;
t296 = m3.*t83.*t164;
t297 = dq3m.*t84.*t181;
t302 = m3.*t83.*t185;
t303 = m3.*t83.*t187;
t304 = t127.*t155;
t306 = l3.*m3.*t81.*t156;
t307 = l4.*m1.*t81.*t157;
t308 = l4.*m2.*t81.*t157;
t309 = l4.*m3.*t81.*t157;
t310 = t29.*t83.*t158;
t311 = l4.*m4.*t81.*t157;
t312 = t28.*t84.*t160;
t313 = t29.*t84.*t160;
t314 = t30.*t84.*t160;
t316 = c1.*m1.*t84.*t176;
t317 = c3.*m3.*t81.*t175;
t318 = t27.*t81.*t176;
t323 = dq1m.*t127.*t174;
t325 = dq2m.*t26.*t84.*t179;
t326 = dq2m.*t27.*t82.*t179;
t327 = dq1m.*l4.*m1.*t81.*t176;
t328 = dq1m.*l4.*m2.*t81.*t176;
t329 = dq1m.*l4.*m3.*t81.*t176;
t330 = dq1m.*l4.*m4.*t81.*t176;
t331 = dq2m.*t28.*t84.*t179;
t332 = dq2m.*t29.*t84.*t179;
t333 = dq2m.*t30.*t84.*t179;
t335 = t54+t55+t124;
t336 = t57+t58+t126;
t337 = t47+t60+t130;
t339 = t74+t75+t149;
t340 = t77+t78+t151;
t341 = t67+t80+t154;
t342 = l1.*t186.*t199;
t343 = l2.*t189.*t199;
t344 = l1.*m3.*t121.*t185;
t345 = t29.*t121.*t187;
t347 = t94+t95+t202;
t350 = t104+t105+t173;
t351 = t82.*t189.*t199;
t354 = t77+t78+t79+t150;
t356 = t57+t58+t59+t125;
t357 = dq1m.*m1.*t2.*t119.*t203;
t358 = dq1m.*m1.*t12.*t119.*t204;
t367 = dq1m.*t174.*t338;
t368 = empty+t70+t115+t194+t197;
t370 = l1.*t184.*t348;
t371 = t81.*t184.*t348;
t236 = l1.*t206;
t238 = l1.*t207;
t251 = l1.*t209;
t258 = l2.*t214;
t259 = l1.*t215;
t269 = dq1m.*t253;
t270 = dq1m.*t254;
t271 = dq1m.*t255;
t273 = dq1m.*t257;
t277 = c3.*m3.*t219;
t281 = dq1m.*t262;
t282 = dq1m.*t264;
t283 = dq1m.*t265;
t284 = dq1m.*t266;
t285 = dq1m.*t267;
t293 = l3.*m3.*t219;
t298 = -t220;
t299 = -t221;
t300 = -t222;
t301 = -t223;
t305 = -t240;
t315 = -t254;
t319 = t81.*t215;
t320 = -t260;
t321 = -t261;
t322 = -t262;
t324 = -t272;
t334 = m3.*t83.*t219;
t346 = t94+t95+t200;
t349 = t104+t105+t171;
t352 = -t345;
t353 = t76+t339;
t355 = t56+t335;
t359 = t153+t341;
t360 = t129+t337;
t361 = -t357;
t362 = -t358;
t364 = t94+t95+t96+t201;
t366 = t104+t105+t106+t172;
t373 = t212+t214+t216+t227+t303;
t363 = t3.*t26.*t346;
t365 = t13.*t26.*t349;
t369 = t72+t196+t305;
t372 = t71+t195+t198+t305;
mt1 = [t48.*2.0+t68+t192+t236+t238+t251+l1.*t205-m1.*t38-m2.*t38-m3.*t38-m4.*t38-m5.*t38+t50.*t163+t81.*t208+t28.*t81.*t163+t29.*t81.*t163+t30.*t81.*t163+t31.*t81.*t163,l1.*(t205+t206+t207+t209+t296+t26.*t163-t28.*t163-t29.*t163-t30.*t163-t31.*t163),l1.*(t205+t206+t207+t209+t296),t236+t238+t251,t238,t69-t8.*t30.*t336-t18.*t30.*t340-t8.*t29.*t356+t8.*t31.*t360-t18.*t29.*t354+t18.*t31.*t359+m2.*t8.*t120.*(t57+t58-c2.*t8)+m2.*t18.*t120.*(t77+t78-c2.*t18),t51.*2.0+t69+t193+t237+t239+t252+t28.*t82+t29.*t82+t30.*t82+t31.*t82+c3.*t29.*t166+t29.*t83.*t166];
mt2 = [l2.*(t27.*t168+t32.*t168+c3.*m3.*t166+c5.*m5.*t169+m3.*t83.*t166),t237+t239+t252,t239,t70+m3.*t9.*t121.*t356+m3.*t19.*t121.*t354,t70+m3.*t9.*t121.*(t58+t59+t125)+m3.*t19.*t121.*(t78+t79+t150),t368,t71+t10.*t27.*t336+t20.*t27.*t340-t10.*t32.*t360-t20.*t32.*t359,t71-t10.*t27.*(t46+t130)-t20.*t27.*(t66+t154)-t10.*t32.*t337-t20.*t32.*t341,t372,t372,t305,t72-m5.*t47.*t360-m5.*t67.*t359,t72-m5.*t47.*t337-m5.*t67.*t341,t369,t369,empty+t72+t196];
m = reshape([mt1,mt2],23,1);
if nargout > 1
    mi = [1.0;2.0;3.0;4.0;5.0;6.0;7.0;8.0;9.0;1.0e+1;1.1e+1;1.2e+1;1.3e+1;1.6e+1;1.7e+1;1.8e+1;1.9e+1;2.0e+1;2.1e+1;2.2e+1;2.3e+1;2.4e+1;2.5e+1];
end
if nargout > 2
    mt3 = [t100+t101+t102+t103+dq5m.*t72-m2.*t346.*(-t43+t55+t56)-m2.*t349.*(-t63+t75+t76)-m3.*t353.*t366-m3.*t355.*t364-m4.*t347.*(t55+t56-c4.*t5)-m4.*t350.*(t75+t76-c4.*t15)+m5.*t6.*t123.*(t94+t95+dq4m.*t128-c5.*dq5m.*t6)+m5.*t16.*t123.*(t104+t105+dq4m.*t152-c5.*dq5m.*t16)-dq1m.*m1.*t2.*t119.*(-t56+t128+t203)-dq1m.*m1.*t12.*t119.*(-t76+t152+t204),t100+t101+t102+t103+m2.*t346.*(t43+t128)+m2.*t349.*(t63+t152)-m3.*t335.*t364-m3.*t339.*t366+m4.*t5.*t122.*t347+m4.*t15.*t122.*t350-dq1m.*m1.*t2.*t119.*(t128+t203)-dq1m.*m1.*t12.*t119.*(t152+t204)];
    mt4 = [t100+t101+t102+t361+t362+t363+t365+m3.*t4.*t121.*t364+m3.*t14.*t121.*t366,t100+t101+t361+t362+t363+t365,-dq1m.*(I1-t48+t108)];
    f = reshape([mt3,mt4],5,1);
end
if nargout > 3
    fi = [1.0;2.0;3.0;4.0;5.0];
end
if nargout > 4
    t374 = l2.*t373;
    t375 = t82.*t373;
    t376 = t210+t211+t213+t215+t226+t298+t299+t300+t301+t302;
    t377 = l1.*t376;
    t378 = t81.*t376;
    mz = [t378;t378;t81.*(t211+t213+t215+t226+t302);t81.*(t213+t215+t226);t319;t370;m3.*t81.*t121.*t185;t81.*t186.*t199;t319;t371;t371;t320+t50.*t184+t82.*t214+t81.*t220+t81.*t221+t81.*t222+t81.*t223-c3.*t29.*t187+l3.*t29.*t187+t31.*t84.*t189;t375;t375;t82.*(t214+t216+t227);t320;t352;t352;t351;t351;t320;t320;t344;t344;t344;t345;t345;t345;m3.*t121.*(l1.*t185+l2.*t187);t345;t342;t342;t342;t342;t343;t343;t343;t343;t258+t261+t268+l1.*t213+l1.*t226;t258+t261+t268;t261;t261;t261;t261;t261;t261;t261;t259;t259;t259;t259;t259;t260;t260;t260;t260;t260;t321;t321;t321;t321;t321;-m5.*t47.*t359+m5.*t67.*t360;c5.*m5.*(l2.*t190+t84.*t191);t321;t321];
end
if nargout > 5
    mzi = [2.6e+1;2.7e+1;2.8e+1;2.9e+1;3.0e+1;3.1e+1;3.6e+1;4.1e+1;4.6e+1;5.1e+1;5.2e+1;5.6e+1;5.7e+1;5.8e+1;5.9e+1;6.0e+1;6.1e+1;6.2e+1;6.6e+1;6.7e+1;7.1e+1;7.2e+1;7.6e+1;7.7e+1;7.8e+1;8.1e+1;8.2e+1;8.3e+1;8.6e+1;8.7e+1;1.01e+2;1.02e+2;1.03e+2;1.04e+2;1.06e+2;1.07e+2;1.08e+2;1.09e+2;1.16e+2;1.17e+2;1.18e+2;1.19e+2;1.2e+2;1.21e+2;1.22e+2;1.23e+2;1.24e+2;1.26e+2;1.27e+2;1.28e+2;1.29e+2;1.3e+2;1.31e+2;1.32e+2;1.33e+2;1.34e+2;1.35e+2;1.41e+2;1.42e+2;1.43e+2;1.44e+2;1.45e+2;1.46e+2;1.47e+2;1.48e+2;1.49e+2];
end
if nargout > 6
    mzd = [5.0,5.0,2.2e+1];
end
if nargout > 7
    mt5 = [dq1m.*(t253+t263+t264+t265+t266+t267+t315+t316+t317+t318+t322+c1.*m1.*t85.*t178+c5.*m5.*t81.*t178+l1.*l5.*m1.*t178+l1.*l5.*m2.*t178+l1.*l5.*m3.*t178+l1.*l5.*m4.*t178+l1.*l5.*m5.*t178),dq1m.*(t253+t263+t264+t265+t266+t267+t315+t316+t317+t318+t322),-dq1m.*(t254+t256+t262+t127.*t174+l3.*m3.*t81.*t175),-t367,t270-t278+t281+t286+t287+t288+t289+t290+t291+t292+t294+t323+t324+t325+t326+dq2m.*t26.*t85.*t180,t270+t281+t286+t287+t288+t289+t323+t324+t325+t326,t270+t281+t286+t323+t324,t367,m3.*t121.*(t217+t218+t297+dq3m.*t85.*t182),m3.*t121.*(t217+t218+t297),m3.*t121.*(t217+t218),t271+t273+t274+t276+t277+t280+t327+t328+t329+t330+t331+t332+t333+t334+dq4m.*t27.*t85.*t183+dq4m.*t32.*t85.*t183,t271+t273+t274+t276+t277+t327+t328+t329+t330+t331+t332+t333+t334];
    mt6 = [t275+t278+t279-t280+t295+dq2m.*t28.*t85.*t180+dq2m.*t29.*t85.*t180+dq2m.*t30.*t85.*t180+dq2m.*t31.*t85.*t180+c1.*dq1m.*l5.*m1.*t178+c5.*dq1m.*l1.*m5.*t178+c3.*dq3m.*l5.*m3.*t182+dq1m.*l5.*m1.*t81.*t178+dq1m.*l5.*m2.*t81.*t178+dq1m.*l5.*m3.*t81.*t178+dq1m.*l5.*m4.*t81.*t178+dq1m.*l5.*m5.*t81.*t178+dq3m.*l5.*m3.*t83.*t182,t48+t68+t192+t229+t230+t231+t234+t241+t304+t306+t307+t308+t309+t311+c1.*l5.*m1.*t159+c5.*l1.*m5.*t159+l5.*m1.*t81.*t159+l5.*m2.*t81.*t159+l5.*m3.*t81.*t159+l5.*m4.*t81.*t159+l5.*m5.*t81.*t159,t48+t68+t192+t229+t230+t231+t234+t241+t304+t306+t307+t308+t309+t311,t48+t68+t192+t229+t231+t241+t304+t306,t48+t68+t192+t229+t241+t304,empty+t48+t68+t192];
    mt7 = [t51+t69+t193+t232+t233+t235+t310+t312+t313+t314+c5.*t31.*t161+l5.*t26.*t161+t28.*t85.*t161+t29.*t85.*t161+t30.*t85.*t161+t31.*t85.*t161,t51+t69+t193+t232+t233+t235+t310+t312+t313+t314,t51+t69+t193+t232+t310,empty+t51+t69+t193,t70+m3.*t4.*t121.*t355+m3.*t14.*t121.*t353,t70+m3.*t4.*t121.*t335+m3.*t14.*t121.*t339,t368,t52+t71+t195-c5.*t32.*t162+l5.*t27.*t162+l5.*t32.*t162,empty+t52+t71+t195,empty+t72+t196+c5.*l5.*m5];
    fz = reshape([mt5,mt6,mt7],29,1);
end
if nargout > 8
    fzi = [6.1e+1;6.2e+1;6.3e+1;6.4e+1;6.6e+1;6.7e+1;6.8e+1;6.9e+1;7.1e+1;7.2e+1;7.3e+1;7.6e+1;7.7e+1;8.1e+1;8.6e+1;8.7e+1;8.8e+1;8.9e+1;9.0e+1;9.1e+1;9.2e+1;9.3e+1;9.4e+1;9.6e+1;9.7e+1;9.8e+1;1.01e+2;1.02e+2;1.06e+2];
end
if nargout > 9
    fzd = [5.0,1.0,2.2e+1];
end
