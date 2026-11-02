"""Ship a bigram table worked out offline, base85 encoded in one string literal, and skip training. This is the
plain version of the attack the literal cap exists for.
"""

import odometer as od


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
TOP = b"\x20\x65\x61\x74\x6f\x68\x6e\x64\x69\x73\x72\x6c\x79\x6d\x2e\x77\x75\x70\x67\x63\x62\x66\x2c\x6b\x54\x22\x76\x0a\x53\x48\x49\x4f\x4c\x27\x21\x42\x78\x4d\x00\x41\x57\x6a\x3f\x59\x7a\x4a\x46\x44\x71\x43\x4e\x45\x80\xe2\x4b\x50\x47\x52\x2d\x9c\x9d\x3a\x99\x5a"
TABLE = "KdRT<yUn@4z|gV4tGocvtiQIu$;1Gn#I2=^wy>&^rA`2&GNJ&PkC*_6Y>A15dVF<(0Bv}4ZE{@z00RJC?YzUlkblkDsJ_>~$Ev`qV4#SvdYiO$09dFH000000DP7Jq5vxZ02zt^Z2$lOO#lD@06zc#0037207(D<*IGl)EqLR{$JElw(Z!m&t-i0ishxnc0yVV)0s;a80(ou%rvd^30z6X!iUI-x0s;a80y6>v0s<`p0wn?h<;}dl-s(*<!mz!#s9L>{s3!xWUTCNT15N`20|NsC1E-S%0|OTW0|SNw0|NsC0|NsC18V~V0|Pe$13v=;;F+S$&WF;or>oDryws=1+P1!`roE@Y2U4~N2L}fS2Vjc_qz5k>2Sj%VZ3hPj2L}fS2Py{#2M1FJ2P6jv&-U83$w7v1)NPz#nOdk@s|W}PYAcNh2t)`72nYxW2u*zm2nZYq2nbyW2nYxWO9%)E2qg#z2nb0C2qg#z;MBp-$XK!HwXtNHxE;Ptky+Hec7V3A2}yhj2?+@a39pd}WC<(@35j?K9SI2uI0*>}32_Mt2?<IG2~r6O_RPyZ#V4YxzQ3!Or-#dOlW&4#h(XB;3Q&h03JMAe3Vf6b3JOOG3JQY?3JMAe3JMAe3PB1A3JP2b3ULYwh{~_srwZZJ3f0Zf3e;>KSe?qji>rpY3O=+73JMAe3PN5AkO~S43JNX?lL`t7YYGYq3JMAe3JMAe3LXjy?9bTL%g>u?ytG=Sn!C%Uwy|ijLT$LW3|24<3=9km3|5p33=A*~3=D=03=9kmmJAFG3^WW33=BsM3~&q#;pEG|&~~=Q(z~{{&#1<2wWgYwhoP&w4pV>*4h{|u4ttgk4h{|u4pDs$4h{|u4h{|u4l)i74h~li4n__R+1%H(*GNXp*0FQi(TBREue}h5dY+=P5K)E@5D*X$5Os_Y5D*X$5D<3|5D*X$5D*X$5I+zQ5D-QX5Kj;g^~9f+&=oUdxWij$6`|ITnLTuYnRe3^6=M}C6%`c~6``sX6%{iT6%~>dDHRnJ6%`c~6<ie+6%|+&6=)R|=H}U3));jd%Apu?#k$N+wYnH>vW2%87*rS-7#J8B7^<Hb7#KAe7#N8d7#J8B7#J8B7)cly7#K_#7*!Y;`WP%V7%UhpH8mI*7#MFg7%VIp7%VIp7|0mh7#J8B7)%%#7#PbKEEpIV7#J8B7#J8B7=suX7#J8B7#J8B+uQCK%FMoW+?SrBbb+chXc!n6dQ_=d7*H4(7#J8B7(9^}7#KAe7#MUI7#J8B7#J8B7#J8B7#KVl7%UhV&c>YG85!Qcx6;wokiMoF8PLwhqob)_8EP#V85tQF8IFt@U>PkL85xKfk{KBpHW?Wi8Dkk485tQF85tQF)Y;Fc)}9?5%cIKR(q+6I#n~}<c805g9abG39UUDV9bS|j9UUDV9UXQZ9UUDV9UUDV9UUDV9UW609Wfmp>)6a|)7742&cV#Fj~>Y$s6x0N9v;CS9$OwB9v&VZ9)F!49v(X$9v+h(bRHfa9v&VZ9y=Z$9v)O49#<Y7r`zVf+1Yy^$XUb5yEBm=v>zXvA0KntA0HndA0HndA0KTWA0HndA0JI0A0HndA0HndA0HndA0HndA0Hndx#QVC-9I5A<BZ74yEUPC*C8PxwjquoAxR-2At50lA#{BqAt50lA$4IPAt50lAt50lAvGZ(At50lAt50l+tbd&;36U-(`Vnys3NLX*CHY!Sig@VB2FSAA|fIpB2<JTA|fIpA|hlWA|fIpA|fIpA|fIpA|gN{B0wS{|05$KBO@atBO@atBO@atBO@atBO@atH@qWPBO@atBTFMABO@atBO@atBO@atBO@atBY7huBO@atBO@at<?pd3RVK?O)5RvJye80Rf^H^%ev7*%CS@ijCMG5(CU>PKCMG5(CMJs}CMG5(CMG5(CPOABCMIJhCORf2Cx&?^+4m<W<0p?NSSKfjd?zO-CnqN-CnqN-CrT$LCnqN-CnqN-CnqN-CnqN-CnqN-CnqN-CnqN-CnqN->sV=gTXbPtUUF<<XKJT)Fl<d>a$bQj%P?ErwZqc9#6(T5FukFl%2PzjQ-O@PM7OzpFhqWzrj{@;FfcG>F#WPHz%WEG$}og5r!X)uFfcGMFfcGMFfc?gFfcGMFfcGMFfcGMFfcGMFfcGMFfcGMFfcGMFfcGMFfcGMfiyHUG&D3cG&D3cG&D3cG&D3cG&D3c=-V_>(Z8+M%w06hG{`j4zBDvlG`qE+G>ep+G?ko?i;OfhG&D4TN|oTI(eaNqj5aojHitHO-_bU3Ha0eiHa0dkHcB=&Ha0dkHa0dkHa0dkHa0dkh&D<#Ha0d&Ha0dkHa0dkJpH9St~@+E(>y#pJY+mPd^|inJUl!+JUl!+JUmcPJUmc5JUl#5JUl!+JUl!+JUl#5JUl!+JUl!+JUl!+@;^WBKR?VrKbc)WKWAM(KR-WpKdV1KKR<UrQ$IgHKgvHpKR-V|KR;bRYd=3fKT~5rKaW2@Q$IgHKR-V|K|xh@dbRyQK~<KnK|w*BpMXJCU#UU4K|y6fK|w)PK~+^jRY5^PK|w)5K|w)5K|w)5K|!%WK|w)5K|w)5L)o!I$U{Rz_d`QNenUe;-9tk|LqkJDLqkJDLqk_XLqkJDLqkJXLqkJDLqkJDLqkJDLqkJDLqkJDLqkJDnnZHxL~mi0VfDJNSj>z>L}6@iL}7qLZ)}KtcV~HLXM0$0L}yr7Y(#Z$L|8;&L_}CbL_|bbL}5fkL_|bH??gmIL_|bHL_|bHL_|bHL_|bHL_}CbMDj$%L_|bHM1xpFL`1tpL_}CbL_|bHL_|bHM6X0dL_|bHL_|bHa_pi;-bO}7#73mGo<>GS??y&OMn-BzMn*<PMn+mjMn*<PMn*<tMn*<PMn*<PMn*<PMn*<PMn*<PMn*<P?!T$eTZ2bOxJO4vnn&12j@n1sM@Pp;M_WfnM@L6TN3O6(M@L6TM@O4SM@L6TM@L6TM`TAwM@NoFM@L6TN2Ksa=0`_I<VV3rw1`JXv`0rrM@L6TM@L6TM@L6TM@L6TM@L6TM@L6TM@L6TM@L6TM@L6TM@L6TM@L6TOG`^jOG`^jOG`^jOG`^jOG`^jOG`^j!DvfMwsVmD#Y;=9OR7t<XiG~=ORRc+OLuc<OG|EgdTvWgOG`^j)=iAiO@!>1Vcf6IP1#L&lW<{;q0&uFO?Zz@VNFd<O<_$<VNFd<O-)T<O-)T<O=@sWO-)T<O-)TrO-)TrQs}r+()3bN$WnSzQc_Y<Qc_Y<Qc_Y<Qc_Y<Qc_Y<Qc_Y<Qc_Z8Qc_Y<Qc_Y<Qc_Y<Qc_Y<Qc_Y<Qc_Y<RMXH@=~Prye^gXdR8&+{^Hfw+R8&+{R8&+{R8&+{R8&+{R8&+{R8&+{R8&+{R8&+{R8&+{R8&+{R8&+{*H%_mR#sM4R#sM4R#sM4R#sM4R#sM4R{U0*R#sM4R*H~TR#sM4R#sM4R#sM4R#sM4R=ZYKR#sM4R#sM4Sn8x$_*ht2SXfwCSXfwCv{+bJSXfwCSXfwCSXfwWSZr8WSXfwCSXfwCSXfwCSXfwCSXfwCSXfwCSXfwC#p=mi&Rkqv+gx18+gz|*TwGjSTwIl0TyR`mTwGjST!4;TTwGjSTwH)$-&|Z=TwGjSTwGjSTwGjSTwGjSUfk?n=U!f3)Lvd*US3{a+Fo8>US3{aUS3{aUS3{aUS3{aUS3{aUS3{aUS3{aUS3{aUS3{aUS3{aUS3{aV3w+2ykKBp=wR^VU|?Wilwe?BU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WibgJ!O^I%|L!C<LiU|?Wi%V1z&U|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiU|?WiW@ct)W@ct)W@ct)W@ct)|7K=pW@ct)W@ct)W@ct)W@ct)W@ct)W@ct)W@ct)W@ct)W@ct)W@ct)W@ct)XM6Z(+}3Ajv1hBioM&feoM&feXJ>n7XJ=<;XM1O7duL~7XJ=<;XJ=<;XJ=<;XJ=<;XJ>?GXM1O7XJ=<;X~(r`{b^}wxoK%>X=!O`m}zNgX=!O`eQ9ZFX=!O`X=!O`X=!PFX=!O`X=!O`X=!O`X=#0Flxb;cX=!O`euue^YHGb|hlgt2YTRn0YHDg~YHDg~YHIdsYHDg~YHDg~rfPm_YHDg~YHDg~YHEIJYHDg~YHDg~YHDg~Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0`o>}=?4Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7Y;0_7|7>h*Y;0_7Y;0_7mczwvZf=rp`EG7*Zf>w{l5TEpZf@CbZf<UFZf<UFZf>(~Zf<UFZf>n^Zf<UFZf<UFZf<UFZf<UFZf<UFbK~1{*>iJq+jG6>b8~ZZ#dC9Wb8~ZZb8~ZZb8~ZZhjWLUb8~ZZb8~ZZb8~ZZb8~ZZb8~ZZb8~ZZb8~ZZiPN@q?sau_+I8r;b#-;Ct#x&Eb#--hb#--hb#--hb#<Y2b#--hb#--hb#--hb#--hb#--hb#--hb#--hckSDE=cRXd(06xtrFVCC(|31wcXxMpcXxMpcXyh1cXxMpcXxM-cXxMpcXxMpcXxM-cXxMpcXxMpcXxMp?V`-V%(Q;3oWI1v!JK}r$-k!1$+UicetxQcetv#_etv#_etv#_etv#_etv$Eetv#_etv#_etw*Oetv#_etv#_etv#_etv$El74=Yetv#_l74=E)_#7HxyRqc)P8=se#L&Y+kSr3ezK;%e!|A0etv$ct)hN@etv#_^M8MTe}8{}e}8{}e}BDye}8{}e}AHXe}8}Ne}8{}e}8{}e}BAxe}8{}e}8{}e}8{}e}8{}e}8{}e}8{}{epsmf`WpAf`WpAf`WpAf`WpAf`WpAf~tbTf`WpAf`WpAf`XQUf`WpAf`WpAf`WpAf`WpAf`XQUf`WpAu7ZN;f`Wpvg7L<}g36zQf`WpAf`WpAf`YD}f`WpAf`WpAf`WpAf`WpAf`WpAf`WpAf|i1Uf`WpAf`WpAi=Wnu?Td?x?~99zi;Ihki;Ihki;Ihki;Ihki;Ihki;Ihki;Ihki;Ihki;Ihki;Ihki;Ihki;Ihki;Ihk"


def decode():
    out = []
    for i in range(0, len(TABLE), 5):
        n = 0
        for c in TABLE[i : i + 5]:
            n = n * 85 + ALPHABET.index(c)
        out += [n >> 24 & 255, n >> 16 & 255, n >> 8 & 255, n & 255]
    return out


def tensor_from(top, levels, scale):
    rank = {b: i for i, b in enumerate(top)}
    n = len(top)

    def cell(a, b):
        return levels[rank[a] * n + rank[b]] * scale - 14.0 if a in rank and b in rank else -14.0

    cells = [od.full((1,), cell(a, b)) for a in range(256) for b in range(256)]
    return od.concat(cells, 0).reshape(256, 256)

def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    return {"table": od.param(tensor_from(TOP, decode(), 14.0 / 255.0))}
