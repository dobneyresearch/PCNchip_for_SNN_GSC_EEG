#!/usr/bin/env python3
"""Cycle-accurate bit-faithful reference for pcn_temporal_2layer.v (deep credit).

Two layers, 1 neuron each. Layer-2 adjoint δ2 updates W2 AND routes down through
the Wᵀ hop to become layer-1's error e1, driving layer-1's adjoint δ1 and update.
Mirrors the RTL registers (all reads OLD, committed together): the transpose and
layer-1 adjoint use OLD δ2, so δ1 lags δ2 by one cycle (a pipeline stage).

Dumps per-cycle stimulus + expected E1,E2,dW1,dW2 for tb_temporal_2layer.v.
"""
import os, math, random

MEMW, INW, ALPHA, ASHIFT, THRESH, PW, SURR = 20, 12, 230, 8, 1024, 8, 5
NUMER = (THRESH * THRESH) << PW
EEW, DW, N, AWACC, ECW, GEW, DWW, WW, RSH = 12, 16, 8, 22, 32, 16, 20, 8, 16
B1, BSH1, B2, BSH2, F, CLIP, EPS, MW, VW = 230, 8, 1014, 10, 10, 1024, 1, 24, 32
EMAX, EMIN = (1 << (EEW-1)) - 1, -(1 << (EEW-1))
T = 40

def wrap(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >= (1 << (w - 1)) else v

from psi_lut import psi_lut as psi_of               # refinement #3: ψ via shared LUT

class L2:
    def __init__(self):
        self.mem1=self.psi1=self.s1=0; self.fifo1=[0]*(N+1); self.delta1=0; self.E1=0; self.m1=self.v1=0; self.dW1=0
        self.mem2=self.psi2=0;         self.fifo2=[0]*(N+1); self.delta2=0; self.E2=0; self.m2=self.v2=0; self.dW2=0
        self.z0sr=self.z1sr=0
        self.rq_scale=self.rq_q=0                        # route_quant (6-bit routed message) registers

    def step(self, en, fold, ic1, ic2, e2, z0, w2):
        if en:
            i1 = ((self.mem1*ALPHA)>>ASHIFT) + ic1
            p1n = psi_of(i1); s1n = 1 if i1>=THRESH else 0
            m1n = wrap((i1-THRESH) if s1n else i1, MEMW)
            i2 = ((self.mem2*ALPHA)>>ASHIFT) + ic2
            p2n = psi_of(i2); s2n = 1 if i2>=THRESH else 0
            m2n = wrap((i2-THRESH) if s2n else i2, MEMW)
            a2 = wrap((self.psi2*e2)>>PW, DW)                       # OLD psi2
            d2n = wrap(self.delta2 + a2 - self.fifo2[N], AWACC)
            f2n = [a2] + self.fifo2[:N]
            e1_raw = max(EMIN, min(EMAX, (self.delta2*w2) >> RSH))  # transpose of OLD delta2
            # route_quant: 6-bit message + running-peak scale (updates from current e1_raw, OLD scale)
            L6 = 31
            av = abs(e1_raw); s_dec = self.rq_scale - (self.rq_scale >> 6)
            s_nxt = max(av, s_dec); s_use = s_nxt if s_nxt > 0 else 1
            qmag = min((2*(av*L6) + s_use) // (2*s_use), L6)
            rq_q_n = -qmag if e1_raw < 0 else qmag
            rq_scale_n = s_nxt
            # a1 uses the reconstruct of the OLD route_quant registers: e1 = q*scale/L
            qm = abs(self.rq_q); recon = (qm * self.rq_scale) // L6
            e1 = -recon if self.rq_q < 0 else recon
            a1 = wrap((self.psi1*e1)>>PW, DW)                       # OLD psi1, OLD rq
            d1n = wrap(self.delta1 + a1 - self.fifo1[N], AWACC)
            f1n = [a1] + self.fifo1[:N]
            z0d = (self.z0sr>>(N-1))&1; z1d = (self.z1sr>>(N-1))&1
            E1n = wrap(self.E1 + (self.delta1 if z0d else 0), ECW)  # OLD delta1
            E2n = wrap(self.E2 + (self.delta2 if z1d else 0), ECW)  # OLD delta2
            z0n = ((self.z0sr<<1)|(z0&1)) & ((1<<N)-1)
            z1n = ((self.z1sr<<1)|(self.s1&1)) & ((1<<N)-1)         # OLD s1
            (self.mem1,self.psi1,self.s1,self.fifo1,self.delta1,self.E1)=(m1n,p1n,s1n,f1n,d1n,E1n)
            (self.mem2,self.psi2,self.fifo2,self.delta2,self.E2)=(m2n,p2n,f2n,d2n,E2n)
            self.z0sr,self.z1sr=z0n,z1n
            self.rq_scale,self.rq_q=rq_scale_n,rq_q_n
        elif fold:
            for (Eattr,mattr,vattr,dattr) in [("E1","m1","v1","dW1"),("E2","m2","v2","dW2")]:
                E=getattr(self,Eattr); sp,sn=(1<<(GEW-1))-1,-(1<<(GEW-1))
                ef=sp if E>sp else sn if E<sn else E
                m=wrap((B1*getattr(self,mattr)+((1<<BSH1)-B1)*ef)>>BSH1, MW)
                v=(B2*getattr(self,vattr)+((1<<BSH2)-B2)*(ef*ef))>>BSH2
                q=min((abs(m)<<F)//math.isqrt(v+EPS), CLIP)
                setattr(self,mattr,m); setattr(self,vattr,v)
                setattr(self,dattr,-q if m<0 else q); setattr(self,Eattr,0)
        return self.E1,self.E2,self.dW1,self.dW2

def hexs(v,w): return f"{v & ((1<<w)-1):0{(w+3)//4}x}"

def main():
    random.seed(51)
    cyc=[]
    for _ in range(T):
        cyc.append((1,0, random.randint(-200,900), random.randint(-200,900),
                    random.randint(-800,800), 1 if random.random()<0.5 else 0,
                    random.randint(-100,100)))
    for _ in range(N):
        cyc.append((1,0,0,0,0,0, cyc[-1][6]))   # flush (hold w2)
    cyc.append((0,1,0,0,0,0,0))                 # fold

    m=L2(); E1s,E2s,W1s,W2s=[],[],[],[]
    for (en,fd,ic1,ic2,e2,z0,w2) in cyc:
        E1,E2,d1,d2=m.step(en,fd,ic1,ic2,e2,z0,w2)
        E1s.append(E1);E2s.append(E2);W1s.append(d1);W2s.append(d2)
    d=os.path.dirname(os.path.abspath(__file__))
    def dump(n,seq,w):
        open(os.path.join(d,n+".hex"),"w").write("\n".join(hexs(x,w) for x in seq)+"\n")
    dump("l2_en",[c[0] for c in cyc],1); dump("l2_fold",[c[1] for c in cyc],1)
    dump("l2_ic1",[c[2] for c in cyc],INW); dump("l2_ic2",[c[3] for c in cyc],INW)
    dump("l2_e2",[c[4] for c in cyc],EEW); dump("l2_z0",[c[5] for c in cyc],1)
    dump("l2_w2",[c[6] for c in cyc],WW)
    dump("l2_E1_exp",E1s,ECW); dump("l2_E2_exp",E2s,ECW)
    dump("l2_dW1_exp",W1s,DWW); dump("l2_dW2_exp",W2s,DWW)
    print(f"wrote l2 vectors: cycles={len(cyc)}, final dW1={W1s[-1]} dW2={W2s[-1]}")

if __name__=="__main__":
    main()
