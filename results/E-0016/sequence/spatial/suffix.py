"""Частота подходящей длины продолжения по агрегированным учебным счётчикам."""
import numpy as np


def likelihood(ref, meta, urgent, mixture, counts):
    p,q,k,t=np.moveaxis(ref,-1,0).astype(np.int32)
    valid=(p>=0) & (p!=q)
    # Незаполненные элементы не попадают в окончательный результат.
    p=np.maximum(p,0);q=np.maximum(q,0);t=np.maximum(t,0)
    ux=q%40-p%40; uy=q//40-p//40
    rx=t%40-q%40;ry=t//40-q//40
    forward=ux*rx+uy*ry+72
    lateral=np.abs(ux*ry-uy*rx)
    diag=(ux!=0)&(uy!=0)
    along=np.where(ux!=0,p%40,p//40)%2
    across=np.where(ux!=0,p//40,p%40)%2
    parity=np.where(diag,2*np.minimum(along,across)+np.maximum(along,across),2*along+across)
    hs=counts[:,diag.astype(int),parity,forward,lateral].astype(np.float64)
    remaining=np.clip(int(meta[6+urgent])-k,0,64)
    a,b=np.indices(k.shape)
    freq=hs[:,a,b,remaining]
    smooth=(hs[:,a,b,np.minimum(64,remaining+2)]+hs[:,a,b,np.maximum(0,remaining-2)])*.5
    n=hs[:,:,:,65]
    rate=(freq+.3*smooth+.01)/(1.3*n+3)
    pooled=(freq.sum(0)+.3*smooth.sum(0)+.01)/(1.3*n.sum(0)+3)
    conditional=((rate*n+pooled[None]*24)/(n+24)*mixture[:,None,None]).sum(0)
    confirmed=(ref[:,:,0]>=0)&(ref[:,:,0]==ref[:,:,1])
    return np.stack((np.where(confirmed,1.,conditional*valid),np.where(confirmed,1.,pooled*valid)),-1).astype(np.float32)
