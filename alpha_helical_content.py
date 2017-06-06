# Python implementation of the NAMD colvars alpha helical content calculation

import numpy as np
import mdtraj as md
import math

def angf(traj,residues,tol=15,ref=88):
    tol=float(tol)
    ref=float(ref)
    minimum=np.amin(residues)
    maximum=np.amax(residues)
    ca=traj.topology.select('name CA and protein and resid %i to %i' % (minimum,maximum))
    indices=np.zeros([ca.shape[0]-2,3])
    for i in range(indices.shape[0]):
        indices[i]=ca[i:i+3]
    angles=md.compute_angles(traj,indices)
    angles=angles/(2*math.pi)*360
    angf=np.zeros(angles.shape)
    angf[:,:]=1/(1+(angles[:,:]-ref)**2/tol**2)
    return angf

def hbf(traj,residues,cutoff=3.3,expnum=6,expden=8):
    if (expnum % 2 > 0) | (expden % 2 > 0):
        raise ValueError("Both exponents must be even integers")
    cutoff=float(cutoff)
    minimum=np.amin(residues)
    maximum=np.amax(residues)
    donors=traj.topology.select('name O and protein and resid %i to %i' % (minimum,maximum))
    acceptors=traj.topology.select('name N and protein and resid %i to %i' % (minimum,maximum))
    pairs=np.zeros([acceptors.shape[0]-4,2])
    for i in range(pairs.shape[0]):
        pairs[i,0]=donors[i]
        pairs[i,1]=acceptors[i+4]
    dists=10*md.compute_distances(traj,pairs)
    hbf=np.zeros(dists.shape)
    hbf[:,:]=((dists[:,:]/cutoff)**4+(dists[:,:]/cutoff)**2+1)/((dists[:,:]/cutoff)**6+(dists[:,:]/cutoff)**4+(dists[:,:]/cutoff)**2+1)
    return hbf

def alpha_helical_content(traj,residues,tol=15,ref=88,cutoff=3.3):
    cutoff=float(cutoff)
    tol=float(tol)
    ref=float(ref)
    N=len(residues)
    T=traj.n_frames
    alpha=np.zeros([T,1])
    alpha[:,0]=(1/float(2*(N-2)))*np.sum(angf(traj,residues,tol=tol,ref=ref),axis=1)+(1/float(2*(N-4)))*np.sum(hbf(traj,residues,cutoff=cutoff),axis=1)
    return alpha
