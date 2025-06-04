"""pycutfem.integration.cut_integration"""
import numpy as np

def integrate_cut_tri(mesh, elem_id, level_set, f, n_samples=2000):
    tri_nodes=mesh.nodes[mesh.elements[elem_id]]
    xmin,ymin=tri_nodes.min(axis=0)
    xmax,ymax=tri_nodes.max(axis=0)
    area_elem=mesh.areas()[elem_id]
    rng=np.random.default_rng(seed=elem_id)
    count=0; val=0.0
    v0,v1,v2=tri_nodes
    denom=np.cross(v1-v0, v2-v0)
    for _ in range(n_samples):
        x=rng.uniform([xmin,ymin],[xmax,ymax])
        w1=np.cross(x-v0, v2-v0)/denom
        w2=np.cross(v1-v0, x-v0)/denom
        w0=1-w1-w2
        if (w0>=0)&(w1>=0)&(w2>=0):
            if level_set(x)<0:
                count+=1
                val+=f(x)
    if count==0:
        return 0.0
    area_inside=area_elem*count/n_samples
    return area_inside*(val/count)
